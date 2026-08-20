#!/usr/bin/env python3
from __future__ import annotations

import os
import pickle
import sys
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin
from typing import Any, Deque, Dict, List, Optional, Tuple

import rclpy
import numpy as np
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Quaternion
from hunav_msgs.msg import Agent, Agents
from hunav_msgs.srv import ComputeAgents
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


@dataclass(frozen=True)
class TrackPoint:
    stamp: float
    x: float
    y: float
    vx: float
    vy: float


class ConstantVelocityPredictor:
    """Small online predictor used until the SPU-BERT runtime is available."""

    def __init__(self, pred_len: int, pred_dt: float) -> None:
        self.pred_len = pred_len
        self.pred_dt = pred_dt

    def predict(self, track: List[TrackPoint], fallback_vx: float, fallback_vy: float) -> List[Tuple[float, float]]:
        if track:
            x0 = track[-1].x
            y0 = track[-1].y
        else:
            x0 = 0.0
            y0 = 0.0

        vx, vy = self._estimate_velocity(track, fallback_vx, fallback_vy)
        return [(x0 + vx * self.pred_dt * step, y0 + vy * self.pred_dt * step) for step in range(1, self.pred_len + 1)]

    @staticmethod
    def _estimate_velocity(track: List[TrackPoint], fallback_vx: float, fallback_vy: float) -> Tuple[float, float]:
        speed = hypot(fallback_vx, fallback_vy)
        if speed > 0.03:
            return fallback_vx, fallback_vy

        if len(track) < 2:
            return 0.0, 0.0

        prev = track[-2]
        last = track[-1]
        dt = max(last.stamp - prev.stamp, 1e-3)
        return (last.x - prev.x) / dt, (last.y - prev.y) / dt


class OccupancyMapProvider:
    """Load a ROS map YAML/PGM and produce target-centered scene patches."""

    def __init__(self, yaml_path: str, logger: Any) -> None:
        if not yaml_path:
            raise RuntimeError("map yaml path is empty")
        if not os.path.isfile(yaml_path):
            raise RuntimeError(f"map yaml does not exist: {yaml_path}")

        self.yaml_path = yaml_path
        self.logger = logger
        meta = self._load_map_yaml(yaml_path)
        image_path = str(meta.get("image", ""))
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(yaml_path), image_path)
        if not os.path.isfile(image_path):
            raise RuntimeError(f"map image does not exist: {image_path}")

        self.resolution = float(meta.get("resolution", 0.05))
        origin = meta.get("origin", [0.0, 0.0, 0.0])
        self.origin_x = float(origin[0])
        self.origin_y = float(origin[1])
        self.negate = int(meta.get("negate", 0))
        self.occupied_thresh = float(meta.get("occupied_thresh", 0.65))

        image = self._load_pgm(image_path)
        if self.negate:
            image = 255 - image
        occ_prob = (255.0 - image.astype(np.float32)) / 255.0
        self.occupied = occ_prob >= self.occupied_thresh
        self.height, self.width = self.occupied.shape
        self._inflated_occupancy_cache: Dict[int, np.ndarray] = {
            0: self.occupied,
        }
        logger.info(
            f"Loaded local occupancy map from {yaml_path}: {self.width}x{self.height}, "
            f"resolution={self.resolution:.3f}, origin=({self.origin_x:.2f}, {self.origin_y:.2f})"
        )

    @staticmethod
    def _load_map_yaml(path: str) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.split("#", 1)[0].strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if value.startswith("[") and value.endswith("]"):
                    meta[key] = [float(item.strip()) for item in value[1:-1].split(",") if item.strip()]
                elif key == "image":
                    meta[key] = value.strip("'\"")
                else:
                    try:
                        if any(ch in value for ch in (".", "e", "E")):
                            meta[key] = float(value)
                        else:
                            meta[key] = int(value)
                    except ValueError:
                        meta[key] = value.strip("'\"")
        return meta

    @staticmethod
    def _read_pgm_token(f) -> bytes:
        token = bytearray()
        while True:
            ch = f.read(1)
            if not ch:
                return bytes(token)
            if ch == b"#":
                f.readline()
                continue
            if ch.isspace():
                if token:
                    return bytes(token)
                continue
            token.extend(ch)

    @classmethod
    def _load_pgm(cls, path: str) -> np.ndarray:
        with open(path, "rb") as f:
            magic = cls._read_pgm_token(f)
            if magic not in (b"P2", b"P5"):
                raise RuntimeError(f"unsupported PGM format {magic!r} in {path}")
            width = int(cls._read_pgm_token(f))
            height = int(cls._read_pgm_token(f))
            maxval = int(cls._read_pgm_token(f))
            if maxval <= 0 or maxval > 255:
                raise RuntimeError(f"unsupported PGM maxval={maxval} in {path}")
            if magic == b"P2":
                values = [int(cls._read_pgm_token(f)) for _ in range(width * height)]
                return np.asarray(values, dtype=np.uint8).reshape((height, width))
            data = np.frombuffer(f.read(width * height), dtype=np.uint8)
            if data.size != width * height:
                raise RuntimeError(f"PGM data length mismatch in {path}")
            return data.reshape((height, width))

    def _world_to_pixel(self, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cols = np.floor((xs - self.origin_x) / self.resolution).astype(np.int64)
        rows_from_bottom = np.floor((ys - self.origin_y) / self.resolution).astype(np.int64)
        rows = self.height - 1 - rows_from_bottom
        valid = (0 <= cols) & (cols < self.width) & (0 <= rows) & (rows < self.height)
        return cols, rows, valid

    def occupancy_at_world(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        cols, rows, valid = self._world_to_pixel(xs, ys)
        values = np.ones(xs.shape, dtype=np.float32)
        values[valid] = self.occupied[rows[valid], cols[valid]].astype(np.float32)
        return values

    def point_occupied(self, x: float, y: float) -> bool:
        return bool(self.occupancy_at_world(np.asarray([x]), np.asarray([y]))[0] > 0.5)

    def _inflated_occupancy(self, radius: float) -> np.ndarray:
        radius = max(float(radius), 0.0)
        if radius <= 0.0:
            return self.occupied
        cache_key = int(round(radius * 1_000_000.0))
        cached = self._inflated_occupancy_cache.get(cache_key)
        if cached is not None:
            return cached

        # Mark a target cell blocked whenever its square can come within the
        # circular footprint radius of an occupied source cell. This catches
        # diagonal corner contact that the old four-offset probe missed.
        limit = int(np.ceil(radius / self.resolution)) + 1
        inflated = np.zeros_like(self.occupied, dtype=bool)
        for row_offset in range(-limit, limit + 1):
            for col_offset in range(-limit, limit + 1):
                separation_x = max(abs(col_offset) - 1, 0) * self.resolution
                separation_y = max(abs(row_offset) - 1, 0) * self.resolution
                if hypot(separation_x, separation_y) > radius + 1e-9:
                    continue

                source_row_start = max(0, -row_offset)
                source_row_end = min(self.height, self.height - row_offset)
                source_col_start = max(0, -col_offset)
                source_col_end = min(self.width, self.width - col_offset)
                target_row_start = source_row_start + row_offset
                target_row_end = source_row_end + row_offset
                target_col_start = source_col_start + col_offset
                target_col_end = source_col_end + col_offset
                inflated[
                    target_row_start:target_row_end,
                    target_col_start:target_col_end,
                ] |= self.occupied[
                    source_row_start:source_row_end,
                    source_col_start:source_col_end,
                ]

        self._inflated_occupancy_cache[cache_key] = inflated
        return inflated

    def point_occupied_with_radius(self, x: float, y: float, radius: float) -> bool:
        radius = max(float(radius), 0.0)
        if radius <= 0.0:
            return self.point_occupied(x, y)

        map_max_x = self.origin_x + self.width * self.resolution
        map_max_y = self.origin_y + self.height * self.resolution
        if (
            x - radius < self.origin_x
            or x + radius > map_max_x
            or y - radius < self.origin_y
            or y + radius > map_max_y
        ):
            return True
        cols, rows, valid = self._world_to_pixel(
            np.asarray([x], dtype=np.float32),
            np.asarray([y], dtype=np.float32),
        )
        if not bool(valid[0]):
            return True
        inflated = self._inflated_occupancy(radius)
        return bool(inflated[rows[0], cols[0]])

    def path_collision_cost(self, path: List[Tuple[float, float]], radius: float, weight: float) -> float:
        if not path:
            return 0.0
        cost = 0.0
        for px, py in path:
            if self.point_occupied_with_radius(float(px), float(py), radius):
                cost += weight
        return cost

    def scene_patches(
        self,
        *,
        trans: Tuple[float, float],
        theta: float,
        env_range: float,
        env_resol: float,
        patch_size: int,
        side_patches: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        side_cells = side_patches * patch_size
        min_local = -0.5 * side_cells * env_resol
        ids = np.arange(side_cells, dtype=np.float32)
        local_xs = min_local + ids * env_resol + 0.5 * env_resol
        local_ys = min_local + ids * env_resol + 0.5 * env_resol
        grid_x, grid_y = np.meshgrid(local_xs, local_ys)

        # Inverse of SPU-BERT target transform:
        # local = rotate(world + trans, theta), so world = rotate(local, -theta) - trans.
        world_x = grid_x * cos(theta) - grid_y * sin(theta) - trans[0]
        world_y = grid_x * sin(theta) + grid_y * cos(theta) - trans[1]
        cols, rows, valid = self._world_to_pixel(world_x, world_y)
        local_occ = np.zeros(world_x.shape, dtype=np.float32)
        known_occ = self.occupied[rows[valid], cols[valid]]
        local_occ[valid] = np.where(known_occ, 2.0, 1.0).astype(np.float32)

        patches: List[np.ndarray] = []
        attn_mask: List[float] = []
        for py in range(side_patches):
            for px in range(side_patches):
                patch = local_occ[
                    py * patch_size : (py + 1) * patch_size,
                    px * patch_size : (px + 1) * patch_size,
                ]
                patches.append(patch.astype(np.float32).reshape(-1))
                attn_mask.append(1.0 if np.any(patch > 0.0) else 0.0)
        return np.stack(patches, axis=0), np.asarray(attn_mask, dtype=np.float32)


class SPUBERTPredictor:
    """Runtime adapter for the official SPU-BERT model.

    SPU-BERT consumes target-centered, heading-aligned trajectories plus nearby
    pedestrian histories. This adapter mirrors the repository dataset encoding
    for online HuNavSim tracks and returns one selected future trajectory.
    """

    def __init__(
        self,
        *,
        repo_path: str,
        model_path: str,
        obs_len: int,
        pred_len: int,
        num_nbr: int,
        view_range: float,
        view_angle: float,
        social_range: float,
        hidden: int,
        layer: int,
        head: int,
        k_sample: int,
        d_sample: int,
        scene: bool,
        env_range: float,
        env_resol: float,
        patch_size: int,
        map_yaml_path: str,
        use_map_collision_filter: bool,
        map_collision_radius: float,
        map_collision_weight: float,
        use_cuda: bool,
        logger: Any,
    ) -> None:
        if not model_path:
            raise RuntimeError("spubert_model_path is empty")
        if not os.path.isfile(model_path):
            raise RuntimeError(f"SPU-BERT model file does not exist: {model_path}")
        if not os.path.isdir(repo_path):
            raise RuntimeError(f"SPU-BERT repo path does not exist: {repo_path}")

        repo_parent = os.path.dirname(repo_path.rstrip(os.sep))
        if repo_parent not in sys.path:
            sys.path.insert(0, repo_parent)

        try:
            import torch
            from SPUBERT.model.spubert import SPUBERTConfig, SPUBERTMGPConfig, SPUBERTModel, SPUBERTTGPConfig
        except Exception as exc:
            raise RuntimeError(f"SPU-BERT dependencies are not importable: {exc}") from exc

        self.torch = torch
        self.logger = logger
        torch.set_num_threads(1)
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.num_nbr = num_nbr
        self.view_range = view_range
        self.view_angle = view_angle
        self.social_range = social_range
        self.k_sample = k_sample
        self.d_sample = d_sample
        self.scene = scene
        self.patch_size = patch_size
        self.env_range = env_range
        self.env_resol = env_resol
        self.map_collision_radius = map_collision_radius
        self.map_collision_weight = map_collision_weight
        self.occupancy_map: Optional[OccupancyMapProvider] = None
        self._logged_scene_patch = False
        self.seq_len = obs_len + pred_len
        self.seq_input_len = (obs_len + 1) * (num_nbr + 1) + pred_len
        self.pad_xy = [-view_range, -view_range]
        self.mask_xy = [view_range, view_range]
        self.sep_xy = [view_range, -view_range]
        self.sot_xy = [-view_range, view_range]
        map_length = int((2.0 * env_range) / env_resol)
        if map_length % 2 != 0:
            map_length += 1
        if map_length % patch_size == 0:
            side_patches = map_length // patch_size
        else:
            side_patches = (map_length // patch_size) + 2
        self.side_patches = side_patches
        self.num_patch = side_patches * side_patches
        if map_yaml_path:
            try:
                self.occupancy_map = OccupancyMapProvider(map_yaml_path, logger)
            except Exception as exc:
                logger.warn(f"Local map crop disabled: {exc}")
        self.use_map_collision_filter = bool(use_map_collision_filter and self.occupancy_map is not None)

        tgp_cfg = SPUBERTTGPConfig(
            hidden_size=hidden,
            num_layer=layer,
            num_head=head,
            obs_len=obs_len,
            pred_len=pred_len,
            num_nbr=num_nbr,
            scene=scene,
            num_patch=self.num_patch,
            patch_size=patch_size,
            view_range=view_range,
            view_angle=view_angle,
            social_range=social_range,
        )
        mgp_cfg = SPUBERTMGPConfig(
            hidden_size=hidden,
            num_layer=layer,
            num_head=head,
            obs_len=obs_len,
            pred_len=pred_len,
            num_nbr=num_nbr,
            scene=scene,
            num_patch=self.num_patch,
            patch_size=patch_size,
            view_range=view_range,
            view_angle=view_angle,
            social_range=social_range,
            k_sample=k_sample,
            goal_latent_size=32,
        )
        cfg = SPUBERTConfig(traj_cfgs=tgp_cfg, goal_cfgs=mgp_cfg)
        self.model = SPUBERTModel(tgp_cfg, mgp_cfg, cfg)
        state = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded SPU-BERT predictor from {model_path} on {self.device}")

    def predict(
        self,
        *,
        agent_id: int,
        tracks: Dict[int, Deque[TrackPoint]],
        goal: Optional[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        candidates = self.predict_candidates(agent_id=agent_id, tracks=tracks)
        if not candidates:
            return []
        return candidates[self._select_candidate_by_goal(candidates, goal)]

    def predict_candidates(
        self,
        *,
        agent_id: int,
        tracks: Dict[int, Deque[TrackPoint]],
    ) -> List[List[Tuple[float, float]]]:
        sample, trans, theta = self._build_sample(agent_id, tracks)
        batch = {key: value.unsqueeze(0).to(self.device) for key, value in sample.items()}
        env_kwargs = self._scene_batch(trans, theta) if self.scene else {}

        with self.torch.no_grad():
            outputs = self.model.inference(
                mgp_spatial_ids=batch["mgp_spatial_ids"],
                mgp_temporal_ids=batch["mgp_temporal_ids"],
                mgp_segment_ids=batch["mgp_segment_ids"],
                mgp_attn_mask=batch["mgp_attn_mask"],
                tgp_temporal_ids=batch["tgp_temporal_ids"],
                tgp_segment_ids=batch["tgp_segment_ids"],
                tgp_attn_mask=batch["tgp_attn_mask"],
                **env_kwargs,
                d_sample=self.d_sample,
            )

        pred = outputs["pred_trajs"][0]
        paths = pred.detach().cpu().tolist()
        return [[self._to_world(x, y, trans, theta) for x, y in path] for path in paths]

    def _build_sample(
        self,
        agent_id: int,
        tracks: Dict[int, Deque[TrackPoint]],
    ) -> Tuple[Dict[str, Any], Tuple[float, float], float]:
        target = self._track_array(tracks[agent_id])
        trans = (-target[-1][0], -target[-1][1])
        theta = self._heading_theta(target)

        trajs = [self._transform_track(target, trans, theta)]
        neighbors: List[List[Tuple[float, float]]] = []
        for other_id, other_track in tracks.items():
            if other_id == agent_id or len(other_track) == 0:
                continue
            local = self._transform_track(self._track_array(other_track), trans, theta)
            dist = hypot(local[-1][0], local[-1][1])
            angle = abs(atan2(local[-1][1], local[-1][0]))
            if dist <= self.view_range and (dist <= self.social_range or angle <= self.view_angle * 0.5):
                neighbors.append(local)

        neighbors.sort(key=lambda traj: hypot(traj[-1][0], traj[-1][1]))
        trajs.extend(neighbors[: self.num_nbr])
        return self._encode(trajs), trans, theta

    def _track_array(self, track: Deque[TrackPoint]) -> List[Tuple[float, float]]:
        points = [(float(p.x), float(p.y)) for p in list(track)[-self.obs_len :]]
        if not points:
            points = [(0.0, 0.0)]
        while len(points) < self.obs_len:
            points.insert(0, points[0])
        return points

    @staticmethod
    def _heading_theta(points: List[Tuple[float, float]]) -> float:
        if len(points) < 2:
            return 0.0
        dx = points[-1][0] - points[-2][0]
        dy = points[-1][1] - points[-2][1]
        return atan2(dy, dx) if hypot(dx, dy) > 1e-4 else 0.0

    @staticmethod
    def _rotate(x: float, y: float, theta: float) -> Tuple[float, float]:
        return x * cos(theta) + y * sin(theta), -x * sin(theta) + y * cos(theta)

    def _transform_track(
        self,
        points: List[Tuple[float, float]],
        trans: Tuple[float, float],
        theta: float,
    ) -> List[Tuple[float, float]]:
        return [self._rotate(x + trans[0], y + trans[1], theta) for x, y in points]

    def _to_world(self, x: float, y: float, trans: Tuple[float, float], theta: float) -> Tuple[float, float]:
        rx, ry = self._rotate(x, y, -theta)
        return rx - trans[0], ry - trans[1]

    def _encode(self, trajs: List[List[Tuple[float, float]]]) -> Dict[str, Any]:
        torch = self.torch
        target = trajs[0]
        mgp_spatial_ids = [self.sot_xy] + target + [self.pad_xy] * (self.pred_len - 1) + [self.mask_xy]
        mgp_segment_ids = [1] * (self.obs_len + 2) + [0] * (self.pred_len - 1)
        mgp_temporal_ids = [0] + list(range(1, self.obs_len + 1)) + [0] * (self.pred_len - 1) + [self.seq_len]
        mgp_attn_mask = [1] * (self.obs_len + 1) + [0] * (self.pred_len - 1) + [1]

        tgp_spatial_ids = [self.sot_xy] + target + [self.mask_xy] * self.pred_len
        tgp_segment_ids = [1] * (self.obs_len + self.pred_len + 1)
        tgp_temporal_ids = [0] + list(range(1, self.seq_len + 1))
        tgp_attn_mask = [1] * (self.obs_len + self.pred_len + 1)

        nbr_spatial: List[List[float]] = []
        nbr_segment: List[int] = []
        nbr_temporal: List[int] = []
        nbr_attn: List[int] = []
        for nbr_idx, nbr in enumerate(trajs[1:], start=2):
            nbr_spatial += [self.sep_xy] + nbr
            nbr_segment += [nbr_idx] * (self.obs_len + 1)
            nbr_temporal += [0] + list(range(1, self.obs_len + 1))
            nbr_attn += [1] * (self.obs_len + 1)

        ext_len = self.seq_input_len - len(mgp_spatial_ids) - len(nbr_spatial)
        if ext_len < 0:
            nbr_spatial = nbr_spatial[:ext_len]
            nbr_segment = nbr_segment[:ext_len]
            nbr_temporal = nbr_temporal[:ext_len]
            nbr_attn = nbr_attn[:ext_len]
            ext_len = 0

        nbr_spatial += [self.pad_xy] * ext_len
        nbr_segment += [0] * ext_len
        nbr_temporal += [0] * ext_len
        nbr_attn += [0] * ext_len

        return {
            "mgp_spatial_ids": torch.tensor(mgp_spatial_ids + nbr_spatial, dtype=torch.float),
            "mgp_segment_ids": torch.tensor(mgp_segment_ids + nbr_segment, dtype=torch.long),
            "mgp_temporal_ids": torch.tensor(mgp_temporal_ids + nbr_temporal, dtype=torch.long),
            "mgp_attn_mask": torch.tensor(mgp_attn_mask + nbr_attn, dtype=torch.float),
            "tgp_spatial_ids": torch.tensor(tgp_spatial_ids + nbr_spatial, dtype=torch.float),
            "tgp_segment_ids": torch.tensor(tgp_segment_ids + nbr_segment, dtype=torch.long),
            "tgp_temporal_ids": torch.tensor(tgp_temporal_ids + nbr_temporal, dtype=torch.long),
            "tgp_attn_mask": torch.tensor(tgp_attn_mask + nbr_attn, dtype=torch.float),
            "mgp_temporal_ids": torch.tensor(mgp_temporal_ids + nbr_temporal, dtype=torch.long),
            "mgp_attn_mask": torch.tensor(mgp_attn_mask + nbr_attn, dtype=torch.float),
            "tgp_spatial_ids": torch.tensor(tgp_spatial_ids + nbr_spatial, dtype=torch.float),
            "tgp_segment_ids": torch.tensor(tgp_segment_ids + nbr_segment, dtype=torch.long),
            "tgp_temporal_ids": torch.tensor(tgp_temporal_ids + nbr_temporal, dtype=torch.long),
            "tgp_attn_mask": torch.tensor(tgp_attn_mask + nbr_attn, dtype=torch.float),
        }

    def _empty_scene_batch(self) -> Dict[str, Any]:
        torch = self.torch
        return {
            "env_spatial_ids": torch.zeros((1, self.num_patch, self.patch_size * self.patch_size), dtype=torch.float, device=self.device),
            "env_segment_ids": torch.arange(
                self.num_nbr + 1,
                self.num_nbr + self.num_patch + 1,
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0),
            "env_temporal_ids": torch.full((1, self.num_patch), self.obs_len, dtype=torch.long, device=self.device),
            "env_attn_mask": torch.zeros((1, self.num_patch), dtype=torch.float, device=self.device),
        }

    def _scene_batch(self, trans: Tuple[float, float], theta: float) -> Dict[str, Any]:
        if self.occupancy_map is None:
            return self._empty_scene_batch()

        torch = self.torch
        patches, attn_mask = self.occupancy_map.scene_patches(
            trans=trans,
            theta=theta,
            env_range=self.env_range,
            env_resol=self.env_resol,
            patch_size=self.patch_size,
            side_patches=self.side_patches,
        )
        if not self._logged_scene_patch:
            self._logged_scene_patch = True
            occupied_cells = int(np.count_nonzero(patches > 1.5))
            active_patches = int(np.count_nonzero(attn_mask))
            self.logger.info(
                "SPU-BERT local map crop active: "
                f"env_spatial_ids=(1, {patches.shape[0]}, {patches.shape[1]}), "
                f"active_patches={active_patches}, occupied_cells={occupied_cells}"
            )
        return {
            "env_spatial_ids": torch.tensor(patches, dtype=torch.float, device=self.device).unsqueeze(0),
            "env_segment_ids": torch.arange(
                self.num_nbr + 1,
                self.num_nbr + self.num_patch + 1,
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0),
            "env_temporal_ids": torch.full((1, self.num_patch), self.obs_len, dtype=torch.long, device=self.device),
            "env_attn_mask": torch.tensor(attn_mask, dtype=torch.float, device=self.device).unsqueeze(0),
        }

    def _select_candidate_by_goal(self, candidates: List[List[Tuple[float, float]]], goal: Optional[Tuple[float, float]]) -> int:
        if goal is None:
            return 0
        best_idx = 0
        best_dist = float("inf")
        for idx, path in enumerate(candidates):
            if not path:
                continue
            dist = hypot(path[-1][0] - goal[0], path[-1][1] - goal[1])
            if self.use_map_collision_filter and self.occupancy_map is not None:
                dist += self.occupancy_map.path_collision_cost(
                    path,
                    radius=self.map_collision_radius,
                    weight=self.map_collision_weight,
                )
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    def map_collision_cost(self, path: List[Tuple[float, float]]) -> float:
        if not self.use_map_collision_filter or self.occupancy_map is None:
            return 0.0
        return self.occupancy_map.path_collision_cost(
            path,
            radius=self.map_collision_radius,
            weight=self.map_collision_weight,
        )


class MOAIRefactoredSPUBERTPredictor(SPUBERTPredictor):
    """Adapter for the locally trained moai_social_bert_refactored checkpoint."""

    def __init__(
        self,
        *,
        repo_path: str,
        model_path: str,
        obs_len: int,
        pred_len: int,
        num_nbr: int,
        view_range: float,
        view_angle: float,
        social_range: float,
        hidden: int,
        layer: int,
        head: int,
        k_sample: int,
        d_sample: int,
        scene: bool,
        env_range: float,
        env_resol: float,
        patch_size: int,
        map_yaml_path: str,
        use_map_collision_filter: bool,
        map_collision_radius: float,
        map_collision_weight: float,
        use_cuda: bool,
        logger: Any,
    ) -> None:
        if not model_path:
            raise RuntimeError("spubert_model_path is empty")
        if not os.path.isfile(model_path):
            raise RuntimeError(f"MOAI SPU-BERT model file does not exist: {model_path}")
        if not os.path.isdir(repo_path):
            raise RuntimeError(f"MOAI refactored repo path does not exist: {repo_path}")

        spubert_root = os.path.join(repo_path, "SPU-BERT")
        for path_entry in (repo_path, spubert_root):
            if path_entry not in sys.path:
                sys.path.insert(0, path_entry)

        try:
            import torch
            from spubert.model import SBertPlusFTConfig, SBertPlusFTModel, SBertPlusMGPConfig, SBertPlusTGPConfig
        except Exception as exc:
            raise RuntimeError(f"MOAI SPU-BERT dependencies are not importable: {exc}") from exc

        self.torch = torch
        self.logger = logger
        torch.set_num_threads(1)
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.num_nbr = num_nbr
        self.view_range = view_range
        self.view_angle = view_angle
        self.social_range = social_range
        self.k_sample = k_sample
        self.d_sample = d_sample
        self.scene = scene
        self.patch_size = patch_size
        self.env_range = env_range
        self.env_resol = env_resol
        self.map_collision_radius = map_collision_radius
        self.map_collision_weight = map_collision_weight
        self.occupancy_map: Optional[OccupancyMapProvider] = None
        self._logged_scene_patch = False
        self.seq_len = obs_len + pred_len
        self.seq_input_len = (obs_len + 1) * (num_nbr + 1) + pred_len
        self.pad_xy = [-view_range, -view_range]
        self.mask_xy = [view_range, view_range]
        self.sep_xy = [view_range, -view_range]
        self.sot_xy = [-view_range, view_range]
        map_length = int((2.0 * env_range) / env_resol)
        if map_length % 2 != 0:
            map_length += 1
        # The local moai_social_bert_refactored training code used ceil(),
        # while the upstream SPU-BERT repo pads by two extra patches.
        side_patches = (map_length + patch_size - 1) // patch_size
        self.side_patches = side_patches
        self.num_patch = side_patches * side_patches
        if map_yaml_path:
            try:
                self.occupancy_map = OccupancyMapProvider(map_yaml_path, logger)
            except Exception as exc:
                logger.warn(f"Local map crop disabled: {exc}")
        self.use_map_collision_filter = bool(use_map_collision_filter and self.occupancy_map is not None)

        common_cfg = {
            "hidden_size": hidden,
            "num_layer": layer,
            "num_head": head,
            "obs_len": obs_len,
            "pred_len": pred_len,
            "num_nbr": num_nbr,
            "scene": scene,
            "num_patch": self.num_patch,
            "patch_size": patch_size,
            "view_range": view_range,
            "view_angle": view_angle,
            "social_range": social_range,
            "dropout_prob": 0.1,
            "act_fn": "relu",
            "backbone_type": "bert",
            "binary_scene": False,
        }
        tgp_cfg = SBertPlusTGPConfig(col_weight=10, traj_weight=1.0, **common_cfg)
        mgp_cfg = SBertPlusMGPConfig(
            k_sample=k_sample,
            goal_hidden_size=64,
            goal_latent_size=32,
            kld_weight=10,
            col_weight=10,
            goal_weight=1.0,
            cvae_sigma=1.0,
            normal=False,
            **common_cfg,
        )
        self.model = SBertPlusFTModel(tgp_cfg, mgp_cfg, SBertPlusFTConfig(tgp_cfg, mgp_cfg, share=False))
        state = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded MOAI refactored SPU-BERT predictor from {model_path} on {self.device}")


class SocialBertComputeAgentsNode(Node):
    def __init__(self) -> None:
        super().__init__("social_bert_compute_agents")

        self.obs_len = int(self.declare_parameter("obs_len", 8).value)
        self.pred_len = int(self.declare_parameter("pred_len", 12).value)
        self.prediction_dt = float(self.declare_parameter("prediction_dt", 0.4).value)
        self.fallback_dt = float(self.declare_parameter("fallback_dt", 0.05).value)
        self.max_speed = float(self.declare_parameter("max_speed", 1.8).value)
        self.max_accel = float(self.declare_parameter("max_accel", 1.8).value)
        self.lookahead_step = int(self.declare_parameter("lookahead_step", 6).value)
        self.min_goal_speed = float(self.declare_parameter("min_goal_speed", 0.9).value)
        self.goal_velocity_blend = float(self.declare_parameter("goal_velocity_blend", 0.7).value)
        self.goal_arrival_distance = float(self.declare_parameter("goal_arrival_distance", 1.5).value)
        self.goal_arrival_min_speed = float(self.declare_parameter("goal_arrival_min_speed", 0.08).value)
        self.robot_personal_space = float(self.declare_parameter("robot_personal_space", 0.9).value)
        self.robot_collision_buffer = float(self.declare_parameter("robot_collision_buffer", 0.35).value)
        self.robot_hard_collision_guard = self._as_bool(
            self.declare_parameter("robot_hard_collision_guard", True).value
        )
        self.agent_personal_space = float(self.declare_parameter("agent_personal_space", 1.25).value)
        self.agent_avoidance_gain = float(self.declare_parameter("agent_avoidance_gain", 0.9).value)
        self.obstacle_avoidance_distance = float(self.declare_parameter("obstacle_avoidance_distance", 1.35).value)
        self.obstacle_avoidance_gain = float(self.declare_parameter("obstacle_avoidance_gain", 0.45).value)
        self.obstacle_collision_buffer = float(self.declare_parameter("obstacle_collision_buffer", 0.42).value)
        self.obstacle_lateral_speed_ratio = float(
            self.declare_parameter("obstacle_lateral_speed_ratio", 0.75).value
        )
        self.velocity_smoothing_alpha = float(self.declare_parameter("velocity_smoothing_alpha", 0.35).value)
        self.collision_buffer = float(self.declare_parameter("collision_buffer", 0.16).value)
        self.max_lateral_speed_ratio = float(self.declare_parameter("max_lateral_speed_ratio", 0.25).value)
        self.max_yaw_rate = float(self.declare_parameter("max_yaw_rate", 1.0).value)
        self.predictor_mode = str(self.declare_parameter("predictor_mode", "constant_velocity").value)
        self.spubert_repo_path = str(
            self.declare_parameter("spubert_repo_path", "/home/hunav_gz_classic_ws/src/SPUBERT").value
        )
        self.moai_spubert_repo_path = str(
            self.declare_parameter(
                "moai_spubert_repo_path",
                "/home/hunav_gz_classic_ws/src/moai_social_bert_refactored",
            ).value
        )
        self.spubert_model_path = str(self.declare_parameter("spubert_model_path", "").value)
        self.spubert_cuda = self._as_bool(self.declare_parameter("spubert_cuda", False).value)
        self.spubert_hidden = int(self.declare_parameter("spubert_hidden", 256).value)
        self.spubert_layer = int(self.declare_parameter("spubert_layer", 4).value)
        self.spubert_head = int(self.declare_parameter("spubert_head", 4).value)
        self.spubert_num_nbr = int(self.declare_parameter("spubert_num_nbr", 4).value)
        self.spubert_k_sample = int(self.declare_parameter("spubert_k_sample", 20).value)
        self.spubert_d_sample = int(self.declare_parameter("spubert_d_sample", 20).value)
        self.spubert_scene = self._as_bool(self.declare_parameter("spubert_scene", True).value)
        self.spubert_env_range = float(self.declare_parameter("spubert_env_range", 10.0).value)
        self.spubert_env_resol = float(self.declare_parameter("spubert_env_resol", 0.2).value)
        self.spubert_patch_size = int(self.declare_parameter("spubert_patch_size", 16).value)
        self.spubert_map_yaml_path = str(self.declare_parameter("spubert_map_yaml_path", "").value)
        self.spubert_use_map_collision_filter = self._as_bool(
            self.declare_parameter("spubert_use_map_collision_filter", True).value
        )
        self.spubert_map_collision_radius = float(self.declare_parameter("spubert_map_collision_radius", 0.35).value)
        self.spubert_map_collision_weight = float(self.declare_parameter("spubert_map_collision_weight", 100.0).value)
        self.spubert_candidate_selection_mode = str(
            self.declare_parameter("spubert_candidate_selection_mode", "guidance_point").value
        ).strip().lower()
        self.guidance_point_radius = float(self.declare_parameter("guidance_point_radius", 8.0).value)
        self.spubert_view_range = float(self.declare_parameter("spubert_view_range", 20.0).value)
        self.spubert_view_angle = float(self.declare_parameter("spubert_view_angle", 2.09).value)
        self.spubert_social_range = float(self.declare_parameter("spubert_social_range", 2.0).value)
        self.spubert_cache_ttl = float(self.declare_parameter("spubert_cache_ttl", 1.5).value)
        self.spubert_refresh_dt = float(self.declare_parameter("spubert_refresh_dt", self.prediction_dt).value)
        self.spubert_sync_on_cache_miss = self._as_bool(
            self.declare_parameter("spubert_sync_on_cache_miss", True).value
        )
        self.save_training_pkl = self._as_bool(self.declare_parameter("save_training_pkl", False).value)
        self.training_pkl_path = str(
            self.declare_parameter("training_pkl_path", "/tmp/moai_gazebo_all_trajs.pkl").value
        )
        self.training_record_dt = float(self.declare_parameter("training_record_dt", self.prediction_dt).value)
        self.training_sample_stride = int(self.declare_parameter("training_sample_stride", 1).value)
        self.training_flush_every = int(self.declare_parameter("training_flush_every", 50).value)
        self.training_max_samples = int(self.declare_parameter("training_max_samples", 0).value)
        self.debug_focus_agent_id = int(self.declare_parameter("debug_focus_agent_id", 1).value)
        self.debug_focus_agent_only = self._as_bool(
            self.declare_parameter("debug_focus_agent_only", False).value
        )
        self.debug_guidance_only = self._as_bool(
            self.declare_parameter("debug_guidance_only", False).value
        )
        
        self.debug_show_all_candidate_paths = self._as_bool(
            self.declare_parameter("debug_show_all_candidate_paths", False).value
        )
        self.debug_show_all_model_io = self._as_bool(self.declare_parameter("debug_show_all_model_io", False).value)

        self._tracks: Dict[int, Deque[TrackPoint]] = defaultdict(lambda: deque(maxlen=self.obs_len))
        self._training_frames: Deque[Dict[str, Any]] = deque(maxlen=self.obs_len + self.pred_len)
        self._training_samples: List[np.ndarray] = []
        self._training_sample_meta: List[Dict[str, Any]] = []
        self._last_training_stamp: Optional[float] = None
        self._training_frame_count = 0
        self._last_flushed_sample_count = 0
        self._last_stamp: float | None = None
        self._last_cmd_vel: Dict[int, Tuple[float, float]] = {}
        self._last_avoid_side: Dict[int, float] = {}
        self._last_yaw: Dict[int, float] = {}
        self._fallback_predictor = ConstantVelocityPredictor(self.pred_len, self.prediction_dt)
        self._spubert_predictor: Optional[SPUBERTPredictor] = None
        self._prediction_cache: Dict[int, Tuple[float, List[List[Tuple[float, float]]]]] = {}
        self._pending_predictions: set[int] = set()
        self._cache_lock = threading.Lock()
        self._inference_lock = threading.Lock()

        self._srv = self.create_service(ComputeAgents, "compute_agents", self._on_compute_agents)
        self._human_pub = self.create_publisher(Agents, "human_states", 1)
        self._robot_pub = self.create_publisher(Agent, "robot_states", 1)
        self._marker_pub = self.create_publisher(MarkerArray, "moai/social_bert_predicted_paths", 1)

        if self.predictor_mode == "spubert":
            self._spubert_predictor = self._try_create_spubert_predictor()
        elif self.predictor_mode == "moai_spubert":
            self._spubert_predictor = self._try_create_moai_spubert_predictor()
        elif self.predictor_mode != "constant_velocity":
            self.get_logger().warn(f"Unknown predictor_mode={self.predictor_mode}; using constant-velocity fallback.")

        active_predictor = "SPU-BERT" if self._spubert_predictor is not None else "constant-velocity fallback"
        self.get_logger().info(f"Serving /compute_agents with MOAI Social-BERT bridge ({active_predictor})")
        self.get_logger().info(
            "Pedestrian candidate guidance: "
            f"mode={self.spubert_candidate_selection_mode}, radius={self.guidance_point_radius:.2f}m"
        )
        if self.save_training_pkl:
            self.get_logger().info(
                f"Recording MOAI all_trajs training pkl to {self.training_pkl_path} "
                f"at dt={self.training_record_dt:.3f}s"
            )

    def _try_create_spubert_predictor(self) -> Optional[SPUBERTPredictor]:
        try:
            return SPUBERTPredictor(
                repo_path=self.spubert_repo_path,
                model_path=self.spubert_model_path,
                obs_len=self.obs_len,
                pred_len=self.pred_len,
                num_nbr=self.spubert_num_nbr,
                view_range=self.spubert_view_range,
                view_angle=self.spubert_view_angle,
                social_range=self.spubert_social_range,
                hidden=self.spubert_hidden,
                layer=self.spubert_layer,
                head=self.spubert_head,
                k_sample=self.spubert_k_sample,
                d_sample=self.spubert_d_sample,
                scene=self.spubert_scene,
                env_range=self.spubert_env_range,
                env_resol=self.spubert_env_resol,
                patch_size=self.spubert_patch_size,
                map_yaml_path=self.spubert_map_yaml_path,
                use_map_collision_filter=self.spubert_use_map_collision_filter,
                map_collision_radius=self.spubert_map_collision_radius,
                map_collision_weight=self.spubert_map_collision_weight,
                use_cuda=self.spubert_cuda,
                logger=self.get_logger(),
            )
        except Exception as exc:
            self.get_logger().warn(f"SPU-BERT predictor unavailable: {exc}; using constant-velocity fallback.")
            return None

    def _try_create_moai_spubert_predictor(self) -> Optional[SPUBERTPredictor]:
        try:
            return MOAIRefactoredSPUBERTPredictor(
                repo_path=self.moai_spubert_repo_path,
                model_path=self.spubert_model_path,
                obs_len=self.obs_len,
                pred_len=self.pred_len,
                num_nbr=self.spubert_num_nbr,
                view_range=self.spubert_view_range,
                view_angle=self.spubert_view_angle,
                social_range=self.spubert_social_range,
                hidden=self.spubert_hidden,
                layer=self.spubert_layer,
                head=self.spubert_head,
                k_sample=self.spubert_k_sample,
                d_sample=self.spubert_d_sample,
                scene=self.spubert_scene,
                env_range=self.spubert_env_range,
                env_resol=self.spubert_env_resol,
                patch_size=self.spubert_patch_size,
                map_yaml_path=self.spubert_map_yaml_path,
                use_map_collision_filter=self.spubert_use_map_collision_filter,
                map_collision_radius=self.spubert_map_collision_radius,
                map_collision_weight=self.spubert_map_collision_weight,
                use_cuda=self.spubert_cuda,
                logger=self.get_logger(),
            )
        except Exception as exc:
            self.get_logger().warn(f"MOAI refactored SPU-BERT predictor unavailable: {exc}; using constant-velocity fallback.")
            return None

    def _on_compute_agents(self, request: ComputeAgents.Request, response: ComputeAgents.Response) -> ComputeAgents.Response:
        stamp = self._stamp_to_float(request.current_agents)
        dt = self._compute_dt(stamp)

        updated = Agents()
        updated.header = request.current_agents.header
        robot = request.robot
        self._robot_pub.publish(robot)
        markers = MarkerArray()
        markers.markers.append(self._delete_all_marker(request.current_agents.header.frame_id or "map"))

        self._record_training_frame(request.current_agents, stamp)

        marker_id = 1
        for agent in request.current_agents.agents:
            self._remember(agent, stamp)
            next_agent, predicted, guidance_xy = self._step_agent(agent, robot, request.current_agents.agents, dt)
            updated.agents.append(next_agent)
            if self.debug_focus_agent_only and not self._is_debug_focus_agent(agent):
                marker_id += 200
                continue
            candidate_paths = self._get_cached_candidate_paths(int(agent.id), stamp)
            if not self.debug_show_all_candidate_paths and not self._is_debug_focus_agent(agent):
                candidate_paths = []
            markers.markers.extend(
                self._path_markers(marker_id, agent, predicted, candidate_paths, guidance_xy, updated.header.frame_id or "map")
            )
            marker_id += 200

        response.updated_agents = updated
        self._human_pub.publish(updated)
        self._marker_pub.publish(markers)
        return response

    def destroy_node(self) -> bool:
        self._flush_training_pkl(force=True)
        return super().destroy_node()

    def _remember(self, agent: Agent, stamp: float) -> None:
        self._tracks[agent.id].append(
            TrackPoint(
                stamp=stamp,
                x=float(agent.position.position.x),
                y=float(agent.position.position.y),
                vx=float(agent.velocity.linear.x),
                vy=float(agent.velocity.linear.y),
            )
        )

    def _record_training_frame(self, agents_msg: Agents, stamp: float) -> None:
        if not self.save_training_pkl:
            return

        if self._last_training_stamp is not None and stamp < self._last_training_stamp:
            self._training_frames.clear()
            self._last_training_stamp = None

        record_dt = max(self.training_record_dt, 1e-3)
        if self._last_training_stamp is not None and stamp - self._last_training_stamp < record_dt:
            return

        agents = {
            int(agent.id): (
                float(agent.position.position.x),
                float(agent.position.position.y),
            )
            for agent in agents_msg.agents
        }
        goals = {
            int(agent.id): (
                float(agent.goals[0].position.x),
                float(agent.goals[0].position.y),
            )
            for agent in agents_msg.agents
            if agent.goals
        }
        if not agents:
            return

        self._last_training_stamp = stamp
        self._training_frame_count += 1
        self._training_frames.append(
            {
                "stamp": stamp,
                "frame": self._training_frame_count,
                "agents": agents,
                "goals": goals,
            }
        )

        seq_len = self.obs_len + self.pred_len
        stride = max(self.training_sample_stride, 1)
        if len(self._training_frames) == seq_len and (self._training_frame_count - seq_len) % stride == 0:
            self._append_training_samples_from_window()

        self._flush_training_pkl(force=False)

    def _append_training_samples_from_window(self) -> None:
        if self.training_max_samples > 0 and len(self._training_samples) >= self.training_max_samples:
            return

        frames = list(self._training_frames)
        seq_len = self.obs_len + self.pred_len
        target_ids = sorted(
            {
                agent_id
                for frame in frames
                for agent_id in frame["agents"].keys()
            }
        )

        for target_id in target_ids:
            if self.training_max_samples > 0 and len(self._training_samples) >= self.training_max_samples:
                break

            if not all(target_id in frame["agents"] for frame in frames):
                continue

            neighbor_ids = [
                agent_id
                for agent_id in target_ids
                if agent_id != target_id and all(agent_id in frame["agents"] for frame in frames[: self.obs_len])
            ]
            agent_ids = [target_id] + neighbor_ids
            trajs = np.full((len(agent_ids), seq_len, 2), np.nan, dtype=np.float32)

            for row, agent_id in enumerate(agent_ids):
                limit = seq_len if row == 0 else self.obs_len
                for t_idx, frame in enumerate(frames[:limit]):
                    pos = frame["agents"].get(agent_id)
                    if pos is None:
                        continue
                    trajs[row, t_idx, 0] = pos[0]
                    trajs[row, t_idx, 1] = pos[1]

            if np.any(np.isnan(trajs[0, :, 0])):
                continue

            obs_end = frames[self.obs_len - 1]
            current_xy = trajs[0, self.obs_len - 1].astype(np.float32)
            final_goal_xy = obs_end.get("goals", {}).get(target_id)
            if final_goal_xy is None:
                final_goal_xy = tuple(trajs[0, -1].astype(np.float32))
            guidance_xy = self._guidance_point_from_xy(current_xy, final_goal_xy)

            self._training_samples.append(trajs)
            self._training_sample_meta.append(
                {
                    "target_id": target_id,
                    "agent_ids": agent_ids,
                    "start_stamp": frames[0]["stamp"],
                    "end_stamp": frames[-1]["stamp"],
                    "start_frame": frames[0]["frame"],
                    "end_frame": frames[-1]["frame"],
                    "final_goal": [float(final_goal_xy[0]), float(final_goal_xy[1])],
                    "guidance_point": [float(guidance_xy[0]), float(guidance_xy[1])],
                    "guidance_radius": float(max(0.0, self.guidance_point_radius)),
                    "guidance_policy": "circle_line_intersection_to_final_goal",
                    "candidate_selection_mode": self.spubert_candidate_selection_mode,
                }
            )

    def _flush_training_pkl(self, *, force: bool) -> None:
        if not self.save_training_pkl:
            return
        if not force and self.training_flush_every <= 0:
            return
        if not force and len(self._training_samples) - self._last_flushed_sample_count < self.training_flush_every:
            return

        output_dir = os.path.dirname(self.training_pkl_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        payload = {
            "all_trajs": self._training_samples,
            "all_scenes": [0] * len(self._training_samples),
            "scales": [1.0],
            "sample_meta": self._training_sample_meta,
            "metadata": {
                "format": "moai_all_trajs_v1",
                "source": "hunavsim_moai_bridge",
                "obs_len": self.obs_len,
                "pred_len": self.pred_len,
                "seq_len": self.obs_len + self.pred_len,
                "dt": max(self.training_record_dt, 1e-3),
                "neighbor_future": "nan",
                "guidance_policy": "circle_line_intersection_to_final_goal",
                "guidance_radius": float(max(0.0, self.guidance_point_radius)),
                "description": "Each all_trajs item has target at row 0 and neighbors in rows 1:.",
            },
        }
        tmp_path = f"{self.training_pkl_path}.tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, self.training_pkl_path)
        self._last_flushed_sample_count = len(self._training_samples)
        if force or len(self._training_samples) > 0:
            self.get_logger().info(
                f"Saved {len(self._training_samples)} MOAI trajectory samples to {self.training_pkl_path}"
            )

    def _step_agent(
        self,
        agent: Agent,
        robot: Agent,
        all_agents: List[Agent],
        dt: float,
    ) -> Tuple[Agent, List[Tuple[float, float]], Optional[Tuple[float, float]]]:
        track = list(self._tracks[agent.id])
        fallback_vx = float(agent.velocity.linear.x)
        fallback_vy = float(agent.velocity.linear.y)

        goals = list(agent.goals)
        goals = self._advance_goals_if_needed(agent, goals)

        # Until the neural predictor is wired, keep the fallback human-like enough
        # to stay inside the scenario: follow the HuNav goals smoothly instead of
        # extrapolating the last velocity forever.
        if goals:
            gx = float(goals[0].position.x)
            gy = float(goals[0].position.y)
            dx = gx - float(agent.position.position.x)
            dy = gy - float(agent.position.position.y)
            norm = hypot(dx, dy)
            if norm > 1e-3:
                desired = max(0.0, min(float(agent.desired_velocity), self.max_speed))
                goal_vx = desired * dx / norm
                goal_vy = desired * dy / norm
                if hypot(fallback_vx, fallback_vy) > 0.03:
                    fallback_vx = 0.35 * fallback_vx + 0.65 * goal_vx
                    fallback_vy = 0.35 * fallback_vy + 0.65 * goal_vy
                else:
                    fallback_vx = goal_vx
                    fallback_vy = goal_vy

        final_goal_xy = None
        if goals:
            final_goal_xy = (float(goals[0].position.x), float(goals[0].position.y))
        guidance_xy = self._candidate_guidance_point(agent, final_goal_xy)
        predicted = self._predict_agent_trajectory(agent, robot, all_agents, track, fallback_vx, fallback_vy, guidance_xy)
        vx, vy = self._trajectory_velocity(agent, goals, predicted)
        vx, vy = self._blend_goal_velocity(agent, goals, vx, vy)
        vx, vy = self._ensure_goal_progress(agent, goals, vx, vy)
        vx, vy = self._soft_agent_avoidance(agent, all_agents, vx, vy)
        vx, vy = self._soft_obstacle_avoidance(agent, goals, vx, vy)
        vx, vy = self._soft_robot_avoidance(agent, robot, goals, vx, vy)
        vx, vy = self._limit_lateral_velocity(agent, goals, vx, vy)
        vx, vy = self._resolve_agent_collision_velocity(agent, all_agents, vx, vy, dt)
        vx, vy = self._resolve_obstacle_collision_velocity(agent, vx, vy, dt)
        vx, vy = self._resolve_map_collision_velocity(agent, vx, vy, dt)
        vx, vy = self._resolve_robot_collision_velocity(agent, robot, goals, vx, vy, dt)
        vx, vy = self._limit_lateral_velocity(agent, goals, vx, vy)
        vx, vy = self._apply_goal_arrival_control(agent, goals, vx, vy, dt)
        if not self._near_hard_hazard(agent, all_agents):
            vx, vy = self._ensure_goal_progress(agent, goals, vx, vy)
        current_yaw = self._agent_yaw(agent)
        vx, vy = self._limit_velocity(agent.id, vx, vy, agent.desired_velocity, dt)
        vx, vy = self._resolve_agent_collision_velocity(agent, all_agents, vx, vy, dt)
        vx, vy = self._resolve_obstacle_collision_velocity(agent, vx, vy, dt)
        vx, vy = self._resolve_map_collision_velocity(agent, vx, vy, dt)
        vx, vy = self._resolve_robot_collision_velocity(agent, robot, goals, vx, vy, dt)
        vx, vy = self._clamp_speed_only(vx, vy, agent.desired_velocity)
        self._last_cmd_vel[agent.id] = (vx, vy)
        if hypot(vx, vy) > 0.03:
            yaw = self._normalize_angle(atan2(vy, vx))
            self._last_yaw[agent.id] = yaw
        else:
            yaw = self._normalize_angle(self._last_yaw.get(agent.id, current_yaw))
            self._last_yaw[agent.id] = yaw

        updated = Agent()
        updated.id = agent.id
        updated.type = agent.type
        updated.skin = agent.skin
        updated.name = agent.name
        updated.group_id = agent.group_id
        updated.position = agent.position
        updated.position.position.x = float(agent.position.position.x) + vx * dt
        updated.position.position.y = float(agent.position.position.y) + vy * dt
        updated.position.orientation = self._yaw_to_quaternion(yaw)
        updated.yaw = yaw
        updated.velocity = agent.velocity
        updated.velocity.linear.x = vx
        updated.velocity.linear.y = vy
        updated.velocity.angular.z = self._angle_diff(yaw, current_yaw) / max(dt, 1e-3)
        updated.desired_velocity = agent.desired_velocity
        updated.radius = agent.radius
        updated.linear_vel = hypot(vx, vy)
        updated.angular_vel = updated.velocity.angular.z
        updated.behavior = agent.behavior
        updated.goals = goals
        updated.cyclic_goals = agent.cyclic_goals
        updated.goal_radius = agent.goal_radius
        updated.closest_obs = agent.closest_obs
        return updated, predicted, guidance_xy

    def _candidate_guidance_point(
        self,
        agent: Agent,
        final_goal_xy: Optional[Tuple[float, float]],
    ) -> Optional[Tuple[float, float]]:
        if final_goal_xy is None:
            return None

        mode = self.spubert_candidate_selection_mode
        if mode in {"none", "disabled", "off"}:
            return None
        if mode in {"final_goal", "goal"}:
            return final_goal_xy

        ax = float(agent.position.position.x)
        ay = float(agent.position.position.y)
        return self._guidance_point_from_xy((ax, ay), final_goal_xy)

    def _guidance_point_from_xy(
        self,
        current_xy: Tuple[float, float] | np.ndarray,
        final_goal_xy: Tuple[float, float] | np.ndarray,
    ) -> Tuple[float, float]:
        ax = float(current_xy[0])
        ay = float(current_xy[1])
        gx = float(final_goal_xy[0])
        gy = float(final_goal_xy[1])
        dx = gx - ax
        dy = gy - ay
        dist = hypot(dx, dy)
        if dist <= 1e-6:
            return gx, gy

        radius = max(0.0, self.guidance_point_radius)
        step = min(radius, dist)
        return ax + dx / dist * step, ay + dy / dist * step

    def _predict_agent_trajectory(
        self,
        agent: Agent,
        robot: Agent,
        all_agents: List[Agent],
        track: List[TrackPoint],
        fallback_vx: float,
        fallback_vy: float,
        goal_xy: Optional[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        agent_id = int(agent.id)
        if self._spubert_predictor is not None and len(track) >= 2:
            stamp = track[-1].stamp
            selected: List[Tuple[float, float]] = []
            refresh_snapshot: Optional[Dict[int, Deque[TrackPoint]]] = None
            with self._cache_lock:
                cached = self._prediction_cache.get(agent_id)
                if cached is not None and stamp - cached[0] <= self.spubert_cache_ttl:
                    needs_refresh = stamp - cached[0] >= self.spubert_refresh_dt
                    selected = self._select_safe_candidate(agent, robot, all_agents, cached[1], goal_xy)
                    if needs_refresh:
                        refresh_snapshot = self._schedule_spubert_prediction_locked(agent_id, stamp)
            if refresh_snapshot is not None:
                self._start_spubert_prediction_thread(agent_id, stamp, refresh_snapshot, goal_xy)
            if selected:
                return selected

            if self.spubert_sync_on_cache_miss:
                candidates = self._run_spubert_prediction_sync(agent_id, stamp)
                if candidates:
                    selected = self._select_safe_candidate(agent, robot, all_agents, candidates, goal_xy)
                    if selected:
                        return selected

            self._schedule_spubert_prediction(agent_id, stamp, goal_xy)

        return self._fallback_predictor.predict(track, fallback_vx, fallback_vy)

    def _is_debug_focus_agent(self, agent: Agent) -> bool:
        return self.debug_focus_agent_id <= 0 or int(agent.id) == self.debug_focus_agent_id

    def _get_cached_candidate_paths(self, agent_id: int, stamp: float) -> List[List[Tuple[float, float]]]:
        with self._cache_lock:
            cached = self._prediction_cache.get(agent_id)
            if cached is None or stamp - cached[0] > self.spubert_cache_ttl:
                return []
            return cached[1]

    def _schedule_spubert_prediction(self, agent_id: int, stamp: float, goal_xy: Optional[Tuple[float, float]]) -> None:
        with self._cache_lock:
            tracks_snapshot = self._schedule_spubert_prediction_locked(agent_id, stamp)
            if tracks_snapshot is None:
                return

        self._start_spubert_prediction_thread(agent_id, stamp, tracks_snapshot, goal_xy)

    def _run_spubert_prediction_sync(
        self,
        agent_id: int,
        stamp: float,
    ) -> List[List[Tuple[float, float]]]:
        with self._cache_lock:
            if agent_id in self._pending_predictions:
                return []
            cached = self._prediction_cache.get(agent_id)
            if cached is not None and stamp - cached[0] <= self.spubert_cache_ttl:
                return cached[1]
            tracks_snapshot = {key: deque(value, maxlen=self.obs_len) for key, value in self._tracks.items()}
            self._pending_predictions.add(agent_id)

        try:
            if self._spubert_predictor is None:
                return []
            with self._inference_lock:
                candidates = self._spubert_predictor.predict_candidates(agent_id=agent_id, tracks=tracks_snapshot)
            with self._cache_lock:
                self._prediction_cache[agent_id] = (stamp, candidates)
            if candidates:
                self.get_logger().debug(
                    f"SPU-BERT sync inference produced {len(candidates)} candidates for agent {agent_id}."
                )
            return candidates
        except Exception as exc:
            self.get_logger().warn(f"SPU-BERT sync inference failed for agent {agent_id}: {exc}; using fallback.")
            return []
        finally:
            with self._cache_lock:
                self._pending_predictions.discard(agent_id)

    def _start_spubert_prediction_thread(
        self,
        agent_id: int,
        stamp: float,
        tracks_snapshot: Dict[int, Deque[TrackPoint]],
        goal_xy: Optional[Tuple[float, float]],
    ) -> None:
        thread = threading.Thread(
            target=self._run_spubert_prediction,
            args=(agent_id, stamp, tracks_snapshot, goal_xy),
            daemon=True,
        )
        thread.start()

    def _schedule_spubert_prediction_locked(
        self, agent_id: int, stamp: float
    ) -> Optional[Dict[int, Deque[TrackPoint]]]:
        if agent_id in self._pending_predictions:
            return None
        tracks_snapshot = {key: deque(value, maxlen=self.obs_len) for key, value in self._tracks.items()}
        self._pending_predictions.add(agent_id)
        return tracks_snapshot

    def _run_spubert_prediction(
        self,
        agent_id: int,
        stamp: float,
        tracks_snapshot: Dict[int, Deque[TrackPoint]],
        goal_xy: Optional[Tuple[float, float]],
    ) -> None:
        try:
            if self._spubert_predictor is None:
                return
            with self._inference_lock:
                candidates = self._spubert_predictor.predict_candidates(agent_id=agent_id, tracks=tracks_snapshot)
            with self._cache_lock:
                self._prediction_cache[agent_id] = (stamp, candidates)
            if candidates:
                self.get_logger().debug(
                    f"SPU-BERT inference produced {len(candidates)} candidates for agent {agent_id}."
                )
        except Exception as exc:
            self.get_logger().warn(f"SPU-BERT inference failed for agent {agent_id}: {exc}; using fallback.")
        finally:
            with self._cache_lock:
                self._pending_predictions.discard(agent_id)

    def _select_safe_candidate(
        self,
        agent: Agent,
        robot: Agent,
        all_agents: List[Agent],
        candidates: List[List[Tuple[float, float]]],
        goal_xy: Optional[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        if not candidates:
            return []

        ax = float(agent.position.position.x)
        ay = float(agent.position.position.y)
        radius = max(float(agent.radius), 0.0)
        obstacles = self._unique_obstacle_points(agent)
        other_agents = [other for other in all_agents if other.id != agent.id]

        goal_dir_x = 0.0
        goal_dir_y = 0.0
        if goal_xy is not None:
            gdx = goal_xy[0] - ax
            gdy = goal_xy[1] - ay
            gdist = hypot(gdx, gdy)
            if gdist > 1e-3:
                goal_dir_x = gdx / gdist
                goal_dir_y = gdy / gdist

        best_path = candidates[0]
        best_score = float("inf")
        for path in candidates:
            if not path:
                continue
            score = 0.0
            if self._spubert_predictor is not None:
                score += self._spubert_predictor.map_collision_cost(path)

            if goal_xy is not None:
                final_x, final_y = path[-1]
                score += 0.35 * hypot(final_x - goal_xy[0], final_y - goal_xy[1])
                progress = (final_x - ax) * goal_dir_x + (final_y - ay) * goal_dir_y
                if progress < 0.0:
                    score += 10.0 + 4.0 * abs(progress)

            prev_x = ax
            prev_y = ay
            prev_step_x = 0.0
            prev_step_y = 0.0
            for idx, (px, py) in enumerate(path):
                weight = 1.0 + 0.05 * idx
                step = hypot(px - prev_x, py - prev_y)
                if step > self.max_speed * self.prediction_dt * 1.8:
                    score += 0.5 * step

                if goal_xy is not None:
                    point_progress = (px - ax) * goal_dir_x + (py - ay) * goal_dir_y
                    if idx < 4 and point_progress < 0.03 * (idx + 1):
                        score += 2.0 * (0.03 * (idx + 1) - point_progress)

                if step > 1e-3 and hypot(prev_step_x, prev_step_y) > 1e-3:
                    dot = (px - prev_x) * prev_step_x + (py - prev_y) * prev_step_y
                    denom = step * hypot(prev_step_x, prev_step_y)
                    turn = max(-1.0, min(1.0, dot / max(denom, 1e-6)))
                    if turn < 0.3:
                        score += 0.25 * (0.3 - turn)
                prev_step_x = px - prev_x
                prev_step_y = py - prev_y
                prev_x, prev_y = px, py

                for ox, oy in obstacles:
                    clearance = hypot(px - ox, py - oy) - radius
                    score += 1.4 * weight * self._clearance_penalty(clearance, soft=1.6, hard=0.55)

                rdist = hypot(px - float(robot.position.position.x), py - float(robot.position.position.y))
                rclearance = rdist - radius - max(float(robot.radius), 0.0)
                score += 1.2 * weight * self._clearance_penalty(rclearance, soft=1.4, hard=0.45)

                for other in other_agents:
                    odist = hypot(px - float(other.position.position.x), py - float(other.position.position.y))
                    oclearance = odist - radius - max(float(other.radius), 0.0)
                    score += 2.2 * weight * self._clearance_penalty(oclearance, soft=1.35, hard=0.45)

            if score < best_score:
                best_score = score
                best_path = path

        return best_path

    def _trajectory_velocity(self, agent: Agent, goals: List, predicted: List[Tuple[float, float]]) -> Tuple[float, float]:
        if not predicted:
            return 0.0, 0.0

        ax = float(agent.position.position.x)
        ay = float(agent.position.position.y)
        desired_speed = max(0.0, min(float(agent.desired_velocity), self.max_speed))
        count = max(1, min(self.lookahead_step, len(predicted)))
        sum_w = 0.0
        vx = 0.0
        vy = 0.0
        for idx, (px, py) in enumerate(predicted[:count]):
            horizon = max(self.prediction_dt * (idx + 1), 1e-3)
            weight = float(idx + 1)
            vx += weight * (float(px) - ax) / horizon
            vy += weight * (float(py) - ay) / horizon
            sum_w += weight

        vx /= max(sum_w, 1e-6)
        vy /= max(sum_w, 1e-6)

        if goals:
            gx = float(goals[0].position.x)
            gy = float(goals[0].position.y)
            gdx = gx - ax
            gdy = gy - ay
            gdist = hypot(gdx, gdy)
            speed = hypot(vx, vy)
            if gdist > 1e-3 and speed > 1e-3:
                goal_dir_x = gdx / gdist
                goal_dir_y = gdy / gdist
                side_x = -goal_dir_y
                side_y = goal_dir_x
                forward = vx * goal_dir_x + vy * goal_dir_y
                lateral = vx * side_x + vy * side_y
                forward = max(0.0, forward)
                forward = 0.25 * min(forward, desired_speed) + 0.75 * desired_speed
                max_lateral = max(0.15, desired_speed * 0.35)
                lateral = max(-max_lateral, min(max_lateral, lateral))
                return forward * goal_dir_x + lateral * side_x, forward * goal_dir_y + lateral * side_y

        return vx, vy

    def _unique_obstacle_points(self, agent: Agent) -> List[Tuple[float, float]]:
        points: List[Tuple[float, float]] = []
        seen = set()
        for obs in agent.closest_obs:
            key = (round(float(obs.x), 2), round(float(obs.y), 2))
            if key in seen:
                continue
            seen.add(key)
            points.append((float(obs.x), float(obs.y)))
        return points

    @staticmethod
    def _clearance_penalty(clearance: float, *, soft: float, hard: float) -> float:
        if clearance >= soft:
            return 0.0
        if clearance <= hard:
            return 100.0 + 50.0 * (hard - clearance)
        return ((soft - clearance) / max(soft - hard, 1e-6)) ** 2

    def _ensure_goal_progress(self, agent: Agent, goals: List, vx: float, vy: float) -> Tuple[float, float]:
        if not goals:
            return vx, vy

        ax = float(agent.position.position.x)
        ay = float(agent.position.position.y)
        gx = float(goals[0].position.x)
        gy = float(goals[0].position.y)
        dx = gx - ax
        dy = gy - ay
        dist = hypot(dx, dy)
        arrival_distance = max(max(float(agent.goal_radius), 0.1) + 0.1, self.goal_arrival_distance)
        if dist <= arrival_distance:
            return vx, vy

        desired_speed = max(0.0, min(float(agent.desired_velocity), self.max_speed))
        min_speed = min(self.min_goal_speed, desired_speed)
        speed = hypot(vx, vy)
        if speed >= min_speed:
            return vx, vy

        goal_vx = min_speed * dx / dist
        goal_vy = min_speed * dy / dist
        if speed <= 1e-3:
            return goal_vx, goal_vy

        # Keep the neural prediction's lateral tendency, but prevent near-zero
        # first-step predictions from freezing the simulated pedestrian.
        blend = 0.65
        out_vx = (1.0 - blend) * vx + blend * goal_vx
        out_vy = (1.0 - blend) * vy + blend * goal_vy
        out_speed = hypot(out_vx, out_vy)
        if out_speed < min_speed:
            out_vx *= min_speed / max(out_speed, 1e-6)
            out_vy *= min_speed / max(out_speed, 1e-6)
        return out_vx, out_vy

    def _blend_goal_velocity(self, agent: Agent, goals: List, vx: float, vy: float) -> Tuple[float, float]:
        if not goals:
            return vx, vy

        ax = float(agent.position.position.x)
        ay = float(agent.position.position.y)
        gx = float(goals[0].position.x)
        gy = float(goals[0].position.y)
        dx = gx - ax
        dy = gy - ay
        dist = hypot(dx, dy)
        if dist <= max(float(agent.goal_radius), 0.1) + 0.1:
            return vx, vy

        desired_speed = max(0.0, min(float(agent.desired_velocity), self.max_speed))
        if desired_speed <= 1e-3:
            return vx, vy

        goal_vx = desired_speed * dx / dist
        goal_vy = desired_speed * dy / dist
        blend = max(0.0, min(self.goal_velocity_blend, 1.0))
        return (1.0 - blend) * vx + blend * goal_vx, (1.0 - blend) * vy + blend * goal_vy

    def _delete_all_marker(self, frame_id: str) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.action = Marker.DELETEALL
        return marker

    def _path_markers(
        self,
        marker_id: int,
        agent: Agent,
        predicted: List[Tuple[float, float]],
        candidate_paths: List[List[Tuple[float, float]]],
        guidance_xy: Optional[Tuple[float, float]],
        frame_id: str,
    ) -> List[Marker]:
        z = max(float(agent.position.position.z), 0.05) + 0.2
        stamp = self.get_clock().now().to_msg()
        points = [
            Point(x=float(agent.position.position.x), y=float(agent.position.position.y), z=z),
            *[Point(x=x, y=y, z=z) for x, y in predicted],
        ]

        line = Marker()
        line.header.frame_id = frame_id
        line.header.stamp = stamp
        line.ns = f"agent_{agent.id}_predicted_path"
        line.id = marker_id
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.06
        line.color = ColorRGBA(r=1.0, g=0.35, b=0.05, a=0.9)
        line.lifetime = Duration(sec=1)
        line.points = points

        dots = Marker()
        dots.header.frame_id = frame_id
        dots.header.stamp = line.header.stamp
        dots.ns = f"agent_{agent.id}_predicted_points"
        dots.id = marker_id + 1
        dots.type = Marker.SPHERE_LIST
        dots.action = Marker.ADD
        dots.pose.orientation.w = 1.0
        dots.scale.x = 0.16
        dots.scale.y = 0.16
        dots.scale.z = 0.16
        dots.color = ColorRGBA(r=1.0, g=0.2, b=0.05, a=0.85)
        dots.lifetime = Duration(sec=1)
        dots.points = points[1:]

        label = Marker()
        label.header.frame_id = frame_id
        label.header.stamp = line.header.stamp
        label.ns = f"agent_{agent.id}_predicted_label"
        label.id = marker_id + 2
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = float(agent.position.position.x)
        label.pose.position.y = float(agent.position.position.y)
        label.pose.position.z = z + 0.35
        label.pose.orientation.w = 1.0
        label.scale.z = 0.28
        label.color = ColorRGBA(r=0.92, g=0.96, b=1.0, a=0.95)
        label.lifetime = Duration(sec=1)
        label.text = f"agent {agent.id}"

        body = Marker()
        body.header.frame_id = frame_id
        body.header.stamp = line.header.stamp
        body.ns = f"agent_{agent.id}_body"
        body.id = marker_id + 3
        body.type = Marker.CYLINDER
        body.action = Marker.ADD
        body.pose.position.x = float(agent.position.position.x)
        body.pose.position.y = float(agent.position.position.y)
        body.pose.position.z = z
        body.pose.orientation.w = 1.0
        diameter = max(0.25, float(agent.radius) * 2.0)
        body.scale.x = diameter
        body.scale.y = diameter
        body.scale.z = 0.45
        body.color = ColorRGBA(r=0.1, g=0.55, b=1.0, a=0.9)
        body.lifetime = Duration(sec=1)

        heading = Marker()
        heading.header.frame_id = frame_id
        heading.header.stamp = line.header.stamp
        heading.ns = f"agent_{agent.id}_heading"
        heading.id = marker_id + 4
        heading.type = Marker.ARROW
        heading.action = Marker.ADD
        heading.pose.orientation.w = 1.0
        heading.scale.x = 0.05
        heading.scale.y = 0.12
        heading.scale.z = 0.12
        heading.color = ColorRGBA(r=0.1, g=0.85, b=1.0, a=0.9)
        heading.lifetime = Duration(sec=1)
        yaw = self._agent_yaw(agent)
        heading.points = [
            Point(x=float(agent.position.position.x), y=float(agent.position.position.y), z=z + 0.28),
            Point(
                x=float(agent.position.position.x) + cos(yaw) * 0.75,
                y=float(agent.position.position.y) + sin(yaw) * 0.75,
                z=z + 0.28,
            ),
        ]

        candidate_lines = Marker()
        candidate_lines.header.frame_id = frame_id
        candidate_lines.header.stamp = line.header.stamp
        candidate_lines.ns = f"agent_{agent.id}_candidate_paths"
        candidate_lines.id = marker_id + 7
        candidate_lines.type = Marker.LINE_LIST
        candidate_lines.action = Marker.ADD
        candidate_lines.pose.orientation.w = 1.0
        candidate_lines.scale.x = 0.045
        candidate_lines.color = ColorRGBA(r=0.0, g=0.35, b=1.0, a=0.75)
        candidate_lines.lifetime = Duration(sec=1)

        candidate_goals = Marker()
        candidate_goals.header.frame_id = frame_id
        candidate_goals.header.stamp = line.header.stamp
        candidate_goals.ns = f"agent_{agent.id}_candidate_goals"
        candidate_goals.id = marker_id + 8
        candidate_goals.type = Marker.SPHERE_LIST
        candidate_goals.action = Marker.ADD
        candidate_goals.pose.orientation.w = 1.0
        candidate_goals.scale.x = 0.2
        candidate_goals.scale.y = 0.2
        candidate_goals.scale.z = 0.2
        candidate_goals.color = ColorRGBA(r=0.0, g=0.35, b=1.0, a=0.9)
        candidate_goals.lifetime = Duration(sec=1)

        start = Point(x=float(agent.position.position.x), y=float(agent.position.position.y), z=z + 0.18)
        for path in candidate_paths:
            if not path:
                continue
            prev = start
            for px, py in path:
                cur = Point(x=float(px), y=float(py), z=z + 0.18)
                candidate_lines.points.extend([prev, cur])
                prev = cur
            candidate_goals.points.append(Point(x=float(path[-1][0]), y=float(path[-1][1]), z=z + 0.28))

        interaction_range = Marker()
        interaction_range.header.frame_id = frame_id
        interaction_range.header.stamp = line.header.stamp
        interaction_range.ns = f"agent_{agent.id}_interaction_range"
        interaction_range.id = marker_id + 9
        interaction_range.type = Marker.LINE_STRIP
        interaction_range.action = Marker.ADD
        interaction_range.pose.orientation.w = 1.0
        interaction_range.scale.x = 0.025
        interaction_range.color = ColorRGBA(r=0.05, g=0.05, b=0.05, a=0.32)
        interaction_range.lifetime = Duration(sec=1)
        radius = max(0.1, float(self.spubert_social_range))
        cx = float(agent.position.position.x)
        cy = float(agent.position.position.y)
        for idx in range(73):
            theta = 2.0 * pi * idx / 72.0
            interaction_range.points.append(Point(x=cx + cos(theta) * radius, y=cy + sin(theta) * radius, z=z - 0.08))

        obstacle_points = Marker()
        obstacle_points.header.frame_id = frame_id
        obstacle_points.header.stamp = line.header.stamp
        obstacle_points.ns = f"agent_{agent.id}_closest_obstacles"
        obstacle_points.id = marker_id + 10
        obstacle_points.type = Marker.CUBE_LIST
        obstacle_points.action = Marker.ADD
        obstacle_points.pose.orientation.w = 1.0
        obstacle_points.scale.x = 0.28
        obstacle_points.scale.y = 0.28
        obstacle_points.scale.z = 0.28
        obstacle_points.color = ColorRGBA(r=0.95, g=0.1, b=0.1, a=0.9)
        obstacle_points.lifetime = Duration(sec=1)
        obstacle_points.points = [
            Point(x=ox, y=oy, z=z - 0.12) for ox, oy in self._unique_obstacle_points(agent)
        ]

        input_markers = []
        if self.debug_show_all_model_io or self._is_debug_focus_agent(agent):
            input_markers = self._input_output_explain_markers(
                marker_id=marker_id + 20,
                agent=agent,
                predicted=predicted,
                frame_id=frame_id,
                stamp=stamp,
                z=z,
            )

        if self.debug_guidance_only:
            markers = [body, heading, line, dots]
        else:
            markers = [
                interaction_range,
                body,
                heading,
                line,
                dots,
                candidate_lines,
                candidate_goals,
                obstacle_points,
                label,
                *input_markers,
            ]
        if guidance_xy is not None:
            guidance_marker = Marker()
            guidance_marker.header.frame_id = frame_id
            guidance_marker.header.stamp = line.header.stamp
            guidance_marker.ns = f"agent_{agent.id}_guidance_point"
            guidance_marker.id = marker_id + 11
            guidance_marker.type = Marker.CUBE
            guidance_marker.action = Marker.ADD
            guidance_marker.pose.position.x = float(guidance_xy[0])
            guidance_marker.pose.position.y = float(guidance_xy[1])
            guidance_marker.pose.position.z = z + 0.18
            guidance_marker.pose.orientation.w = 1.0
            guidance_marker.scale.x = 0.28
            guidance_marker.scale.y = 0.28
            guidance_marker.scale.z = 0.28
            guidance_marker.color = ColorRGBA(r=1.0, g=0.95, b=0.05, a=0.95)
            guidance_marker.lifetime = Duration(sec=1)

            guidance_circle = Marker()
            guidance_circle.header.frame_id = frame_id
            guidance_circle.header.stamp = line.header.stamp
            guidance_circle.ns = f"agent_{agent.id}_guidance_radius"
            guidance_circle.id = marker_id + 12
            guidance_circle.type = Marker.LINE_STRIP
            guidance_circle.action = Marker.ADD
            guidance_circle.pose.orientation.w = 1.0
            guidance_circle.scale.x = 0.06
            guidance_circle.color = ColorRGBA(r=0.02, g=0.02, b=0.02, a=0.78)
            guidance_circle.lifetime = Duration(sec=1)
            gr = max(0.0, self.guidance_point_radius)
            cx = float(agent.position.position.x)
            cy = float(agent.position.position.y)
            if gr > 1e-3:
                for idx in range(73):
                    theta = 2.0 * pi * idx / 72.0
                    guidance_circle.points.append(
                        Point(x=cx + cos(theta) * gr, y=cy + sin(theta) * gr, z=z + 0.04)
                    )
            markers.extend([guidance_marker, guidance_circle])

        if agent.goals:
            goal = agent.goals[0]

            goal_marker = Marker()
            goal_marker.header.frame_id = frame_id
            goal_marker.header.stamp = line.header.stamp
            goal_marker.ns = f"agent_{agent.id}_goal"
            goal_marker.id = marker_id + 5
            goal_marker.type = Marker.SPHERE
            goal_marker.action = Marker.ADD
            goal_marker.pose.position.x = float(goal.position.x)
            goal_marker.pose.position.y = float(goal.position.y)
            goal_marker.pose.position.z = z
            goal_marker.pose.orientation.w = 1.0
            goal_marker.scale.x = 0.35
            goal_marker.scale.y = 0.35
            goal_marker.scale.z = 0.35
            goal_marker.color = ColorRGBA(r=0.15, g=1.0, b=0.35, a=0.9)
            goal_marker.lifetime = Duration(sec=1)

            goal_line = Marker()
            goal_line.header.frame_id = frame_id
            goal_line.header.stamp = line.header.stamp
            goal_line.ns = f"agent_{agent.id}_goal_line"
            goal_line.id = marker_id + 6
            goal_line.type = Marker.LINE_STRIP
            goal_line.action = Marker.ADD
            goal_line.pose.orientation.w = 1.0
            goal_line.scale.x = 0.035
            goal_line.color = ColorRGBA(r=0.15, g=1.0, b=0.35, a=0.55)
            goal_line.lifetime = Duration(sec=1)
            goal_line.points = [
                Point(x=float(agent.position.position.x), y=float(agent.position.position.y), z=z + 0.1),
                Point(x=float(goal.position.x), y=float(goal.position.y), z=z + 0.1),
            ]
            markers.extend([goal_marker, goal_line])

        return markers

    def _input_output_explain_markers(
        self,
        *,
        marker_id: int,
        agent: Agent,
        predicted: List[Tuple[float, float]],
        frame_id: str,
        stamp: Any,
        z: float,
    ) -> List[Marker]:
        track = list(self._tracks.get(agent.id, []))
        obs_points = [Point(x=float(p.x), y=float(p.y), z=z + 0.38) for p in track[-self.obs_len :]]
        pred_points = [Point(x=float(x), y=float(y), z=z + 0.5) for x, y in predicted[: self.pred_len]]

        obs_line = Marker()
        obs_line.header.frame_id = frame_id
        obs_line.header.stamp = stamp
        obs_line.ns = f"agent_{agent.id}_model_input_obs_line"
        obs_line.id = marker_id
        obs_line.type = Marker.LINE_STRIP
        obs_line.action = Marker.ADD
        obs_line.pose.orientation.w = 1.0
        obs_line.scale.x = 0.075
        obs_line.color = ColorRGBA(r=0.05, g=0.45, b=1.0, a=0.95)
        obs_line.lifetime = Duration(sec=1)
        obs_line.points = obs_points

        obs_dots = Marker()
        obs_dots.header.frame_id = frame_id
        obs_dots.header.stamp = stamp
        obs_dots.ns = f"agent_{agent.id}_model_input_obs_points"
        obs_dots.id = marker_id + 1
        obs_dots.type = Marker.SPHERE_LIST
        obs_dots.action = Marker.ADD
        obs_dots.pose.orientation.w = 1.0
        obs_dots.scale.x = 0.22
        obs_dots.scale.y = 0.22
        obs_dots.scale.z = 0.22
        obs_dots.color = ColorRGBA(r=0.05, g=0.45, b=1.0, a=0.95)
        obs_dots.lifetime = Duration(sec=1)
        obs_dots.points = obs_points

        pred_labels_anchor = pred_points
        pred_dots = Marker()
        pred_dots.header.frame_id = frame_id
        pred_dots.header.stamp = stamp
        pred_dots.ns = f"agent_{agent.id}_model_output_pred_points"
        pred_dots.id = marker_id + 2
        pred_dots.type = Marker.SPHERE_LIST
        pred_dots.action = Marker.ADD
        pred_dots.pose.orientation.w = 1.0
        pred_dots.scale.x = 0.19
        pred_dots.scale.y = 0.19
        pred_dots.scale.z = 0.19
        pred_dots.color = ColorRGBA(r=1.0, g=0.45, b=0.0, a=0.95)
        pred_dots.lifetime = Duration(sec=1)
        pred_dots.points = pred_points

        lookahead = Marker()
        lookahead.header.frame_id = frame_id
        lookahead.header.stamp = stamp
        lookahead.ns = f"agent_{agent.id}_controller_lookahead_point"
        lookahead.id = marker_id + 3
        lookahead.type = Marker.SPHERE
        lookahead.action = Marker.ADD
        lookahead.pose.orientation.w = 1.0
        lookahead.scale.x = 0.38
        lookahead.scale.y = 0.38
        lookahead.scale.z = 0.38
        lookahead.color = ColorRGBA(r=1.0, g=0.95, b=0.0, a=0.98)
        lookahead.lifetime = Duration(sec=1)
        if pred_points:
            lookahead_idx = min(max(self.lookahead_step, 1), len(pred_points)) - 1
            lookahead.pose.position = pred_points[lookahead_idx]

        title = Marker()
        title.header.frame_id = frame_id
        title.header.stamp = stamp
        title.ns = f"agent_{agent.id}_model_io_title"
        title.id = marker_id + 4
        title.type = Marker.TEXT_VIEW_FACING
        title.action = Marker.ADD
        title.pose.position.x = float(agent.position.position.x)
        title.pose.position.y = float(agent.position.position.y)
        title.pose.position.z = z + 0.95
        title.pose.orientation.w = 1.0
        title.scale.z = 0.24
        title.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.96)
        title.lifetime = Duration(sec=1)
        title.text = f"agent {agent.id}: blue=input 8, orange=pred 12"

        labels = []
        for idx, point in enumerate(obs_points, start=1):
            labels.append(
                self._text_marker(
                    marker_id + 10 + idx,
                    f"agent_{agent.id}_model_input_obs_labels",
                    frame_id,
                    stamp,
                    point,
                    f"obs{idx}",
                    ColorRGBA(r=0.65, g=0.86, b=1.0, a=0.96),
                    0.18,
                    z_offset=0.13,
                )
            )

        for idx, point in enumerate(pred_labels_anchor, start=1):
            labels.append(
                self._text_marker(
                    marker_id + 30 + idx,
                    f"agent_{agent.id}_model_output_pred_labels",
                    frame_id,
                    stamp,
                    point,
                    f"p{idx}",
                    ColorRGBA(r=1.0, g=0.82, b=0.5, a=0.96),
                    0.18,
                    z_offset=0.15,
                )
            )

        if pred_points:
            lookahead_idx = min(max(self.lookahead_step, 1), len(pred_points)) - 1
            labels.append(
                self._text_marker(
                    marker_id + 60,
                    f"agent_{agent.id}_controller_lookahead_label",
                    frame_id,
                    stamp,
                    pred_points[lookahead_idx],
                    f"lookahead p{lookahead_idx + 1}",
                    ColorRGBA(r=1.0, g=1.0, b=0.35, a=0.98),
                    0.22,
                    z_offset=0.3,
                )
            )

        return [obs_line, obs_dots, pred_dots, lookahead, title, *labels]

    def _text_marker(
        self,
        marker_id: int,
        namespace: str,
        frame_id: str,
        stamp: Any,
        point: Point,
        text: str,
        color: ColorRGBA,
        scale: float,
        *,
        z_offset: float = 0.0,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = point.x
        marker.pose.position.y = point.y
        marker.pose.position.z = point.z + z_offset
        marker.pose.orientation.w = 1.0
        marker.scale.z = scale
        marker.color = color
        marker.lifetime = Duration(sec=1)
        marker.text = text
        return marker

    def _advance_goals_if_needed(self, agent: Agent, goals: List) -> List:
        if not goals:
            return goals

        ax = float(agent.position.position.x)
        ay = float(agent.position.position.y)
        gx = float(goals[0].position.x)
        gy = float(goals[0].position.y)
        radius = max(float(agent.goal_radius), 0.1) + 0.1
        if hypot(gx - ax, gy - ay) > radius:
            return goals

        reached = goals.pop(0)
        if agent.cyclic_goals:
            goals.append(reached)
        return goals

    def _soft_agent_avoidance(self, agent: Agent, all_agents: List[Agent], vx: float, vy: float) -> Tuple[float, float]:
        ax = float(agent.position.position.x)
        ay = float(agent.position.position.y)
        radius = max(float(agent.radius), 0.0)

        avoid_x = 0.0
        avoid_y = 0.0
        speed = hypot(vx, vy)
        for other in all_agents:
            if other.id == agent.id:
                continue
            ox = float(other.position.position.x)
            oy = float(other.position.position.y)
            dx = ax - ox
            dy = ay - oy
            dist = hypot(dx, dy)
            clearance = dist - radius - max(float(other.radius), 0.0)
            if dist <= 1e-3 or clearance >= self.agent_personal_space:
                continue

            strength = (self.agent_personal_space - max(clearance, 0.0)) / self.agent_personal_space
            strength *= strength
            away_x = dx / dist
            away_y = dy / dist
            side_x = -vy / speed if speed > 0.05 else -away_y
            side_y = vx / speed if speed > 0.05 else away_x
            side_sign = self._choose_avoid_side(
                int(agent.id),
                side_x,
                side_y,
                away_x,
                away_y,
                vx,
                vy,
                goal_dir_x=0.0,
                goal_dir_y=0.0,
            )
            avoid_x += self.agent_avoidance_gain * strength * (0.45 * away_x + 0.65 * side_sign * side_x)
            avoid_y += self.agent_avoidance_gain * strength * (0.45 * away_y + 0.65 * side_sign * side_y)

        return vx + avoid_x, vy + avoid_y

    def _soft_obstacle_avoidance(self, agent: Agent, goals: List, vx: float, vy: float) -> Tuple[float, float]:
        ax = float(agent.position.position.x)
        ay = float(agent.position.position.y)
        goal_dir_x = 0.0
        goal_dir_y = 0.0
        if goals:
            gx = float(goals[0].position.x)
            gy = float(goals[0].position.y)
            gdx = gx - ax
            gdy = gy - ay
            gdist = hypot(gdx, gdy)
            if gdist > 1e-3:
                goal_dir_x = gdx / gdist
                goal_dir_y = gdy / gdist

        avoid_x = 0.0
        avoid_y = 0.0
        used = 0
        for obs in agent.closest_obs:
            ox = float(obs.x)
            oy = float(obs.y)
            dx = ax - ox
            dy = ay - oy
            dist = hypot(dx, dy)
            if dist <= 1e-3 or dist >= self.obstacle_avoidance_distance:
                continue

            strength = (self.obstacle_avoidance_distance - dist) / self.obstacle_avoidance_distance
            strength *= strength
            speed = hypot(vx, vy)
            away_x = dx / dist
            away_y = dy / dist
            if hypot(goal_dir_x, goal_dir_y) > 0.5:
                side_x = -goal_dir_y
                side_y = goal_dir_x
            elif speed > 0.05:
                side_x = -vy / speed
                side_y = vx / speed
            else:
                side_x = -dy / dist
                side_y = dx / dist

            lateral_gain = self.obstacle_avoidance_gain * (0.85 + 1.65 * strength)
            away_gain = self.obstacle_avoidance_gain * 0.35 * strength
            side_sign = self._choose_avoid_side(
                int(agent.id),
                side_x,
                side_y,
                away_x,
                away_y,
                vx,
                vy,
                goal_dir_x=goal_dir_x,
                goal_dir_y=goal_dir_y,
                lateral_gain=lateral_gain,
                away_gain=away_gain,
            )
            avoid_x += lateral_gain * side_sign * side_x + away_gain * away_x
            avoid_y += lateral_gain * side_sign * side_y + away_gain * away_y

            obs_dir_x = -dx / dist
            obs_dir_y = -dy / dist
            toward_obstacle = vx * obs_dir_x + vy * obs_dir_y
            if toward_obstacle > 0.0 and dist < self.obstacle_avoidance_distance * 0.75:
                brake = min(0.45, 0.12 + 0.55 * strength) * toward_obstacle
                avoid_x -= brake * obs_dir_x
                avoid_y -= brake * obs_dir_y
            if dist < self.obstacle_avoidance_distance * 0.35:
                hard_push = 1.6 * self.obstacle_avoidance_gain * (self.obstacle_avoidance_distance * 0.35 - dist)
                avoid_x += hard_push * away_x
                avoid_y += hard_push * away_y
            used += 1

        if used > 0:
            # Nearby obstacle samples often arrive as a cluster. Averaging by
            # count made the avoidance too weak, so keep clustered walls firm.
            divisor = max(1.0, used ** 0.5)
            avoid_x /= divisor
            avoid_y /= divisor
        return vx + avoid_x, vy + avoid_y

    def _soft_robot_avoidance(
        self,
        agent: Agent,
        robot: Agent,
        goals: List,
        vx: float,
        vy: float,
    ) -> Tuple[float, float]:
        dx = float(agent.position.position.x) - float(robot.position.position.x)
        dy = float(agent.position.position.y) - float(robot.position.position.y)
        dist = hypot(dx, dy)
        if dist <= 1e-3 or dist >= self.robot_personal_space:
            return vx, vy

        goal_dir_x = 0.0
        goal_dir_y = 0.0
        if goals:
            gx = float(goals[0].position.x)
            gy = float(goals[0].position.y)
            gdx = gx - float(agent.position.position.x)
            gdy = gy - float(agent.position.position.y)
            gdist = hypot(gdx, gdy)
            if gdist > 1e-3:
                goal_dir_x = gdx / gdist
                goal_dir_y = gdy / gdist

        strength = (self.robot_personal_space - dist) / self.robot_personal_space
        away_x = dx / dist
        away_y = dy / dist

        if hypot(goal_dir_x, goal_dir_y) > 0.5:
            side_x = -goal_dir_y
            side_y = goal_dir_x
        else:
            speed = hypot(vx, vy)
            side_x = -vy / speed if speed > 0.05 else -away_y
            side_y = vx / speed if speed > 0.05 else away_x

        lateral_gain = 0.95 * strength
        away_gain = 0.08 * strength
        side_sign = self._choose_avoid_side(
            int(agent.id),
            side_x,
            side_y,
            away_x,
            away_y,
            vx,
            vy,
            goal_dir_x=goal_dir_x,
            goal_dir_y=goal_dir_y,
            lateral_gain=lateral_gain,
            away_gain=away_gain,
        )
        # Do not let robot avoidance become a U-turn. If the robot is in front
        # of the pedestrian, the away vector points backward, so use only the
        # lateral component and keep the final-goal direction intact.
        if hypot(goal_dir_x, goal_dir_y) > 0.5:
            away_goal_dot = away_x * goal_dir_x + away_y * goal_dir_y
            if away_goal_dot < 0.0:
                away_gain = 0.0

        return (
            vx + lateral_gain * side_sign * side_x + away_gain * away_x,
            vy + lateral_gain * side_sign * side_y + away_gain * away_y,
        )

    def _resolve_robot_collision_velocity(
        self,
        agent: Agent,
        robot: Agent,
        goals: List,
        vx: float,
        vy: float,
        dt: float,
    ) -> Tuple[float, float]:
        if not self.robot_hard_collision_guard:
            return vx, vy

        ax = float(agent.position.position.x)
        ay = float(agent.position.position.y)
        rx = float(robot.position.position.x)
        ry = float(robot.position.position.y)
        rvx = float(robot.velocity.linear.x)
        rvy = float(robot.velocity.linear.y)
        next_x = ax + vx * dt
        next_y = ay + vy * dt
        robot_next_x = rx + rvx * dt
        robot_next_y = ry + rvy * dt

        dx = next_x - robot_next_x
        dy = next_y - robot_next_y
        dist = hypot(dx, dy)
        if dist <= 1e-4:
            dx = ax - rx
            dy = ay - ry
            dist = hypot(dx, dy)
            if dist <= 1e-4:
                dx, dy, dist = 1.0, 0.0, 1.0

        agent_radius = max(float(agent.radius), 0.25)
        robot_radius = max(float(robot.radius), 0.35)
        min_dist = agent_radius + robot_radius + max(self.robot_collision_buffer, 0.0)
        if dist >= min_dist:
            return vx, vy

        away_x = dx / dist
        away_y = dy / dist
        push = min_dist - dist
        correction_x = push * away_x / max(dt, 1e-3)
        correction_y = push * away_y / max(dt, 1e-3)

        rel_vx = vx - rvx
        rel_vy = vy - rvy
        closing_speed = -(rel_vx * away_x + rel_vy * away_y)
        if closing_speed > 0.0:
            correction_x += closing_speed * away_x
            correction_y += closing_speed * away_y

        out_vx = vx + correction_x
        out_vy = vy + correction_y
        return self._preserve_goal_progress(agent, goals, out_vx, out_vy, min_ratio=0.25)

    def _preserve_goal_progress(
        self,
        agent: Agent,
        goals: List,
        vx: float,
        vy: float,
        *,
        min_ratio: float,
    ) -> Tuple[float, float]:
        if not goals:
            return vx, vy

        ax = float(agent.position.position.x)
        ay = float(agent.position.position.y)
        gx = float(goals[0].position.x)
        gy = float(goals[0].position.y)
        dx = gx - ax
        dy = gy - ay
        dist = hypot(dx, dy)
        if dist <= max(float(agent.goal_radius), 0.1) + 0.1:
            return vx, vy

        goal_x = dx / max(dist, 1e-6)
        goal_y = dy / max(dist, 1e-6)
        side_x = -goal_y
        side_y = goal_x
        forward = vx * goal_x + vy * goal_y
        lateral = vx * side_x + vy * side_y
        desired = max(0.0, min(float(agent.desired_velocity), self.max_speed))
        min_forward = min(desired * max(0.0, min(min_ratio, 1.0)), self.min_goal_speed)
        if forward >= min_forward:
            return vx, vy

        max_lateral = max(0.35, desired * max(0.7, self.max_lateral_speed_ratio))
        lateral = max(-max_lateral, min(max_lateral, lateral))
        return min_forward * goal_x + lateral * side_x, min_forward * goal_y + lateral * side_y

    def _choose_avoid_side(
        self,
        agent_id: int,
        side_x: float,
        side_y: float,
        away_x: float,
        away_y: float,
        vx: float,
        vy: float,
        *,
        goal_dir_x: float,
        goal_dir_y: float,
        lateral_gain: float = 0.65,
        away_gain: float = 0.25,
    ) -> float:
        previous = self._last_avoid_side.get(agent_id)

        def score(sign: float) -> float:
            cand_x = vx + lateral_gain * sign * side_x + away_gain * away_x
            cand_y = vy + lateral_gain * sign * side_y + away_gain * away_y
            cand_speed = hypot(cand_x, cand_y)
            current_speed = hypot(vx, vy)

            # Prefer the side that immediately increases clearance from the
            # obstacle/person, while still making goal progress when possible.
            clearance_gain = cand_x * away_x + cand_y * away_y
            goal_progress = cand_x * goal_dir_x + cand_y * goal_dir_y
            cost = -1.6 * clearance_gain - 0.45 * goal_progress

            if cand_speed > 1e-4 and current_speed > 1e-4:
                turn_cos = (cand_x * vx + cand_y * vy) / max(cand_speed * current_speed, 1e-6)
                cost += 0.25 * (1.0 - max(-1.0, min(1.0, turn_cos)))

            if previous is not None and sign != previous:
                cost += 0.12
            return cost

        left_score = score(1.0)
        right_score = score(-1.0)
        candidate = 1.0 if left_score <= right_score else -1.0

        previous = self._last_avoid_side.get(agent_id)
        if previous is not None:
            prev_score = left_score if previous > 0.0 else right_score
            cand_score = left_score if candidate > 0.0 else right_score
            if cand_score > prev_score - 0.08:
                candidate = previous

        self._last_avoid_side[agent_id] = candidate
        return candidate

    def _limit_lateral_velocity(self, agent: Agent, goals: List, vx: float, vy: float) -> Tuple[float, float]:
        if not goals:
            return vx, vy

        ax = float(agent.position.position.x)
        ay = float(agent.position.position.y)
        gx = float(goals[0].position.x)
        gy = float(goals[0].position.y)
        dx = gx - ax
        dy = gy - ay
        dist = hypot(dx, dy)
        if dist <= max(float(agent.goal_radius), 0.1) + 0.1:
            return vx, vy

        goal_x = dx / max(dist, 1e-6)
        goal_y = dy / max(dist, 1e-6)
        side_x = -goal_y
        side_y = goal_x
        forward = vx * goal_x + vy * goal_y
        lateral = vx * side_x + vy * side_y
        desired = max(0.0, min(float(agent.desired_velocity), self.max_speed))
        near_obstacle = self._closest_obstacle_distance(agent) < self.obstacle_avoidance_distance
        lateral_ratio = self.obstacle_lateral_speed_ratio if near_obstacle else self.max_lateral_speed_ratio
        max_lateral = max(0.2, desired * max(0.0, min(lateral_ratio, 1.0)))
        lateral = max(-max_lateral, min(max_lateral, lateral))
        if not near_obstacle and forward < self.min_goal_speed * 0.35:
            forward = self.min_goal_speed * 0.35
        return forward * goal_x + lateral * side_x, forward * goal_y + lateral * side_y

    def _closest_obstacle_distance(self, agent: Agent) -> float:
        ax = float(agent.position.position.x)
        ay = float(agent.position.position.y)
        best = float("inf")
        for obs in agent.closest_obs:
            best = min(best, hypot(ax - float(obs.x), ay - float(obs.y)))
        return best

    def _apply_goal_arrival_control(
        self,
        agent: Agent,
        goals: List,
        vx: float,
        vy: float,
        dt: float,
    ) -> Tuple[float, float]:
        if not goals:
            return vx, vy

        ax = float(agent.position.position.x)
        ay = float(agent.position.position.y)
        gx = float(goals[0].position.x)
        gy = float(goals[0].position.y)
        dx = gx - ax
        dy = gy - ay
        dist = hypot(dx, dy)
        goal_radius = max(float(agent.goal_radius), 0.1)
        if dist <= goal_radius + 0.03:
            self._last_cmd_vel[int(agent.id)] = (0.0, 0.0)
            return 0.0, 0.0

        arrival_distance = max(goal_radius + 0.05, self.goal_arrival_distance)
        if dist > arrival_distance:
            return vx, vy
        if self._closest_obstacle_distance(agent) < goal_radius + self.obstacle_collision_buffer:
            return vx, vy

        desired_speed = max(0.0, min(float(agent.desired_velocity), self.max_speed))
        remaining = max(0.0, dist - goal_radius * 0.5)
        no_overshoot_speed = remaining / max(dt, 1e-3)
        slowdown_speed = max(self.goal_arrival_min_speed, desired_speed * dist / max(arrival_distance, 1e-6))
        speed = min(desired_speed, slowdown_speed, no_overshoot_speed)
        if speed <= 1e-3:
            return 0.0, 0.0

        return speed * dx / dist, speed * dy / dist

    def _near_hard_hazard(self, agent: Agent, all_agents: List[Agent]) -> bool:
        ax = float(agent.position.position.x)
        ay = float(agent.position.position.y)
        radius = max(float(agent.radius), 0.0)

        for other in all_agents:
            if other.id == agent.id:
                continue
            clearance = hypot(ax - float(other.position.position.x), ay - float(other.position.position.y))
            clearance -= radius + max(float(other.radius), 0.0)
            if clearance < self.collision_buffer + 0.12:
                return True

        for obs in agent.closest_obs:
            if hypot(ax - float(obs.x), ay - float(obs.y)) < self.obstacle_collision_buffer:
                return True

        return False

    def _resolve_obstacle_collision_velocity(
        self,
        agent: Agent,
        vx: float,
        vy: float,
        dt: float,
    ) -> Tuple[float, float]:
        ax = float(agent.position.position.x)
        ay = float(agent.position.position.y)
        next_x = ax + vx * dt
        next_y = ay + vy * dt
        min_dist = max(self.obstacle_collision_buffer, float(agent.radius) + 0.12)
        correction_x = 0.0
        correction_y = 0.0

        for obs in agent.closest_obs:
            ox = float(obs.x)
            oy = float(obs.y)
            dx = next_x - ox
            dy = next_y - oy
            dist = hypot(dx, dy)
            if dist >= min_dist:
                continue
            if dist <= 1e-4:
                dx = ax - ox
                dy = ay - oy
                dist = hypot(dx, dy)
                if dist <= 1e-4:
                    dx, dy, dist = 1.0, 0.0, 1.0
            push = min_dist - dist
            correction_x += push * dx / dist
            correction_y += push * dy / dist

            obs_dir_x = (ox - ax) / max(hypot(ox - ax, oy - ay), 1e-6)
            obs_dir_y = (oy - ay) / max(hypot(ox - ax, oy - ay), 1e-6)
            toward = vx * obs_dir_x + vy * obs_dir_y
            if toward > 0.0:
                correction_x -= toward * dt * obs_dir_x
                correction_y -= toward * dt * obs_dir_y

        if hypot(correction_x, correction_y) <= 1e-6:
            return vx, vy
        return vx + correction_x / max(dt, 1e-3), vy + correction_y / max(dt, 1e-3)

    def _active_occupancy_map(self) -> Optional[OccupancyMapProvider]:
        predictor = self._spubert_predictor
        if predictor is None:
            return None
        return predictor.occupancy_map

    @staticmethod
    def _disk_offsets(radius: float) -> List[Tuple[float, float]]:
        radius = max(float(radius), 0.0)
        if radius <= 1e-3:
            return [(0.0, 0.0)]
        diag = radius * 0.70710678
        return [
            (0.0, 0.0),
            (radius, 0.0),
            (-radius, 0.0),
            (0.0, radius),
            (0.0, -radius),
            (diag, diag),
            (diag, -diag),
            (-diag, diag),
            (-diag, -diag),
        ]

    def _map_position_free(self, x: float, y: float, radius: float) -> bool:
        occupancy = self._active_occupancy_map()
        if occupancy is None:
            return True
        for ox, oy in self._disk_offsets(radius):
            if occupancy.point_occupied(float(x) + ox, float(y) + oy):
                return False
        return True

    def _resolve_map_collision_velocity(
        self,
        agent: Agent,
        vx: float,
        vy: float,
        dt: float,
    ) -> Tuple[float, float]:
        occupancy = self._active_occupancy_map()
        if occupancy is None:
            return vx, vy

        ax = float(agent.position.position.x)
        ay = float(agent.position.position.y)
        next_x = ax + vx * dt
        next_y = ay + vy * dt
        radius = max(float(agent.radius) + 0.08, self.spubert_map_collision_radius)
        if self._map_position_free(next_x, next_y, radius):
            return vx, vy

        original_next = np.asarray([next_x, next_y], dtype=np.float32)
        speed = hypot(vx, vy)
        step = max(speed * max(dt, 1e-3), 0.35)
        best: Optional[Tuple[float, float]] = None
        best_score = float("inf")
        for ring in (step, step + 0.25, step + 0.5, step + 0.8):
            for idx in range(24):
                theta = 2.0 * pi * idx / 24.0
                cx = ax + ring * cos(theta)
                cy = ay + ring * sin(theta)
                if not self._map_position_free(cx, cy, radius):
                    continue
                candidate = np.asarray([cx, cy], dtype=np.float32)
                score = float(np.linalg.norm(candidate - original_next))
                if score < best_score:
                    best_score = score
                    best = (cx, cy)
            if best is not None:
                break

        if best is None:
            # Last-resort gradient away from occupied samples around the next pose.
            push_x = 0.0
            push_y = 0.0
            for ox, oy in self._disk_offsets(max(radius, 0.35)):
                sx = next_x + ox
                sy = next_y + oy
                if occupancy.point_occupied(sx, sy):
                    dx = next_x - sx
                    dy = next_y - sy
                    dist = hypot(dx, dy)
                    if dist <= 1e-4:
                        dx, dy, dist = ax - sx, ay - sy, hypot(ax - sx, ay - sy)
                    if dist <= 1e-4:
                        dx, dy, dist = 1.0, 0.0, 1.0
                    push_x += dx / dist
                    push_y += dy / dist
            norm = hypot(push_x, push_y)
            if norm <= 1e-6:
                return 0.0, 0.0
            return vx + 0.6 * push_x / norm / max(dt, 1e-3), vy + 0.6 * push_y / norm / max(dt, 1e-3)

        return (best[0] - ax) / max(dt, 1e-3), (best[1] - ay) / max(dt, 1e-3)

    def _resolve_agent_collision_velocity(
        self,
        agent: Agent,
        all_agents: List[Agent],
        vx: float,
        vy: float,
        dt: float,
    ) -> Tuple[float, float]:
        ax = float(agent.position.position.x)
        ay = float(agent.position.position.y)
        next_x = ax + vx * dt
        next_y = ay + vy * dt
        radius = max(float(agent.radius), 0.0)
        correction_x = 0.0
        correction_y = 0.0

        for other in all_agents:
            if other.id == agent.id:
                continue
            ox = float(other.position.position.x)
            oy = float(other.position.position.y)
            ovx = float(other.velocity.linear.x)
            ovy = float(other.velocity.linear.y)
            other_next_x = ox + ovx * dt
            other_next_y = oy + ovy * dt
            dx = next_x - other_next_x
            dy = next_y - other_next_y
            dist = hypot(dx, dy)
            min_dist = radius + max(float(other.radius), 0.0) + self.collision_buffer
            if dist >= min_dist:
                continue
            if dist <= 1e-4:
                dx = ax - ox
                dy = ay - oy
                dist = hypot(dx, dy)
                if dist <= 1e-4:
                    dx, dy, dist = 1.0, 0.0, 1.0
            push = min_dist - dist
            correction_x += push * dx / dist
            correction_y += push * dy / dist

        if hypot(correction_x, correction_y) <= 1e-6:
            return vx, vy
        return vx + correction_x / max(dt, 1e-3), vy + correction_y / max(dt, 1e-3)

    def _limit_velocity(self, agent_id: int, vx: float, vy: float, desired_velocity: float, dt: float) -> Tuple[float, float]:
        vx, vy = self._clamp_speed_only(vx, vy, desired_velocity)

        prev = self._last_cmd_vel.get(agent_id)
        if prev is None:
            prev_vx, prev_vy = vx, vy
        else:
            prev_vx, prev_vy = prev
            alpha = max(0.0, min(self.velocity_smoothing_alpha, 1.0))
            vx = (1.0 - alpha) * prev_vx + alpha * vx
            vy = (1.0 - alpha) * prev_vy + alpha * vy

        max_delta = self.max_accel * max(dt, 1e-3)
        dvx = vx - prev_vx
        dvy = vy - prev_vy
        delta = hypot(dvx, dvy)
        if delta > max_delta:
            vx = prev_vx + dvx * max_delta / delta
            vy = prev_vy + dvy * max_delta / delta

        self._last_cmd_vel[agent_id] = (vx, vy)
        return vx, vy

    def _clamp_speed_only(self, vx: float, vy: float, desired_velocity: float) -> Tuple[float, float]:
        max_speed = max(0.05, min(float(desired_velocity) if desired_velocity > 0.0 else self.max_speed, self.max_speed))
        speed = hypot(vx, vy)
        if speed > max_speed:
            scale = max_speed / speed
            return vx * scale, vy * scale
        return vx, vy

    def _limit_heading_rate(
        self,
        agent_id: int,
        current_yaw: float,
        vx: float,
        vy: float,
        dt: float,
    ) -> Tuple[float, float, float]:
        speed = hypot(vx, vy)
        if speed <= 0.03:
            yaw = self._last_yaw.get(agent_id, current_yaw)
            self._last_yaw[agent_id] = yaw
            return yaw, vx, vy

        desired_yaw = atan2(vy, vx)
        prev_yaw = self._last_yaw.get(agent_id, current_yaw)
        max_turn = max(0.05, self.max_yaw_rate) * max(dt, 1e-3)
        yaw_error = self._angle_diff(desired_yaw, prev_yaw)
        yaw_error = max(-max_turn, min(max_turn, yaw_error))
        yaw = self._normalize_angle(prev_yaw + yaw_error)
        self._last_yaw[agent_id] = yaw
        return yaw, speed * cos(yaw), speed * sin(yaw)

    def _limit_yaw_only(
        self,
        agent_id: int,
        current_yaw: float,
        desired_yaw: float,
        dt: float,
    ) -> float:
        prev_yaw = self._last_yaw.get(agent_id, current_yaw)
        max_turn = max(0.05, self.max_yaw_rate) * max(dt, 1e-3)
        yaw_error = self._angle_diff(desired_yaw, prev_yaw)
        yaw_error = max(-max_turn, min(max_turn, yaw_error))
        yaw = self._normalize_angle(prev_yaw + yaw_error)
        self._last_yaw[agent_id] = yaw
        return yaw

    def _compute_dt(self, stamp: float) -> float:
        if self._last_stamp is None:
            self._last_stamp = stamp
            return self.fallback_dt
        dt = stamp - self._last_stamp
        self._last_stamp = stamp
        if dt <= 0.0 or dt > 1.0:
            return self.fallback_dt
        return dt

    def _stamp_to_float(self, agents: Agents) -> float:
        stamp = float(agents.header.stamp.sec) + float(agents.header.stamp.nanosec) * 1e-9
        if stamp <= 0.0:
            stamp = self.get_clock().now().nanoseconds * 1e-9
        return stamp

    @staticmethod
    def _yaw_to_quaternion(yaw: float) -> Quaternion:
        quat = Quaternion()
        quat.z = sin(yaw * 0.5)
        quat.w = cos(yaw * 0.5)
        return quat

    @staticmethod
    def _agent_yaw(agent: Agent) -> float:
        quat = agent.position.orientation
        siny_cosp = 2.0 * (float(quat.w) * float(quat.z) + float(quat.x) * float(quat.y))
        cosy_cosp = 1.0 - 2.0 * (float(quat.y) * float(quat.y) + float(quat.z) * float(quat.z))
        return atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        diff = SocialBertComputeAgentsNode._normalize_angle(a) - SocialBertComputeAgentsNode._normalize_angle(b)
        while diff > 3.141592653589793:
            diff -= 6.283185307179586
        while diff < -3.141592653589793:
            diff += 6.283185307179586
        return diff

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > 3.141592653589793:
            angle -= 6.283185307179586
        while angle < -3.141592653589793:
            angle += 6.283185307179586
        return angle

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)


def main(args: List[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SocialBertComputeAgentsNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

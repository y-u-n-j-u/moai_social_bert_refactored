from __future__ import annotations

import importlib
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


XY = Tuple[float, float]


def seed_runtime_rng(seed: int, torch_module: Any) -> None:
    """Seed the stochastic MGP sampler once for a reproducible run."""
    normalized_seed = int(seed)
    random.seed(normalized_seed)
    np.random.seed(normalized_seed)
    torch_module.manual_seed(normalized_seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed(normalized_seed)
        torch_module.cuda.manual_seed_all(normalized_seed)
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class GuidedInferenceResult:
    path_world: List[XY]
    candidate_goals_world: List[XY]
    selected_goal_world: XY
    guidance_point_world: XY
    selected_goal_valid: bool
    trajectory_map_safe: bool
    execution_valid: bool
    candidate_safe_mask: Tuple[bool, ...] = ()
    candidate_rank: int = 1
    candidate_index: int = -1
    guidance_distance_m: float = math.inf


def guidance_point(current: XY, final_goal: XY, radius: float) -> XY:
    dx = float(final_goal[0]) - float(current[0])
    dy = float(final_goal[1]) - float(current[1])
    distance = math.hypot(dx, dy)
    if distance <= 1e-8:
        return float(final_goal[0]), float(final_goal[1])
    step = min(max(float(radius), 0.0), distance)
    return (
        float(current[0]) + dx * step / distance,
        float(current[1]) + dy * step / distance,
    )


def guidance_point_along_path(
    path: Sequence[XY],
    current: XY,
    final_goal: XY,
    radius: float,
) -> XY:
    """Select a point up to ``radius`` metres ahead on a global path."""
    points = [(float(x), float(y)) for x, y in path]
    if not points:
        raise ValueError("global path is empty")
    if len(points) == 1:
        return float(final_goal[0]), float(final_goal[1])

    current_x = float(current[0])
    current_y = float(current[1])
    best_distance_sq = math.inf
    best_segment = 0
    best_point = points[0]

    # Segment projection keeps lookahead stable on sparse Nav2 plans.
    for index, (start, end) in enumerate(zip(points, points[1:])):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-12:
            fraction = 0.0
        else:
            fraction = max(
                0.0,
                min(
                    1.0,
                    ((current_x - start[0]) * dx + (current_y - start[1]) * dy)
                    / length_sq,
                ),
            )
        projected = (start[0] + fraction * dx, start[1] + fraction * dy)
        distance_sq = (
            (current_x - projected[0]) ** 2 + (current_y - projected[1]) ** 2
        )
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_segment = index
            best_point = projected

    remaining = max(float(radius), 0.0)
    segment_start = best_point
    segment_end = points[best_segment + 1]
    first_length = math.hypot(
        segment_end[0] - segment_start[0],
        segment_end[1] - segment_start[1],
    )
    if remaining <= first_length and first_length > 1e-12:
        ratio = remaining / first_length
        return (
            segment_start[0] + ratio * (segment_end[0] - segment_start[0]),
            segment_start[1] + ratio * (segment_end[1] - segment_start[1]),
        )
    remaining -= first_length

    for index in range(best_segment + 1, len(points) - 1):
        start = points[index]
        end = points[index + 1]
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        if remaining <= length and length > 1e-12:
            ratio = remaining / length
            return (
                start[0] + ratio * (end[0] - start[0]),
                start[1] + ratio * (end[1] - start[1]),
            )
        remaining -= length

    return float(final_goal[0]), float(final_goal[1])


def sample_polyline(points: Sequence[XY], max_spacing: float) -> List[XY]:
    """Densely sample every segment, including both polyline endpoints."""
    spacing = float(max_spacing)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("max_spacing must be a positive finite value")

    normalized = [(float(x), float(y)) for x, y in points]
    if not normalized:
        return []

    sampled = [normalized[0]]
    for start, end in zip(normalized, normalized[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.hypot(dx, dy)
        steps = max(1, int(math.ceil(distance / spacing)))
        sampled.extend(
            (
                start[0] + dx * step / steps,
                start[1] + dy * step / steps,
            )
            for step in range(1, steps + 1)
        )
    return sampled


def pad_history(points: Sequence[XY], length: int) -> List[XY]:
    result = [(float(x), float(y)) for x, y in points[-max(int(length), 1) :]]
    if not result:
        result = [(0.0, 0.0)]
    while len(result) < length:
        result.insert(0, result[0])
    return result[-length:]


def heading_from_history(points: Sequence[XY], fallback_yaw: float) -> float:
    if len(points) >= 2:
        dx = float(points[-1][0]) - float(points[-2][0])
        dy = float(points[-1][1]) - float(points[-2][1])
        if math.hypot(dx, dy) > 1e-3:
            return math.atan2(dy, dx)
    return float(fallback_yaw)


def world_to_local(point: XY, origin: XY, theta: float) -> XY:
    dx = float(point[0]) - float(origin[0])
    dy = float(point[1]) - float(origin[1])
    ct = math.cos(theta)
    st = math.sin(theta)
    return ct * dx + st * dy, -st * dx + ct * dy


def local_to_world(point: XY, origin: XY, theta: float) -> XY:
    ct = math.cos(theta)
    st = math.sin(theta)
    return (
        float(origin[0]) + ct * float(point[0]) - st * float(point[1]),
        float(origin[1]) + st * float(point[0]) + ct * float(point[1]),
    )


class GuidedSpubertRuntime:
    """Load the guidance-conditioned SPU-BERT and build live Gazebo inputs."""

    def __init__(
        self,
        *,
        repo_path: str,
        config_path: str,
        checkpoint_path: str,
        map_provider: Any,
        use_cuda: bool,
        d_sample: int,
        runtime_seed: int,
        guidance_radius: float,
        tgp_top_k: int,
        footprint_radius: float,
        logger: Any,
    ) -> None:
        self.repo_path = os.path.abspath(os.path.expanduser(repo_path))
        self.config_path = os.path.abspath(os.path.expanduser(config_path))
        self.checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
        self.map_provider = map_provider
        self.guidance_radius = float(guidance_radius)
        self.runtime_seed = int(runtime_seed)
        self.tgp_top_k = max(int(tgp_top_k), 1)
        self.footprint_radius = max(float(footprint_radius), 0.0)
        self.logger = logger

        for label, path, predicate in (
            ("model repository", self.repo_path, os.path.isdir),
            ("model config", self.config_path, os.path.isfile),
            ("model checkpoint", self.checkpoint_path, os.path.isfile),
        ):
            if not predicate(path):
                raise RuntimeError(f"{label} does not exist: {path}")
        if map_provider is None:
            raise RuntimeError("guided SPU-BERT requires a static occupancy map")

        spubert_root = os.path.join(self.repo_path, "SPU-BERT")
        # Keep the refactored repository ahead of ROS/workspace compatibility
        # packages that may expose the same generic names (spubert/configs/src).
        for path_entry in (self.repo_path, spubert_root):
            if path_entry not in sys.path:
                sys.path.insert(0, path_entry)

        self._discard_incompatible_module("spubert", spubert_root)
        self._discard_incompatible_module("configs", self.repo_path)
        self._discard_incompatible_module("src", self.repo_path)

        try:
            import torch
            from configs.loader import load_runtime_namespace
            from spubert.datasets.moai_social_nav_extended_goal import (
                _SpatialTokens,
                build_mgp_tgp_streams,
            )
            from spubert.model import SBertPlusFTConfig, SBertPlusFTModel
            from spubert.training import _build_spubert_mgp_config, _build_spubert_tgp_config
        except Exception as exc:
            raise RuntimeError(f"guided SPU-BERT imports failed: {exc}") from exc

        self.torch = torch
        self._SpatialTokens = _SpatialTokens
        self._build_streams = build_mgp_tgp_streams
        torch.set_num_threads(1)

        args = load_runtime_namespace(self.config_path, command="test", cli_dry_run=False)
        if not bool(getattr(args, "guidance_conditioned", False)):
            raise RuntimeError("model config must enable guidance_conditioned")
        if not bool(getattr(args, "scene", False)):
            raise RuntimeError("model config must enable the scene branch")

        self.args = args
        self.obs_len = int(args.obs_len)
        self.pred_len = int(args.pred_len)
        self.num_nbr = int(args.num_nbr)
        self.view_range = float(args.view_range)
        self.view_angle = float(args.view_angle)
        self.social_range = float(args.social_range)
        self.patch_size = int(args.patch_size)
        self.local_map_size_m = float(args.local_map_size_m)
        self.local_map_grid_size = int(args.local_map_grid_size)
        self.env_resol = float(args.env_resol)
        self.d_sample = max(int(d_sample), int(args.k_sample))
        self.seq_input_len = (
            (self.obs_len + 1) * (self.num_nbr + 1) + self.pred_len + 1
        )

        expected_resolution = self.local_map_size_m / max(self.local_map_grid_size, 1)
        if not math.isclose(self.env_resol, expected_resolution, rel_tol=1e-5, abs_tol=1e-6):
            raise RuntimeError(
                "model map resolution mismatch: "
                f"env_resol={self.env_resol}, expected={expected_resolution}"
            )
        if self.local_map_grid_size % self.patch_size != 0:
            raise RuntimeError("local map grid size must be divisible by patch size")
        self.side_patches = self.local_map_grid_size // self.patch_size
        if self.side_patches * self.side_patches != int(args.num_patch):
            raise RuntimeError(
                f"scene patch mismatch: config={args.num_patch}, runtime={self.side_patches ** 2}"
            )

        tgp_cfg = _build_spubert_tgp_config(args)
        mgp_cfg = _build_spubert_mgp_config(args)
        model_cfg = SBertPlusFTConfig(tgp_cfg, mgp_cfg, share=args.share)
        self.model = SBertPlusFTModel(tgp_cfg, mgp_cfg, model_cfg)
        state = dict(
            self._load_state_dict(
                torch.load(self.checkpoint_path, map_location="cpu")
            )
        )
        # Hugging Face versions differ in whether the deterministic
        # embeddings.position_ids buffer is persisted in checkpoints.  It is
        # not a learned weight, so recover only that buffer from the freshly
        # constructed model while keeping strict loading for everything else.
        generated_position_ids = []
        for key, value in self.model.state_dict().items():
            if key.endswith(".embeddings.position_ids") and key not in state:
                state[key] = value
                generated_position_ids.append(key)
        if generated_position_ids:
            logger.info(
                "Generated non-learned position_ids buffers missing from checkpoint: "
                + ", ".join(generated_position_ids)
            )
        self.model.load_state_dict(state, strict=True)

        self.device = torch.device(
            "cuda" if bool(use_cuda) and torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()
        # Seed after model construction/loading so inference sampling does not
        # depend on implementation details of model initialization.
        seed_runtime_rng(self.runtime_seed, torch)
        logger.info(
            "Loaded guidance-conditioned robot SPU-BERT "
            f"from {self.checkpoint_path} on {self.device}; "
            f"runtime_seed={self.runtime_seed}"
        )

    @staticmethod
    def _discard_incompatible_module(name: str, expected_root: str) -> None:
        loaded = sys.modules.get(name)
        loaded_file = os.path.abspath(getattr(loaded, "__file__", "") or "")
        if loaded_file and not loaded_file.startswith(os.path.abspath(expected_root)):
            # This bridge is its own ROS process.  Removing a stale package and
            # its children here is safe and lets the imports below resolve from
            # model_repo_path instead of a workspace-level compatibility shim.
            for module_name in tuple(sys.modules):
                if module_name == name or module_name.startswith(f"{name}."):
                    del sys.modules[module_name]
            importlib.invalidate_caches()

    @staticmethod
    def _load_state_dict(payload: Any) -> Mapping[str, Any]:
        state = payload
        if isinstance(state, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                candidate = state.get(key)
                if isinstance(candidate, dict):
                    state = candidate
                    break
        if not isinstance(state, dict):
            raise RuntimeError("checkpoint does not contain a model state_dict")
        if state and all(str(key).startswith("module.") for key in state):
            state = {str(key)[7:]: value for key, value in state.items()}
        return state

    def predict(
        self,
        *,
        robot_history: Sequence[XY],
        robot_yaw: float,
        human_histories: Mapping[int, Sequence[XY]],
        final_goal: XY,
        guidance_point_world: Optional[XY] = None,
    ) -> GuidedInferenceResult:
        return self.predict_candidates(
            robot_history=robot_history,
            robot_yaw=robot_yaw,
            human_histories=human_histories,
            final_goal=final_goal,
            guidance_point_world=guidance_point_world,
        )[0]

    def predict_candidates(
        self,
        *,
        robot_history: Sequence[XY],
        robot_yaw: float,
        human_histories: Mapping[int, Sequence[XY]],
        final_goal: XY,
        guidance_point_world: Optional[XY] = None,
    ) -> List[GuidedInferenceResult]:
        robot_world = pad_history(robot_history, self.obs_len)
        origin = robot_world[-1]
        theta = heading_from_history(robot_world, robot_yaw)
        target_local = [world_to_local(point, origin, theta) for point in robot_world]

        neighbors: List[Tuple[float, List[XY]]] = []
        for points in human_histories.values():
            history = pad_history(points, self.obs_len)
            local = [world_to_local(point, origin, theta) for point in history]
            distance = math.hypot(local[-1][0], local[-1][1])
            angle = abs(math.atan2(local[-1][1], local[-1][0]))
            visible = distance <= self.social_range or angle <= self.view_angle * 0.5
            if distance < self.view_range and visible:
                neighbors.append((distance, local))
        neighbors.sort(key=lambda item: item[0])

        rows = 1 + min(len(neighbors), self.num_nbr)
        trajectories = np.full(
            (rows, self.obs_len + self.pred_len, 2),
            np.nan,
            dtype=np.float32,
        )
        trajectories[0, : self.obs_len] = np.asarray(target_local, dtype=np.float32)
        for row, (_, history) in enumerate(neighbors[: self.num_nbr], start=1):
            trajectories[row, : self.obs_len] = np.asarray(history, dtype=np.float32)

        gp_world = (
            (float(guidance_point_world[0]), float(guidance_point_world[1]))
            if guidance_point_world is not None
            else guidance_point(origin, final_goal, self.guidance_radius)
        )
        gp_local = np.asarray(world_to_local(gp_world, origin, theta), dtype=np.float32)
        streams = self._build_streams(
            trajs=trajectories,
            goal_lbl=np.zeros(2, dtype=np.float32),
            guidance_lbl=gp_local,
            obs_len=self.obs_len,
            pred_len=self.pred_len,
            num_nbr=self.num_nbr,
            seq_input_len=self.seq_input_len,
            spatial=self._SpatialTokens(self.view_range),
        )
        scene = self._build_scene(origin=origin, theta=theta)
        batch = self._tensor_batch(streams, scene, gp_local)

        with self.torch.no_grad():
            output = self.model.inference_guided_candidates(
                mgp_spatial_ids=batch["mgp_spatial_ids"],
                mgp_temporal_ids=batch["mgp_temporal_ids"],
                mgp_segment_ids=batch["mgp_segment_ids"],
                mgp_attn_mask=batch["mgp_attn_mask"],
                tgp_temporal_ids=batch["tgp_temporal_ids"],
                tgp_segment_ids=batch["tgp_segment_ids"],
                tgp_attn_mask=batch["tgp_attn_mask"],
                guidance_points=batch["guidance_lbl"],
                env_spatial_ids=batch["env_spatial_ids"],
                env_temporal_ids=batch["env_temporal_ids"],
                env_segment_ids=batch["env_segment_ids"],
                env_attn_mask=batch["env_attn_mask"],
                envs=batch["envs"],
                envs_params=batch["envs_params"],
                d_sample=self.d_sample,
                reject_unknown=bool(self.args.reject_unknown_goals),
                top_k=self.tgp_top_k,
                candidate_safety_fn=lambda goals: self._candidate_footprint_safe_mask(
                    goals,
                    origin=origin,
                    theta=theta,
                ),
            )

        paths_local = output["guided_pred_trajs"][0].detach().cpu().numpy()
        candidates_local = output["candidate_goals"][0].detach().cpu().numpy()
        selected_local = output["guided_candidate_goals"][0].detach().cpu().numpy()
        candidate_indices = output["guided_candidate_indices"][0].detach().cpu().numpy()
        goal_valid = output["guided_candidate_goal_valid"][0].detach().cpu().numpy()
        guidance_distances = output[
            "guided_candidate_guidance_distances"
        ][0].detach().cpu().numpy()
        trajectory_map_safe = output[
            "guided_trajectory_map_safe"
        ][0].detach().cpu().numpy()
        execution_valid = output["guided_execution_valid"][0].detach().cpu().numpy()
        candidate_safe = output["candidate_safe_mask"][0].detach().cpu().numpy()
        candidates_world = []
        candidate_safe_world = []
        for index, point in enumerate(candidates_local):
            if not np.isfinite(point[:2]).all():
                continue
            candidates_world.append(
                local_to_world((float(point[0]), float(point[1])), origin, theta)
            )
            candidate_safe_world.append(bool(candidate_safe[index]))
        valid_goal_count = int(np.count_nonzero(goal_valid))
        attempt_count = max(valid_goal_count, 1)
        results = []
        for rank in range(min(attempt_count, len(paths_local))):
            path_world = [
                local_to_world((float(point[0]), float(point[1])), origin, theta)
                for point in paths_local[rank]
            ]
            selected_world = local_to_world(
                (float(selected_local[rank][0]), float(selected_local[rank][1])),
                origin,
                theta,
            )
            results.append(
                GuidedInferenceResult(
                    path_world=path_world,
                    candidate_goals_world=candidates_world,
                    selected_goal_world=selected_world,
                    guidance_point_world=gp_world,
                    selected_goal_valid=bool(goal_valid[rank]),
                    trajectory_map_safe=bool(trajectory_map_safe[rank]),
                    execution_valid=bool(execution_valid[rank]),
                    candidate_safe_mask=tuple(candidate_safe_world),
                    candidate_rank=rank + 1,
                    candidate_index=int(candidate_indices[rank]),
                    guidance_distance_m=float(guidance_distances[rank]),
                )
            )
        return results

    def _candidate_footprint_safe_mask(
        self,
        candidate_goals: Any,
        *,
        origin: XY,
        theta: float,
    ) -> Any:
        candidates = candidate_goals.detach().cpu().numpy()
        if candidates.ndim != 3 or candidates.shape[0] != 1:
            raise ValueError(
                "live candidate footprint filtering expects shape (1,K,2+)"
            )
        mask = np.zeros(candidates.shape[:2], dtype=np.bool_)
        for index, point in enumerate(candidates[0]):
            if not np.isfinite(point[:2]).all():
                continue
            world = local_to_world(
                (float(point[0]), float(point[1])),
                origin,
                theta,
            )
            mask[0, index] = self.map_provider.path_collision_cost(
                [world],
                radius=self.footprint_radius,
                weight=1.0,
            ) <= 0.0
        return self.torch.as_tensor(
            mask,
            dtype=self.torch.bool,
            device=candidate_goals.device,
        )

    def _build_scene(self, *, origin: XY, theta: float) -> Dict[str, np.ndarray]:
        patches, patch_mask = self.map_provider.scene_patches(
            trans=(-float(origin[0]), -float(origin[1])),
            theta=float(theta),
            env_range=0.5 * self.local_map_size_m,
            env_resol=self.env_resol,
            patch_size=self.patch_size,
            side_patches=self.side_patches,
        )
        grid = np.zeros(
            (self.local_map_grid_size, self.local_map_grid_size),
            dtype=np.float32,
        )
        patch_index = 0
        for row in range(self.side_patches):
            for col in range(self.side_patches):
                patch = patches[patch_index].reshape(self.patch_size, self.patch_size)
                row_start = row * self.patch_size
                col_start = col * self.patch_size
                grid[
                    row_start : row_start + self.patch_size,
                    col_start : col_start + self.patch_size,
                ] = patch
                patch_index += 1

        return {
            "env_spatial_ids": patches.astype(np.float32),
            "env_segment_ids": np.arange(
                self.num_nbr + 2,
                self.num_nbr + 2 + len(patches),
                dtype=np.int64,
            ),
            "env_temporal_ids": np.full(len(patches), self.obs_len, dtype=np.int64),
            "env_attn_mask": patch_mask.astype(np.float32),
            "envs": grid,
            "envs_params": np.asarray(
                [
                    -0.5 * self.local_map_size_m,
                    -0.5 * self.local_map_size_m,
                    float(self.local_map_grid_size),
                    float(self.local_map_grid_size),
                    self.env_resol,
                    2.0,
                ],
                dtype=np.float32,
            ),
        }

    def _tensor_batch(
        self,
        streams: Mapping[str, np.ndarray],
        scene: Mapping[str, np.ndarray],
        guidance_local: np.ndarray,
    ) -> Dict[str, Any]:
        torch = self.torch
        float_keys = {
            "mgp_spatial_ids",
            "mgp_attn_mask",
            "tgp_spatial_ids",
            "tgp_attn_mask",
            "env_spatial_ids",
            "env_attn_mask",
            "envs",
            "envs_params",
        }
        values: Dict[str, np.ndarray] = {**streams, **scene}
        values["guidance_lbl"] = guidance_local
        result: Dict[str, Any] = {}
        for key, value in values.items():
            dtype = torch.float if key in float_keys or key == "guidance_lbl" else torch.long
            result[key] = torch.tensor(value, dtype=dtype, device=self.device).unsqueeze(0)
        return result

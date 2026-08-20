#!/usr/bin/env python3
"""Post-process HuNav pedestrian trajectory PKLs into model-ready samples."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np


def _parse_simple_yaml(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith("["):
            out[key] = [float(v.strip()) for v in value.strip("[]").split(",") if v.strip()]
        else:
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value.strip("\"'")
    return out


def _read_pgm_token(stream) -> bytes:
    token = bytearray()
    while True:
        char = stream.read(1)
        if not char:
            return bytes(token)
        if char == b"#" and not token:
            stream.readline()
            continue
        if char.isspace():
            if token:
                return bytes(token)
            continue
        token.extend(char)


def _read_pgm(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic = _read_pgm_token(f)
        width = int(_read_pgm_token(f))
        height = int(_read_pgm_token(f))
        max_value = int(_read_pgm_token(f))
        if magic == b"P2":
            pixels = np.asarray(
                [int(_read_pgm_token(f)) for _ in range(width * height)],
                dtype=np.float32,
            )
        elif magic == b"P5":
            dtype = np.uint8 if max_value < 256 else np.dtype(">u2")
            pixels = np.frombuffer(f.read(), dtype=dtype).astype(np.float32)
        else:
            raise ValueError(f"Unsupported PGM format {magic!r}: {path}")
    if pixels.size != width * height:
        raise ValueError(f"PGM size mismatch: expected {width * height}, got {pixels.size}")
    pixels = pixels.reshape(height, width)
    return pixels / float(max_value)


def load_map(map_yaml: Path) -> dict[str, Any]:
    info = _parse_simple_yaml(map_yaml)
    image_path = Path(str(info["image"]))
    if not image_path.is_absolute():
        image_path = map_yaml.parent / image_path
    gray = _read_pgm(image_path)
    # Preserve the Gazebo adapter's source encoding:
    # 0=free, 0.5=unknown, 1=occupied. The model adapter later maps this to
    # 0=unknown/padding, 1=free, 2=occupied.
    negate = bool(int(info.get("negate", 0)))
    occupancy_probability = gray if negate else (1.0 - gray)
    occupied_thresh = float(info.get("occupied_thresh", 0.65))
    free_thresh = float(info.get("free_thresh", 0.196))
    occupied = np.full(gray.shape, 0.5, dtype=np.float32)
    occupied[occupancy_probability > occupied_thresh] = 1.0
    occupied[occupancy_probability < free_thresh] = 0.0
    return {
        "yaml_path": str(map_yaml),
        "image_path": str(image_path),
        "occupied": occupied,
        "resolution": float(info.get("resolution", 0.05)),
        "origin": tuple(float(v) for v in info.get("origin", [0.0, 0.0, 0.0])[:3]),
    }


def world_to_pixel(x: float, y: float, map_data: dict[str, Any]) -> tuple[int, int]:
    occ = map_data["occupied"]
    resolution = map_data["resolution"]
    origin_x, origin_y, _ = map_data["origin"]
    col = int(round((x - origin_x) / resolution))
    row = int(round(occ.shape[0] - 1 - (y - origin_y) / resolution))
    return row, col


def local_map_patch(
    center_xy: np.ndarray,
    map_data: dict[str, Any],
    size_m: float,
    grid_size: int,
) -> np.ndarray:
    occ = map_data["occupied"]
    resolution = map_data["resolution"]
    raw_size = max(1, int(round(size_m / resolution)))
    center_row, center_col = world_to_pixel(float(center_xy[0]), float(center_xy[1]), map_data)
    half = raw_size // 2

    # Cells outside the source map are unknown, not occupied.
    crop = np.full((raw_size, raw_size), 0.5, dtype=np.float32)
    src_r0 = max(0, center_row - half)
    src_r1 = min(occ.shape[0], center_row - half + raw_size)
    src_c0 = max(0, center_col - half)
    src_c1 = min(occ.shape[1], center_col - half + raw_size)

    dst_r0 = src_r0 - (center_row - half)
    dst_c0 = src_c0 - (center_col - half)
    crop[dst_r0 : dst_r0 + (src_r1 - src_r0), dst_c0 : dst_c0 + (src_c1 - src_c0)] = occ[
        src_r0:src_r1, src_c0:src_c1
    ]

    patch = np.zeros((grid_size, grid_size), dtype=np.float32)
    edges = np.linspace(0, raw_size, grid_size + 1).round().astype(int)
    for r in range(grid_size):
        for c in range(grid_size):
            block = crop[edges[r] : edges[r + 1], edges[c] : edges[c + 1]]
            patch[r, c] = float(block.max()) if block.size else 1.0
    return patch


def pixel_to_world(row: int, col: int, map_data: dict[str, Any]) -> np.ndarray:
    occ = map_data["occupied"]
    resolution = float(map_data["resolution"])
    origin_x, origin_y, _ = map_data["origin"]
    return np.asarray(
        [
            float(origin_x) + float(col) * resolution,
            float(origin_y) + float(occ.shape[0] - 1 - row) * resolution,
        ],
        dtype=np.float32,
    )


def inflate_occupancy_grid(
    occupied: np.ndarray,
    resolution: float,
    clearance_m: float,
) -> np.ndarray:
    """Return cells forbidden to the robot center, including unknown space."""
    blocked = np.asarray(occupied, dtype=np.float32) >= 0.5
    radius_cells = max(0, int(math.ceil(float(clearance_m) / float(resolution))))
    if radius_cells == 0:
        return blocked.copy()

    height, width = blocked.shape
    inflated = blocked.copy()
    radius_sq = radius_cells * radius_cells
    for row_delta in range(-radius_cells, radius_cells + 1):
        for col_delta in range(-radius_cells, radius_cells + 1):
            if row_delta * row_delta + col_delta * col_delta > radius_sq:
                continue
            src_r0 = max(0, -row_delta)
            src_r1 = min(height, height - row_delta)
            src_c0 = max(0, -col_delta)
            src_c1 = min(width, width - col_delta)
            dst_r0 = src_r0 + row_delta
            dst_r1 = src_r1 + row_delta
            dst_c0 = src_c0 + col_delta
            dst_c1 = src_c1 + col_delta
            inflated[dst_r0:dst_r1, dst_c0:dst_c1] |= blocked[
                src_r0:src_r1,
                src_c0:src_c1,
            ]

    # A footprint extending outside the known map is unsafe as well.
    inflated[:radius_cells, :] = True
    inflated[-radius_cells:, :] = True
    inflated[:, :radius_cells] = True
    inflated[:, -radius_cells:] = True
    return inflated


def coarsen_map_for_planning(
    map_data: dict[str, Any],
    planning_resolution_m: float,
) -> dict[str, Any]:
    source_resolution = float(map_data["resolution"])
    factor = max(1, int(round(float(planning_resolution_m) / source_resolution)))
    if factor == 1:
        return dict(map_data)

    occupied = np.asarray(map_data["occupied"], dtype=np.float32)
    height, width = occupied.shape
    padded_height = int(math.ceil(height / factor) * factor)
    padded_width = int(math.ceil(width / factor) * factor)
    padded = np.full((padded_height, padded_width), 0.5, dtype=np.float32)
    padded[:height, :width] = occupied
    coarse = padded.reshape(
        padded_height // factor,
        factor,
        padded_width // factor,
        factor,
    ).max(axis=(1, 3))
    return {
        **map_data,
        "occupied": coarse,
        "resolution": source_resolution * factor,
        "planning_downsample_factor": factor,
    }


class OccupancyGridRoutePlanner:
    """Nav2-compatible 8-connected shortest paths on an inflated static map."""

    MOVES = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )

    def __init__(
        self,
        map_data: dict[str, Any],
        clearance_m: float,
        cache_size: int = 32,
        planning_resolution_m: float = 0.10,
    ) -> None:
        self.source_map_data = map_data
        self.map_data = coarsen_map_for_planning(
            map_data,
            planning_resolution_m,
        )
        self.resolution = float(self.map_data["resolution"])
        self.blocked = inflate_occupancy_grid(
            self.map_data["occupied"],
            self.resolution,
            clearance_m,
        )
        self.clearance_m = float(clearance_m)
        self.cache_size = max(1, int(cache_size))
        self._goal_fields: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()

    def is_safe_world(self, xy: np.ndarray) -> bool:
        row, col = world_to_pixel(float(xy[0]), float(xy[1]), self.map_data)
        return (
            0 <= row < self.blocked.shape[0]
            and 0 <= col < self.blocked.shape[1]
            and not bool(self.blocked[row, col])
        )

    def _nearest_free_cell(
        self,
        xy: np.ndarray,
        max_snap_distance_m: float,
        label: str,
    ) -> tuple[tuple[int, int], float]:
        row, col = world_to_pixel(float(xy[0]), float(xy[1]), self.map_data)
        height, width = self.blocked.shape
        if not (0 <= row < height and 0 <= col < width):
            raise ValueError(f"{label} is outside the occupancy map")
        if not self.blocked[row, col]:
            cell_xy = pixel_to_world(row, col, self.map_data)
            return (row, col), float(np.linalg.norm(cell_xy - xy[:2]))

        max_snap_distance_m = max(0.0, float(max_snap_distance_m))
        radius_cells = int(math.ceil(max_snap_distance_m / self.resolution))
        best_cell: tuple[int, int] | None = None
        best_distance = math.inf
        for candidate_row in range(max(0, row - radius_cells), min(height, row + radius_cells + 1)):
            for candidate_col in range(max(0, col - radius_cells), min(width, col + radius_cells + 1)):
                if self.blocked[candidate_row, candidate_col]:
                    continue
                candidate_xy = pixel_to_world(candidate_row, candidate_col, self.map_data)
                distance = float(np.linalg.norm(candidate_xy - xy[:2]))
                if distance <= max_snap_distance_m + 1e-6 and distance < best_distance:
                    best_cell = (candidate_row, candidate_col)
                    best_distance = distance
        if best_cell is None:
            raise ValueError(
                f"{label} has no footprint-safe cell within "
                f"{max_snap_distance_m:.3f} m"
            )
        return best_cell, best_distance

    def _goal_field(self, goal: tuple[int, int]) -> np.ndarray:
        cached = self._goal_fields.pop(goal, None)
        if cached is not None:
            self._goal_fields[goal] = cached
            return cached

        height, width = self.blocked.shape
        # Keep exact-enough costs for heap stale-entry checks. float32 rounding
        # can repeatedly reinsert long diagonal paths and make this search
        # orders of magnitude slower on a 5-10 cm grid.
        distances = np.full((height, width), np.inf, dtype=np.float64)
        next_cell = np.full((height, width), -1, dtype=np.int32)
        goal_row, goal_col = goal
        goal_index = goal_row * width + goal_col
        distances[goal_row, goal_col] = 0.0
        next_cell[goal_row, goal_col] = goal_index
        frontier: list[tuple[float, int, int]] = [(0.0, goal_row, goal_col)]

        while frontier:
            distance, row, col = heapq.heappop(frontier)
            if distance > float(distances[row, col]) + 1e-12:
                continue
            current_index = row * width + col
            for row_delta, col_delta, step_cost in self.MOVES:
                next_row = row + row_delta
                next_col = col + col_delta
                if not (0 <= next_row < height and 0 <= next_col < width):
                    continue
                if self.blocked[next_row, next_col]:
                    continue
                if row_delta != 0 and col_delta != 0:
                    if self.blocked[next_row, col] or self.blocked[row, next_col]:
                        continue
                candidate_distance = distance + step_cost
                if candidate_distance + 1e-12 >= float(distances[next_row, next_col]):
                    continue
                distances[next_row, next_col] = candidate_distance
                next_cell[next_row, next_col] = current_index
                heapq.heappush(
                    frontier,
                    (candidate_distance, next_row, next_col),
                )

        self._goal_fields[goal] = next_cell
        while len(self._goal_fields) > self.cache_size:
            self._goal_fields.popitem(last=False)
        return next_cell

    def route(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        start_snap_distance_m: float,
        goal_snap_distance_m: float,
    ) -> tuple[list[np.ndarray], dict[str, float]]:
        start_xy = np.asarray(start_xy, dtype=np.float32)[:2]
        goal_xy = np.asarray(goal_xy, dtype=np.float32)[:2]
        start, start_snap = self._nearest_free_cell(
            start_xy,
            start_snap_distance_m,
            "route start",
        )
        goal, goal_snap = self._nearest_free_cell(
            goal_xy,
            goal_snap_distance_m,
            "final goal",
        )
        field = self._goal_field(goal)
        height, width = self.blocked.shape
        route_cells = [start]
        current = start
        for _ in range(height * width):
            if current == goal:
                break
            next_index = int(field[current])
            if next_index < 0:
                raise ValueError("no collision-free occupancy-grid route to final goal")
            next_row, next_col = divmod(next_index, width)
            next_value = (int(next_row), int(next_col))
            if next_value == current:
                raise ValueError("occupancy-grid route contains a loop")
            route_cells.append(next_value)
            current = next_value
        else:
            raise ValueError("occupancy-grid route exceeded map cell count")

        route_points = [start_xy]
        for row, col in route_cells:
            point = pixel_to_world(row, col, self.map_data)
            if float(np.linalg.norm(point - route_points[-1])) > 1e-6:
                route_points.append(point)
        if float(np.linalg.norm(goal_xy - route_points[-1])) > 1e-6:
            route_points.append(goal_xy)
        metrics = {
            "route_start_snap_m": float(start_snap),
            "route_goal_snap_m": float(goal_snap),
            "route_path_length_m": path_length(np.asarray(route_points)),
            "route_path_pose_count": float(len(route_points)),
            "route_planning_resolution_m": float(self.resolution),
        }
        return route_points, metrics


def guidance_point_along_route(
    route_points: list[np.ndarray],
    radius: float,
) -> np.ndarray:
    if not route_points:
        raise ValueError("route has no points")
    remaining = max(0.0, float(radius))
    current = np.asarray(route_points[0], dtype=np.float32)[:2]
    if remaining <= 0.0:
        return current
    for raw_next in route_points[1:]:
        next_point = np.asarray(raw_next, dtype=np.float32)[:2]
        segment = next_point - current
        segment_length = float(np.linalg.norm(segment))
        if segment_length <= 1e-9:
            current = next_point
            continue
        if segment_length >= remaining:
            return (current + segment * (remaining / segment_length)).astype(np.float32)
        remaining -= segment_length
        current = next_point
    return np.asarray(route_points[-1], dtype=np.float32)[:2]


def trajectory_footprint_metrics(
    points: np.ndarray,
    planner: OccupancyGridRoutePlanner,
) -> dict[str, float]:
    """Check both recorded poses and the swept path between poses."""
    points = np.asarray(points, dtype=np.float32)
    pose_safe = [planner.is_safe_world(point) for point in points]
    sample_spacing = max(float(planner.resolution) * 0.5, 0.025)
    swept_safe: list[bool] = []
    if len(points):
        swept_safe.append(planner.is_safe_world(points[0]))
    for start, end in zip(points[:-1], points[1:]):
        segment = end - start
        segment_length = float(np.linalg.norm(segment))
        sample_count = max(1, int(math.ceil(segment_length / sample_spacing)))
        for sample_index in range(1, sample_count + 1):
            point = start + segment * (sample_index / sample_count)
            swept_safe.append(planner.is_safe_world(point))

    pose_collision_count = sum(not safe for safe in pose_safe)
    swept_collision_count = sum(not safe for safe in swept_safe)
    return {
        "target_pose_collision_count": float(pose_collision_count),
        "target_swept_collision_count": float(swept_collision_count),
        "target_swept_sample_count": float(len(swept_safe)),
        "target_map_safe": 1.0 if swept_collision_count == 0 else 0.0,
    }


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def min_target_neighbor_distance(trajs: np.ndarray, obs_len: int) -> float:
    if trajs.shape[0] <= 1:
        return math.inf
    target_obs = trajs[0, :obs_len]
    neighbor_obs = trajs[1:, :obs_len]
    d = np.linalg.norm(neighbor_obs - target_obs[None, :, :], axis=2)
    d = d[np.isfinite(d)]
    return float(d.min()) if d.size else math.inf


def local_map_metrics(
    center_xy: np.ndarray,
    map_data: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, float]:
    patch = local_map_patch(center_xy, map_data, args.map_size_m, args.map_grid_size)
    finite = patch[np.isfinite(patch)]
    if finite.size == 0:
        return {
            "local_occupancy_ratio": 0.0,
            "local_occupied_cells": 0.0,
        }
    return {
        "local_occupancy_ratio": float(finite.mean()),
        "local_occupied_cells": float(finite.sum()),
    }


def _clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def heuristic_quality_scores(q: dict[str, float], args: argparse.Namespace) -> dict[str, float]:
    motion_score = 0.5 * _clip01(q.get("future_disp", 0.0) / max(args.min_future_disp * 3.0, 1e-6))
    motion_score += 0.5 * _clip01(q.get("mean_speed", 0.0) / 1.0)

    interaction_distance = q.get("same_time_min_distance_all", math.inf)
    if not math.isfinite(interaction_distance):
        interaction_distance = q.get("min_social_distance_obs", math.inf)
    interaction_score = _clip01(
        (args.max_social_distance - interaction_distance)
        / max(args.max_social_distance - args.min_social_distance, 1e-6)
    )
    if math.isfinite(interaction_distance) and interaction_distance < args.collision_distance:
        interaction_score *= 0.5

    future_distance = q.get("same_time_min_distance_future", math.inf)
    if not math.isfinite(future_distance):
        future_distance = interaction_distance
    crossing_score = _clip01(
        (args.max_social_distance - future_distance)
        / max(args.max_social_distance - args.collision_distance, 1e-6)
    )

    occupancy_ratio = q.get("local_occupancy_ratio", 0.0)
    map_presence = _clip01(occupancy_ratio / 0.03)
    map_not_saturated = _clip01((0.45 - occupancy_ratio) / 0.20)
    map_score = map_presence * map_not_saturated

    quality_score = (
        0.35 * motion_score
        + 0.35 * interaction_score
        + 0.20 * crossing_score
        + 0.10 * map_score
    )
    return {
        "motion_score": float(motion_score),
        "interaction_score": float(interaction_score),
        "crossing_score": float(crossing_score),
        "map_score": float(map_score),
        "quality_score": float(quality_score),
    }


def same_time_robot_human_metrics(
    trajs: np.ndarray,
    obs_len: int,
    pred_len: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    if trajs.shape[0] <= 1:
        return {
            "same_time_min_distance_obs": math.inf,
            "same_time_min_distance_future": math.inf,
            "same_time_min_distance_all": math.inf,
            "human_future_available": 0.0,
            "collision_risk": 0.0,
        }

    seq_len = obs_len + pred_len
    target = trajs[0, :seq_len]
    neighbors = trajs[1:, :seq_len]
    distance = np.linalg.norm(neighbors - target[None, :, :], axis=2)
    finite = np.isfinite(distance)

    obs_distance = distance[:, :obs_len]
    obs_finite = finite[:, :obs_len]
    fut_distance = distance[:, obs_len:seq_len]
    fut_finite = finite[:, obs_len:seq_len]

    obs_min = float(obs_distance[obs_finite].min()) if obs_finite.any() else math.inf
    fut_min = float(fut_distance[fut_finite].min()) if fut_finite.any() else math.inf
    all_min = float(distance[finite].min()) if finite.any() else math.inf
    threshold = float(args.collision_distance)
    # If human futures are recorded, use future+observed distance. For older
    # files with NaN human futures, this falls back to observed distance only.
    risk_distance = min(obs_min, fut_min) if fut_finite.any() else obs_min
    return {
        "same_time_min_distance_obs": obs_min,
        "same_time_min_distance_future": fut_min,
        "same_time_min_distance_all": all_min,
        "human_future_available": 1.0 if fut_finite.any() else 0.0,
        "collision_risk": 1.0 if risk_distance <= threshold else 0.0,
    }


def _recent_velocity(points: np.ndarray, dt: float, window: int) -> np.ndarray | None:
    points = np.asarray(points, dtype=np.float32)
    finite = np.isfinite(points).all(axis=1)
    valid_indices = np.flatnonzero(finite)
    if valid_indices.size < 2:
        return None
    end = int(valid_indices[-1])
    start = max(0, end - max(1, int(window)))
    segment = points[start : end + 1]
    if len(segment) < 2 or not np.isfinite(segment).all():
        return None
    elapsed = max(float(dt) * (len(segment) - 1), 1e-6)
    return (segment[-1] - segment[0]) / elapsed


def avoidance_conflict_metrics(
    trajs: np.ndarray,
    obs_len: int,
    pred_len: int,
    dt: float,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Estimate a pre-response constant-velocity conflict and robot slowdown."""
    if trajs.shape[0] <= 1:
        return {
            "cv_min_distance_m": math.inf,
            "cv_time_to_closest_s": math.inf,
            "cv_closing_speed_mps": 0.0,
            "cv_conflict_human_index": -1.0,
            "robot_observed_speed_mps": 0.0,
            "robot_response_min_speed_mps": 0.0,
            "robot_slowdown_ratio": 0.0,
            "avoidance_conflict": 0.0,
        }

    dt = max(float(dt), 1e-6)
    target_obs = trajs[0, :obs_len]
    target_velocity = _recent_velocity(
        target_obs,
        dt,
        args.avoidance_velocity_window,
    )
    if target_velocity is None or not np.isfinite(target_obs[-1]).all():
        target_velocity = np.zeros(2, dtype=np.float32)
    target_speed = float(np.linalg.norm(target_velocity))
    horizon = min(
        float(pred_len) * dt,
        max(float(args.avoidance_max_ttc), 0.0),
    )

    best_distance = math.inf
    best_ttc = math.inf
    best_closing_speed = 0.0
    best_human_index = -1
    for human_index, neighbor in enumerate(trajs[1:, :obs_len]):
        neighbor_velocity = _recent_velocity(
            neighbor,
            dt,
            args.avoidance_velocity_window,
        )
        if neighbor_velocity is None or not np.isfinite(neighbor[-1]).all():
            continue
        relative_position = neighbor[-1] - target_obs[-1]
        relative_velocity = neighbor_velocity - target_velocity
        relative_speed_sq = float(np.dot(relative_velocity, relative_velocity))
        if relative_speed_sq <= 1e-8:
            ttc = 0.0
        else:
            ttc = float(
                np.clip(
                    -float(np.dot(relative_position, relative_velocity))
                    / relative_speed_sq,
                    0.0,
                    horizon,
                )
            )
        closest_vector = relative_position + relative_velocity * ttc
        distance = float(np.linalg.norm(closest_vector))
        current_distance = max(float(np.linalg.norm(relative_position)), 1e-6)
        closing_speed = max(
            0.0,
            -float(np.dot(relative_position, relative_velocity)) / current_distance,
        )
        if distance < best_distance:
            best_distance = distance
            best_ttc = ttc
            best_closing_speed = closing_speed
            best_human_index = human_index

    future = trajs[0, obs_len : obs_len + pred_len]
    response_points = np.vstack([target_obs[-1], future])
    finite_response = np.isfinite(response_points).all(axis=1)
    response_points = response_points[finite_response]
    response_speeds = (
        np.linalg.norm(np.diff(response_points, axis=0), axis=1) / dt
        if len(response_points) >= 2
        else np.asarray([], dtype=np.float32)
    )
    if response_speeds.size:
        response_steps = max(
            int(args.avoidance_response_window),
            int(math.ceil((best_ttc + float(args.avoidance_response_extra_time)) / dt))
            if math.isfinite(best_ttc)
            else int(args.avoidance_response_window),
        )
        response_speeds = response_speeds[: max(1, response_steps)]
        window = min(max(1, int(args.avoidance_response_window)), len(response_speeds))
        rolling = np.convolve(response_speeds, np.ones(window) / window, mode="valid")
        response_min_speed = float(rolling.min())
    else:
        response_min_speed = 0.0
    slowdown_ratio = (
        _clip01((target_speed - response_min_speed) / target_speed)
        if target_speed >= args.avoidance_min_reference_speed
        else 0.0
    )
    conflict = (
        math.isfinite(best_distance)
        and best_distance <= args.avoidance_conflict_distance
        and args.avoidance_min_ttc <= best_ttc <= args.avoidance_max_ttc
        and best_closing_speed >= args.avoidance_min_closing_speed
    )
    return {
        "cv_min_distance_m": float(best_distance),
        "cv_time_to_closest_s": float(best_ttc),
        "cv_closing_speed_mps": float(best_closing_speed),
        "cv_conflict_human_index": float(best_human_index),
        "robot_observed_speed_mps": float(target_speed),
        "robot_response_min_speed_mps": float(response_min_speed),
        "robot_slowdown_ratio": float(slowdown_ratio),
        "avoidance_conflict": 1.0 if conflict else 0.0,
    }


def point_to_polyline_distances(points: np.ndarray, route: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    route = np.asarray(route, dtype=np.float32).reshape(-1, 2)
    if not len(points):
        return np.asarray([], dtype=np.float32)
    if not len(route):
        return np.full(len(points), math.inf, dtype=np.float32)
    if len(route) == 1:
        return np.linalg.norm(points - route[0], axis=1)

    starts = route[:-1]
    segments = route[1:] - starts
    segment_norm_sq = np.sum(segments * segments, axis=1)
    relative = points[:, None, :] - starts[None, :, :]
    projection = np.divide(
        np.sum(relative * segments[None, :, :], axis=2),
        segment_norm_sq[None, :],
        out=np.zeros((len(points), len(segments)), dtype=np.float32),
        where=segment_norm_sq[None, :] > 1e-8,
    )
    projection = np.clip(projection, 0.0, 1.0)
    closest = starts[None, :, :] + projection[:, :, None] * segments[None, :, :]
    return np.linalg.norm(points[:, None, :] - closest, axis=2).min(axis=1)


def route_response_metrics(
    target: np.ndarray,
    obs_len: int,
    pred_len: int,
    route_points: np.ndarray,
) -> dict[str, float]:
    future = np.asarray(target[obs_len : obs_len + pred_len], dtype=np.float32)
    future = future[np.isfinite(future).all(axis=1)]
    distances = point_to_polyline_distances(future, route_points)
    return {
        "route_future_mean_deviation_m": (
            float(distances.mean()) if distances.size else math.inf
        ),
        "route_future_max_deviation_m": (
            float(distances.max()) if distances.size else math.inf
        ),
    }


def avoidance_classification(
    q: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, float]:
    slowdown = q.get("robot_slowdown_ratio", 0.0) >= args.avoidance_min_slowdown_ratio
    route_deviation = (
        q.get("route_future_max_deviation_m", -math.inf)
        >= args.avoidance_min_route_deviation
    )
    response = slowdown or route_deviation
    safe_outcome = q.get("same_time_min_distance_all", -math.inf) >= args.min_social_distance
    conflict = q.get("avoidance_conflict", 0.0) >= 1.0
    conflict_score = _clip01(
        (args.avoidance_conflict_distance - q.get("cv_min_distance_m", math.inf))
        / max(args.avoidance_conflict_distance, 1e-6)
    )
    response_score = max(
        _clip01(
            q.get("robot_slowdown_ratio", 0.0)
            / max(args.avoidance_min_slowdown_ratio, 1e-6)
        ),
        _clip01(
            q.get("route_future_max_deviation_m", 0.0)
            / max(args.avoidance_min_route_deviation, 1e-6)
        ),
    )
    return {
        "avoidance_slowdown": 1.0 if slowdown else 0.0,
        "avoidance_route_deviation": 1.0 if route_deviation else 0.0,
        "avoidance_response": 1.0 if response else 0.0,
        "avoidance_safe_outcome": 1.0 if safe_outcome else 0.0,
        "avoidance_reactive": 1.0 if conflict and response and safe_outcome else 0.0,
        "avoidance_score": float(min(conflict_score, response_score)) if safe_outcome else 0.0,
    }


def teacher_avoidance_classification(
    meta: dict[str, Any],
    q: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, float]:
    """Verify anticipatory teacher avoidance using recorded intervention state."""
    raw_active = meta.get("human_avoidance_active_any", False)
    intervention_active = (
        raw_active
        if isinstance(raw_active, bool)
        else str(raw_active).strip().lower() in {"1", "true", "yes", "on"}
    )
    continuous_mode = (
        str(meta.get("human_avoidance_mode", "")).strip().lower()
        == "continuous"
    )
    response = q.get("avoidance_response", 0.0) >= 1.0
    safe_outcome = q.get("avoidance_safe_outcome", 0.0) >= 1.0
    verified = intervention_active and continuous_mode and response and safe_outcome
    existing_reactive = q.get("avoidance_reactive", 0.0) >= 1.0
    return {
        "avoidance_teacher_intervention": 1.0 if intervention_active else 0.0,
        "avoidance_teacher_verified": 1.0 if verified else 0.0,
        "avoidance_reactive": 1.0 if existing_reactive or verified else 0.0,
        "avoidance_score": max(
            float(q.get("avoidance_score", 0.0)),
            1.0 if verified else 0.0,
        ),
    }


def sample_quality(trajs: np.ndarray, obs_len: int, pred_len: int, dt: float) -> dict[str, float]:
    target = trajs[0]
    obs = target[:obs_len]
    fut = target[obs_len : obs_len + pred_len]
    full = target[: obs_len + pred_len]
    total_path = path_length(full)
    total_disp = float(np.linalg.norm(full[-1] - full[0]))
    obs_disp = float(np.linalg.norm(obs[-1] - obs[0]))
    fut_disp = float(np.linalg.norm(fut[-1] - fut[0]))
    step = np.linalg.norm(np.diff(full, axis=0), axis=1)
    safe_dt = max(dt, 1e-6)
    velocity = np.diff(full, axis=0) / safe_dt
    speeds = np.linalg.norm(velocity, axis=1)
    accelerations = (
        np.linalg.norm(np.diff(velocity, axis=0), axis=1) / safe_dt
        if len(velocity) >= 2
        else np.asarray([], dtype=np.float32)
    )
    headings = np.arctan2(velocity[:, 1], velocity[:, 0]) if len(velocity) else np.asarray([])
    valid_heading = step >= 0.02
    yaw_rates: list[float] = []
    for index in range(1, len(headings)):
        if not (valid_heading[index - 1] and valid_heading[index]):
            continue
        angle_delta = math.atan2(
            math.sin(float(headings[index] - headings[index - 1])),
            math.cos(float(headings[index] - headings[index - 1])),
        )
        yaw_rates.append(abs(angle_delta) / safe_dt)
    return {
        "obs_disp": obs_disp,
        "future_disp": fut_disp,
        "total_disp": total_disp,
        "path_length": total_path,
        "path_efficiency": total_disp / total_path if total_path > 1e-6 else 0.0,
        "mean_speed": float(speeds.mean()) if speeds.size else 0.0,
        "max_speed": float(speeds.max()) if speeds.size else 0.0,
        "max_acceleration": (
            float(accelerations.max()) if accelerations.size else 0.0
        ),
        "max_yaw_rate": max(yaw_rates, default=0.0),
        "min_social_distance_obs": min_target_neighbor_distance(trajs, obs_len),
        "neighbor_count": float(max(0, trajs.shape[0] - 1)),
    }


def sample_timing_metrics(
    meta: dict[str, Any],
    seq_len: int,
    expected_dt: float,
) -> dict[str, float]:
    expected_dt = max(float(expected_dt), 1e-6)
    intervals = np.asarray(meta.get("frame_intervals_s", []), dtype=np.float64).reshape(-1)
    intervals = intervals[np.isfinite(intervals)]
    if intervals.size == 0:
        try:
            start_stamp = float(meta["start_stamp"])
            end_stamp = float(meta["end_stamp"])
            if seq_len > 1 and math.isfinite(start_stamp) and math.isfinite(end_stamp):
                intervals = np.asarray(
                    [(end_stamp - start_stamp) / (seq_len - 1)],
                    dtype=np.float64,
                )
        except (KeyError, TypeError, ValueError):
            pass
    if intervals.size == 0:
        return {
            "timing_available": 0.0,
            "mean_frame_interval_s": math.nan,
            "max_frame_interval_error_s": math.nan,
        }
    return {
        "timing_available": 1.0,
        "mean_frame_interval_s": float(intervals.mean()),
        "max_frame_interval_error_s": float(np.max(np.abs(intervals - expected_dt))),
    }


def between_humans_metrics(
    trajs: np.ndarray,
    obs_len: int,
    pred_len: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    target = trajs[0, : obs_len + pred_len]
    if target.shape[0] < obs_len + pred_len or not np.isfinite(target).all():
        return {
            "between_humans": 0.0,
            "between_humans_gap": math.inf,
            "between_humans_forward_gap": math.inf,
            "between_humans_candidates": 0.0,
        }

    start = target[obs_len - 1].astype(np.float32)
    end = target[obs_len + pred_len - 1].astype(np.float32)
    path_vec = end - start
    path_dist = float(np.linalg.norm(path_vec))
    if path_dist < args.between_min_path_length:
        return {
            "between_humans": 0.0,
            "between_humans_gap": math.inf,
            "between_humans_forward_gap": math.inf,
            "between_humans_candidates": 0.0,
        }

    forward = path_vec / max(path_dist, 1e-6)
    normal = np.asarray([-forward[1], forward[0]], dtype=np.float32)
    candidates: list[tuple[float, float]] = []
    for neighbor in trajs[1:, :obs_len]:
        finite = np.isfinite(neighbor[:, 0]) & np.isfinite(neighbor[:, 1])
        if not finite.any():
            continue
        pos = neighbor[np.flatnonzero(finite)[-1]].astype(np.float32)
        rel = pos - start
        forward_pos = float(np.dot(rel, forward))
        lateral_pos = float(np.dot(rel, normal))
        lateral_abs = abs(lateral_pos)
        if (
            -args.between_behind_margin <= forward_pos <= path_dist + args.between_ahead_margin
            and args.between_min_side_distance <= lateral_abs <= args.between_max_side_distance
        ):
            candidates.append((forward_pos, lateral_pos))

    best_gap = math.inf
    best_forward_gap = math.inf
    for i, (s_i, l_i) in enumerate(candidates):
        for s_j, l_j in candidates[i + 1 :]:
            if l_i * l_j >= 0.0:
                continue
            forward_gap = abs(s_i - s_j)
            lateral_gap = abs(l_i) + abs(l_j)
            if (
                forward_gap <= args.between_max_forward_gap
                and args.between_min_gap <= lateral_gap <= args.between_max_gap
            ):
                if forward_gap < best_forward_gap or (
                    math.isclose(forward_gap, best_forward_gap) and lateral_gap < best_gap
                ):
                    best_forward_gap = forward_gap
                    best_gap = lateral_gap

    return {
        "between_humans": 1.0 if math.isfinite(best_gap) else 0.0,
        "between_humans_gap": best_gap,
        "between_humans_forward_gap": best_forward_gap,
        "between_humans_candidates": float(len(candidates)),
    }


def basic_filter_reasons(q: dict[str, float], args: argparse.Namespace) -> list[str]:
    checks = (
        (q["neighbor_count"] >= args.min_neighbors, "too_few_neighbors"),
        (q["obs_disp"] >= args.min_obs_disp, "insufficient_observed_motion"),
        (q["future_disp"] >= args.min_future_disp, "insufficient_future_motion"),
        (q["path_length"] >= args.min_path_length, "short_path"),
        (q["path_efficiency"] >= args.min_path_efficiency, "low_path_efficiency"),
        # The model learns all 20 synchronized positions. Ground truth must be
        # socially and geometrically safe across observation and prediction.
        (
            q["same_time_min_distance_all"] >= args.min_social_distance,
            "human_clearance_violation",
        ),
        (
            q["min_social_distance_obs"] >= args.min_social_distance,
            "observed_human_clearance_violation",
        ),
        (q["max_speed"] <= args.max_speed, "speed_limit"),
        (q["max_acceleration"] <= args.max_acceleration, "acceleration_limit"),
        (q["max_yaw_rate"] <= args.max_yaw_rate, "yaw_rate_limit"),
        (q.get("target_map_safe", 0.0) >= 1.0, "target_map_collision"),
        (
            q.get("timing_available", 0.0) < 1.0
            or q.get("max_frame_interval_error_s", math.inf)
            <= args.max_frame_interval_error,
            "irregular_frame_interval",
        ),
    )
    return [reason for passed, reason in checks if not passed]


def pass_basic_filter(q: dict[str, float], args: argparse.Namespace) -> bool:
    return not basic_filter_reasons(q, args)


def pass_social_filter(q: dict[str, float], args: argparse.Namespace) -> bool:
    return pass_basic_filter(q, args) and q["min_social_distance_obs"] <= args.max_social_distance


def pass_between_humans_filter(q: dict[str, float], args: argparse.Namespace) -> bool:
    return pass_social_filter(q, args) and q.get("between_humans", 0.0) >= 1.0


def pass_avoidance_filter(q: dict[str, float], args: argparse.Namespace) -> bool:
    return pass_basic_filter(q, args) and q.get("avoidance_reactive", 0.0) >= 1.0


def guidance_point_from_goal(current_xy: np.ndarray, final_goal: np.ndarray, radius: float) -> np.ndarray:
    current_xy = np.asarray(current_xy, dtype=np.float32)[:2]
    final_goal = np.asarray(final_goal, dtype=np.float32)[:2]
    delta = final_goal - current_xy
    dist = float(np.linalg.norm(delta))
    if dist <= 1e-6:
        return final_goal.astype(np.float32)
    step = min(max(float(radius), 0.0), dist)
    return (current_xy + delta / dist * step).astype(np.float32)


def xy_from_meta(meta: dict[str, Any], key: str) -> np.ndarray | None:
    if key not in meta:
        return None
    value = np.asarray(meta[key], dtype=np.float32).reshape(-1)
    if value.size < 2 or not np.isfinite(value[:2]).all():
        return None
    return value[:2].astype(np.float32)


def make_model_sample(
    trajs: np.ndarray,
    meta: dict[str, Any],
    quality: dict[str, float],
    obs_len: int,
    pred_len: int,
    map_data: dict[str, Any],
    args: argparse.Namespace,
    source_index: int,
    guidance_point_override: np.ndarray | None = None,
    guidance_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target_past = trajs[0, :obs_len].astype(np.float32)
    target_future = trajs[0, obs_len : obs_len + pred_len].astype(np.float32)
    neighbor_past = trajs[1:, :obs_len].astype(np.float32)
    neighbor_future = trajs[1:, obs_len : obs_len + pred_len].astype(np.float32)
    neighbor_mask = np.isfinite(neighbor_past[..., 0]).any(axis=1).astype(np.float32)
    local_map = local_map_patch(target_past[-1], map_data, args.map_size_m, args.map_grid_size)
    final_goal = xy_from_meta(meta, "final_goal")
    if final_goal is None:
        final_goal = target_future[-1].astype(np.float32)
    guidance_point = guidance_point_override
    if guidance_point is None:
        guidance_point = xy_from_meta(meta, "guidance_point")
    if guidance_point is None:
        guidance_point = guidance_point_from_goal(
            target_past[-1],
            final_goal,
            args.guidance_radius,
        )
    guidance_point = np.asarray(guidance_point, dtype=np.float32)[:2]
    guidance_traj = np.linspace(target_past[-1], guidance_point, pred_len).astype(np.float32)
    sample_meta = dict(meta, source_index=source_index)
    if guidance_metadata:
        sample_meta.update(guidance_metadata)
    sample_meta["guidance_point"] = [
        float(guidance_point[0]),
        float(guidance_point[1]),
    ]
    return {
        "target_past": target_past,
        "neighbor_past": neighbor_past,
        "neighbor_future": neighbor_future,
        "neighbor_mask": neighbor_mask,
        "guidance_point": guidance_point.astype(np.float32),
        "guidance_traj": guidance_traj,
        "guidance_mask": np.ones((pred_len,), dtype=np.float32),
        "final_goal": final_goal,
        "local_map": local_map.astype(np.float32),
        "target_future": target_future,
        "meta": sample_meta,
        "quality": quality,
    }


def summarize(qualities: list[dict[str, float]]) -> dict[str, Any]:
    if not qualities:
        return {}
    keys = sorted({key for q in qualities for key in q.keys()})
    out: dict[str, Any] = {"count": len(qualities)}
    for key in keys:
        raw_values = []
        for q in qualities:
            try:
                raw_values.append(float(q.get(key, math.nan)))
            except (TypeError, ValueError):
                continue
        values = np.asarray(raw_values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            out[key] = {
                "finite_count": 0,
                "min": None,
                "mean": None,
                "median": None,
                "max": None,
            }
            continue
        out[key] = {
            "finite_count": int(values.size),
            "min": float(np.min(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "max": float(np.max(values)),
        }
    return out


def write_quality_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _finite_metric_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, math.nan))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def _numeric_metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for row in rows for key in row.keys()})
    summary: dict[str, Any] = {"count": len(rows), "metrics": {}}
    for key in keys:
        values = _finite_metric_values(rows, key)
        if values.size == 0:
            continue
        summary["metrics"][key] = {
            "finite_count": int(values.size),
            "min": float(np.min(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)),
            "max": float(np.max(values)),
        }
    return summary


def _hist(ax: Any, rows: list[dict[str, Any]], key: str, title: str, bins: int = 32) -> None:
    values = _finite_metric_values(rows, key)
    if values.size == 0:
        ax.text(0.5, 0.5, "no finite values", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.hist(values, bins=min(bins, max(6, int(math.sqrt(values.size)) + 1)), color="#2563eb", alpha=0.82)
        ax.axvline(float(np.median(values)), color="#dc2626", linewidth=1.5)
    ax.set_title(title)
    ax.grid(True, alpha=0.22)


def _format_kpi(value: float | int | None, suffix: str = "", decimals: int = 1) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    if suffix == "%":
        return f"{numeric:.{decimals}f}%"
    if abs(numeric - round(numeric)) < 1e-9 and not suffix:
        return str(int(round(numeric)))
    return f"{numeric:.{decimals}f}{suffix}"


def _rate(count: int, total: int) -> float:
    return 100.0 * float(count) / float(total) if total > 0 else 0.0


def _count_where(rows: list[dict[str, Any]], key: str, op: str, threshold: float) -> int:
    values = _finite_metric_values(rows, key)
    if values.size == 0:
        return 0
    if op == ">=":
        return int(np.count_nonzero(values >= threshold))
    if op == "<=":
        return int(np.count_nonzero(values <= threshold))
    if op == ">":
        return int(np.count_nonzero(values > threshold))
    if op == "<":
        return int(np.count_nonzero(values < threshold))
    raise ValueError(f"Unsupported op: {op}")


def _median_or_none(rows: list[dict[str, Any]], key: str) -> float | None:
    values = _finite_metric_values(rows, key)
    if values.size == 0:
        return None
    return float(np.median(values))


def _mean_or_none(rows: list[dict[str, Any]], key: str) -> float | None:
    values = _finite_metric_values(rows, key)
    if values.size == 0:
        return None
    return float(np.mean(values))


def _draw_kpi_card(ax: Any, title: str, value: str, subtitle: str, color: str) -> None:
    ax.set_facecolor("#f8fafc")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.04, 0.78, title, transform=ax.transAxes, fontsize=9, color="#475569", weight="bold")
    ax.text(0.04, 0.37, value, transform=ax.transAxes, fontsize=20, color=color, weight="bold")
    ax.text(0.04, 0.12, subtitle, transform=ax.transAxes, fontsize=8.5, color="#64748b")
    ax.axhline(0.02, color=color, linewidth=4, alpha=0.75)


def _write_quality_onepage(
    report_dir: Path,
    name: str,
    rows: list[dict[str, Any]],
    split_counts: dict[str, int],
    args: argparse.Namespace,
    plt: Any,
) -> None:
    total = len(rows)
    avoidance_count = split_counts.get("clean_avoidance", 0)
    social_count = (
        split_counts.get("clean_social", 0)
        + split_counts.get("clean_between_humans", 0)
        + avoidance_count
    )
    between_count = split_counts.get("clean_between_humans", 0)
    rejected_count = split_counts.get("rejected", 0)
    q70_count = _count_where(rows, "quality_score", ">=", 0.7)

    clean_rows = [
        row
        for row in rows
        if str(row.get("split"))
        in {"clean_all", "clean_social", "clean_avoidance", "clean_between_humans"}
    ]
    social_rows = [
        row
        for row in rows
        if str(row.get("split"))
        in {"clean_social", "clean_avoidance", "clean_between_humans"}
    ]

    fig = plt.figure(figsize=(17, 8.9), facecolor="white")
    gs = fig.add_gridspec(4, 6, height_ratios=[0.34, 0.78, 2.05, 2.05], hspace=0.60, wspace=0.55)

    header_ax = fig.add_subplot(gs[0, :])
    header_ax.axis("off")
    header_ax.text(
        0.0,
        0.76,
        "Dataset Quality One-Page Report",
        transform=header_ax.transAxes,
        fontsize=21,
        weight="bold",
        color="#0f172a",
    )
    header_ax.text(
        0.0,
        0.30,
        name,
        transform=header_ax.transAxes,
        fontsize=11,
        color="#475569",
    )
    verdict = "GOOD for initial training" if social_count >= 300 and between_count >= 30 and q70_count / max(total, 1) >= 0.65 else "NEEDS MORE COLLECTION"
    verdict_color = "#16a34a" if verdict.startswith("GOOD") else "#f97316"
    header_ax.text(
        1.0,
        0.55,
        verdict,
        transform=header_ax.transAxes,
        fontsize=13,
        weight="bold",
        ha="right",
        color=verdict_color,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8fafc", "edgecolor": verdict_color, "linewidth": 1.2},
    )

    kpis = [
        ("Total", str(total), "raw windows", "#0f172a"),
        ("Clean Social", f"{social_count}", f"{_rate(social_count, total):.1f}% kept", "#2563eb"),
        ("Reactive Avoidance", f"{avoidance_count}", f"{_rate(avoidance_count, total):.1f}% verified", "#0f766e"),
        ("Between Humans", f"{between_count}", f"{_rate(between_count, total):.1f}% focused", "#16a34a"),
        ("Rejected", f"{rejected_count}", f"{_rate(rejected_count, total):.1f}% removed", "#dc2626"),
        ("Median Quality", _format_kpi(_median_or_none(rows, "quality_score"), decimals=2), "all samples", "#7c3aed"),
    ]
    for i, (title, value, subtitle, color) in enumerate(kpis):
        _draw_kpi_card(fig.add_subplot(gs[1, i]), title, value, subtitle, color)

    color_map = {
        "clean_avoidance": "#0f766e",
        "clean_between_humans": "#16a34a",
        "clean_social": "#2563eb",
        "clean_all": "#64748b",
        "not_between_humans": "#f97316",
        "rejected": "#dc2626",
    }

    ax = fig.add_subplot(gs[2, 0:2])
    split_order = [
        "clean_avoidance",
        "clean_between_humans",
        "clean_social",
        "clean_all",
        "rejected",
    ]
    labels = [split for split in split_order if split_counts.get(split, 0) > 0]
    counts = [split_counts[split] for split in labels]
    bars = ax.bar(range(len(labels)), counts, color=[color_map.get(label, "#475569") for label in labels])
    ax.set_title("Usable Samples by Split", weight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([label.replace("clean_", "").replace("_", "\n") for label in labels], fontsize=9)
    ax.set_ylabel("samples")
    ax.grid(True, axis="y", alpha=0.22)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + max(counts) * 0.015, str(count), ha="center", fontsize=9, weight="bold")

    ax = fig.add_subplot(gs[2, 2:4])
    values = _finite_metric_values(rows, "quality_score")
    if values.size:
        ax.hist(values, bins=24, color="#2563eb", alpha=0.82)
        ax.axvline(float(np.median(values)), color="#dc2626", linewidth=2.0, label=f"median {np.median(values):.2f}")
        ax.axvline(0.7, color="#16a34a", linestyle="--", linewidth=1.6, label="good >= 0.70")
        ax.legend(fontsize=8)
    ax.set_title("Quality Score Distribution", weight="bold")
    ax.set_xlabel("quality score")
    ax.set_ylabel("samples")
    ax.grid(True, alpha=0.22)

    ax = fig.add_subplot(gs[2, 4:6])
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        try:
            x = float(row.get("future_disp", math.nan))
            y = float(row.get("same_time_min_distance_all", math.nan))
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    if xs:
        ax.scatter(xs, ys, s=18, c="#2563eb", alpha=0.48, linewidths=0)
    ax.set_title("Motion vs Human Proximity", weight="bold")
    ax.set_xlabel("robot future displacement [m]")
    ax.set_ylabel("closest robot-human distance [m]")
    ax.grid(True, alpha=0.22)

    ax = fig.add_subplot(gs[3, 0:2])
    future = _finite_metric_values(clean_rows, "future_disp")
    if future.size:
        ax.hist(future, bins=22, color="#0891b2", alpha=0.82)
        ax.axvline(float(np.median(future)), color="#dc2626", linewidth=2.0)
    ax.set_title("Robot Future Motion (Clean)", weight="bold")
    ax.set_xlabel("future displacement [m]")
    ax.set_ylabel("samples")
    ax.grid(True, alpha=0.22)

    ax = fig.add_subplot(gs[3, 2:4])
    dist = _finite_metric_values(social_rows or rows, "same_time_min_distance_all")
    if dist.size:
        ax.hist(dist, bins=22, color="#16a34a", alpha=0.80)
        ax.axvline(float(np.median(dist)), color="#dc2626", linewidth=2.0)
    ax.set_title("Robot-Human Interaction Distance", weight="bold")
    ax.set_xlabel("closest same-time distance [m]")
    ax.set_ylabel("samples")
    ax.grid(True, alpha=0.22)

    ax = fig.add_subplot(gs[3, 4:6])
    occ = _finite_metric_values(rows, "local_occupancy_ratio")
    if occ.size:
        ax.hist(occ, bins=22, color="#7c3aed", alpha=0.78)
        ax.axvline(float(np.median(occ)), color="#dc2626", linewidth=2.0, label=f"median {np.median(occ):.2f}")
        ax.legend(fontsize=8)
    ax.set_title("Local Map Occupancy", weight="bold")
    ax.set_xlabel("occupancy ratio")
    ax.set_ylabel("samples")
    ax.grid(True, alpha=0.22)

    fig.text(
        0.01,
        0.01,
        "Recommended use: train with clean_social; oversample clean_avoidance for reactive behavior.",
        fontsize=9,
        color="#475569",
    )
    fig.savefig(report_dir / f"{name}_quality_onepage.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def maybe_write_quality_report(out_dir: Path, name: str, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    report_dir = out_dir / "quality_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    split_order = [
        "clean_avoidance",
        "clean_between_humans",
        "clean_social",
        "clean_all",
        "not_between_humans",
        "rejected",
    ]
    split_counts = {split: 0 for split in split_order}
    for row in rows:
        split = str(row.get("split", "unknown"))
        split_counts[split] = split_counts.get(split, 0) + 1

    summary = _numeric_metric_summary(rows)
    summary["split_counts"] = split_counts
    (report_dir / f"{name}_quality_report_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    _write_quality_onepage(report_dir, name, rows, split_counts, args, plt)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    labels = [split for split in split_order if split_counts.get(split, 0) > 0]
    labels += sorted(split for split, count in split_counts.items() if count > 0 and split not in labels)
    counts = [split_counts[split] for split in labels]
    axes[0, 0].bar(labels, counts, color="#475569")
    axes[0, 0].set_title("sample counts by split")
    axes[0, 0].tick_params(axis="x", labelrotation=25)
    axes[0, 0].grid(True, axis="y", alpha=0.22)

    _hist(axes[0, 1], rows, "quality_score", "heuristic quality score", bins=28)

    x = _finite_metric_values(rows, "future_disp")
    y_all: list[float] = []
    x_all: list[float] = []
    for row in rows:
        try:
            xx = float(row.get("future_disp", math.nan))
            yy = float(row.get("same_time_min_distance_all", math.nan))
        except (TypeError, ValueError):
            continue
        if math.isfinite(xx) and math.isfinite(yy):
            x_all.append(xx)
            y_all.append(yy)
    if x_all:
        axes[1, 0].scatter(x_all, y_all, s=16, c="#2563eb", alpha=0.52, linewidths=0)
    else:
        axes[1, 0].text(0.5, 0.5, "no finite values", ha="center", va="center", transform=axes[1, 0].transAxes)
    axes[1, 0].set_title("motion vs closest human distance")
    axes[1, 0].set_xlabel("robot future displacement [m]")
    axes[1, 0].set_ylabel("min same-time robot-human distance [m]")
    axes[1, 0].grid(True, alpha=0.22)

    _hist(axes[1, 1], rows, "local_occupancy_ratio", "local occupancy ratio", bins=28)
    fig.tight_layout()
    fig.savefig(report_dir / f"{name}_quality_dashboard.png", dpi=170)
    plt.close(fig)

    motion_metrics = [
        ("obs_disp", "observed displacement [m]"),
        ("future_disp", "future displacement [m]"),
        ("mean_speed", "mean speed [m/s]"),
        ("path_efficiency", "path efficiency"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, title) in zip(axes.flat, motion_metrics):
        _hist(ax, rows, key, title)
    fig.tight_layout()
    fig.savefig(report_dir / f"{name}_motion_metrics.png", dpi=170)
    plt.close(fig)

    interaction_metrics = [
        ("neighbor_count", "neighbor count"),
        ("min_social_distance_obs", "min observed social distance [m]"),
        ("same_time_min_distance_all", "same-time min distance [m]"),
        ("same_time_min_distance_future", "future same-time min distance [m]"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, title) in zip(axes.flat, interaction_metrics):
        _hist(ax, rows, key, title)
    fig.tight_layout()
    fig.savefig(report_dir / f"{name}_interaction_metrics.png", dpi=170)
    plt.close(fig)

    avoidance_metrics = [
        ("cv_min_distance_m", "constant-velocity closest distance [m]"),
        ("cv_time_to_closest_s", "constant-velocity time to closest [s]"),
        ("robot_slowdown_ratio", "robot slowdown ratio"),
        ("route_future_max_deviation_m", "max deviation from static route [m]"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, title) in zip(axes.flat, avoidance_metrics):
        _hist(ax, rows, key, title)
    fig.tight_layout()
    fig.savefig(report_dir / f"{name}_avoidance_metrics.png", dpi=170)
    plt.close(fig)


def _visualized_neighbor_indices(
    sample: dict[str, Any],
    mode: str,
    index: int,
) -> list[int]:
    neighbor_past = np.asarray(sample["neighbor_past"], dtype=np.float32)
    neighbor_mask = np.asarray(sample["neighbor_mask"], dtype=np.float32)
    valid = [
        idx
        for idx, mask in enumerate(neighbor_mask)
        if mask > 0 and idx < neighbor_past.shape[0] and np.isfinite(neighbor_past[idx, :, 0]).any()
    ]
    if mode == "all":
        return valid
    if not valid:
        return []
    if mode == "index":
        return [index] if index in valid else []

    target_past = np.asarray(sample["target_past"], dtype=np.float32)
    best_idx = valid[0]
    best_dist = math.inf
    for idx in valid:
        nbr = neighbor_past[idx]
        finite = np.isfinite(nbr[:, 0]) & np.isfinite(nbr[:, 1])
        if not finite.any():
            continue
        aligned_target = target_past[: nbr.shape[0]][finite]
        dist = float(np.linalg.norm(nbr[finite] - aligned_target, axis=1).min())
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return [best_idx]


def maybe_write_visualizations(
    out_dir: Path,
    samples: list[dict[str, Any]],
    count: int,
    neighbor_mode: str,
    neighbor_index: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    for i, sample in enumerate(samples[:count]):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax = axes[0]
        ax.plot(sample["target_past"][:, 0], sample["target_past"][:, 1], "bo-", label="target past 8")
        ax.plot(sample["target_future"][:, 0], sample["target_future"][:, 1], "ro-", label="target future 12")
        shown_neighbors = _visualized_neighbor_indices(sample, neighbor_mode, neighbor_index)
        for label_count, j in enumerate(shown_neighbors):
            nbr = sample["neighbor_past"][j]
            ax.plot(
                nbr[:, 0],
                nbr[:, 1],
                "k.--",
                alpha=0.72,
                linewidth=1.8,
                label=f"neighbor {j} past 8" if label_count == 0 else None,
            )
        ax.scatter(
            [sample["guidance_point"][0]],
            [sample["guidance_point"][1]],
            marker="D",
            s=70,
            color="#16a34a",
            label="guidance point",
        )
        ax.scatter(
            [sample["final_goal"][0]],
            [sample["final_goal"][1]],
            marker="*",
            s=110,
            color="#dc2626",
            label="final goal",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        ax.set_title(f"sample {sample['meta']['source_index']}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        axes[1].imshow(sample["local_map"], cmap="gray_r", origin="upper")
        axes[1].set_title("local occupancy patch")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        fig.tight_layout()
        fig.savefig(vis_dir / f"clean_sample_{i:03d}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--map-yaml", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--min-neighbors", type=int, default=1)
    parser.add_argument("--min-obs-disp", type=float, default=0.5)
    parser.add_argument("--min-future-disp", type=float, default=0.5)
    parser.add_argument("--min-path-length", type=float, default=1.0)
    parser.add_argument("--min-path-efficiency", type=float, default=0.15)
    parser.add_argument("--min-social-distance", type=float, default=1.2)
    parser.add_argument("--max-social-distance", type=float, default=3.0)
    parser.add_argument("--max-speed", type=float, default=1.5)
    parser.add_argument("--max-acceleration", type=float, default=2.5)
    parser.add_argument("--max-yaw-rate", type=float, default=1.5)
    parser.add_argument("--max-frame-interval-error", type=float, default=0.15)
    parser.add_argument("--collision-distance", type=float, default=0.65)
    parser.add_argument("--avoidance-conflict-distance", type=float, default=1.2)
    parser.add_argument("--avoidance-min-ttc", type=float, default=0.2)
    parser.add_argument("--avoidance-max-ttc", type=float, default=4.0)
    parser.add_argument("--avoidance-min-closing-speed", type=float, default=0.2)
    parser.add_argument("--avoidance-velocity-window", type=int, default=3)
    parser.add_argument("--avoidance-min-reference-speed", type=float, default=0.25)
    parser.add_argument("--avoidance-response-window", type=int, default=3)
    parser.add_argument("--avoidance-response-extra-time", type=float, default=1.0)
    parser.add_argument("--avoidance-min-slowdown-ratio", type=float, default=0.20)
    parser.add_argument("--avoidance-min-route-deviation", type=float, default=0.35)
    parser.add_argument("--map-size-m", type=float, default=8.0)
    parser.add_argument("--map-grid-size", type=int, default=32)
    parser.add_argument("--guidance-radius", type=float, default=8.0)
    parser.add_argument(
        "--guidance-policy",
        choices=["route", "metadata"],
        default="route",
        help="Use a collision-free map route or preserve legacy metadata/straight GP.",
    )
    parser.add_argument("--route-robot-radius", type=float, default=0.275)
    parser.add_argument("--route-safety-margin", type=float, default=0.10)
    parser.add_argument("--route-start-snap-distance", type=float, default=0.75)
    parser.add_argument("--route-goal-snap-distance", type=float, default=0.0)
    parser.add_argument("--route-cache-size", type=int, default=32)
    parser.add_argument("--route-planning-resolution", type=float, default=0.10)
    parser.add_argument("--between-humans-only", action="store_true")
    parser.add_argument("--between-min-path-length", type=float, default=1.0)
    parser.add_argument("--between-min-side-distance", type=float, default=0.35)
    parser.add_argument("--between-max-side-distance", type=float, default=2.2)
    parser.add_argument("--between-min-gap", type=float, default=1.0)
    parser.add_argument("--between-max-gap", type=float, default=4.0)
    parser.add_argument("--between-max-forward-gap", type=float, default=2.0)
    parser.add_argument("--between-ahead-margin", type=float, default=0.5)
    parser.add_argument("--between-behind-margin", type=float, default=0.5)
    parser.add_argument("--visualize", type=int, default=12)
    parser.add_argument("--visualize-neighbor-mode", choices=["closest", "index", "all"], default="all")
    parser.add_argument("--visualize-neighbor-index", type=int, default=0)
    parser.add_argument("--skip-quality-report", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or args.input.stem

    with args.input.open("rb") as f:
        raw = pickle.load(f)
    map_data = load_map(args.map_yaml)
    route_clearance_m = max(
        0.0,
        float(args.route_robot_radius) + float(args.route_safety_margin),
    )
    footprint_planner = OccupancyGridRoutePlanner(
        map_data,
        clearance_m=route_clearance_m,
        cache_size=args.route_cache_size,
        planning_resolution_m=args.route_planning_resolution,
    )
    route_planner = footprint_planner if args.guidance_policy == "route" else None

    metadata = dict(raw.get("metadata", {}))
    obs_len = int(metadata.get("obs_len", 8))
    pred_len = int(metadata.get("pred_len", 12))
    dt = float(metadata.get("dt", 0.4))
    all_trajs = raw.get("all_trajs", [])
    all_meta = raw.get("sample_meta", [{} for _ in all_trajs])

    quality_rows: list[dict[str, Any]] = []
    all_clean_samples: list[dict[str, Any]] = []
    social_clean_samples: list[dict[str, Any]] = []
    avoidance_clean_samples: list[dict[str, Any]] = []
    between_humans_samples: list[dict[str, Any]] = []
    all_clean_trajs: list[np.ndarray] = []
    social_clean_trajs: list[np.ndarray] = []
    avoidance_clean_trajs: list[np.ndarray] = []
    between_humans_trajs: list[np.ndarray] = []
    all_clean_meta: list[dict[str, Any]] = []
    social_clean_meta: list[dict[str, Any]] = []
    avoidance_clean_meta: list[dict[str, Any]] = []
    between_humans_meta: list[dict[str, Any]] = []

    rejection_counts: dict[str, int] = {
        "nan_target": 0,
        "basic_filter": 0,
        "route_guidance_failed": 0,
        "not_social_window": 0,
        "not_avoidance_window": 0,
        "not_between_humans": 0,
    }
    basic_failure_reasons: dict[str, int] = {}
    route_failure_reasons: dict[str, int] = {}

    for idx, trajs_in in enumerate(all_trajs):
        trajs = np.asarray(trajs_in, dtype=np.float32)
        meta = dict(all_meta[idx]) if idx < len(all_meta) else {}
        target = trajs[0, : obs_len + pred_len]
        if not np.isfinite(target).all():
            rejection_counts["nan_target"] += 1
            continue
        q = sample_quality(trajs, obs_len, pred_len, dt)
        q.update(sample_timing_metrics(meta, obs_len + pred_len, dt))
        q.update(same_time_robot_human_metrics(trajs, obs_len, pred_len, args))
        q.update(avoidance_conflict_metrics(trajs, obs_len, pred_len, dt, args))
        q.update(between_humans_metrics(trajs, obs_len, pred_len, args))
        q.update(local_map_metrics(target[obs_len - 1], map_data, args))
        q.update(trajectory_footprint_metrics(target, footprint_planner))
        q.update(heuristic_quality_scores(q, args))
        q.update({
            "route_future_mean_deviation_m": math.nan,
            "route_future_max_deviation_m": math.nan,
        })
        q.update(avoidance_classification(q, args))
        route_metrics = {
            "route_guidance_valid": 0.0,
            "route_start_snap_m": math.nan,
            "route_goal_snap_m": math.nan,
            "route_path_length_m": math.nan,
            "route_path_pose_count": math.nan,
            "route_planning_resolution_m": math.nan,
            "route_guidance_lookahead_m": math.nan,
        }
        row = {"source_index": idx, **meta, **q, **route_metrics}
        if pass_basic_filter(q, args):
            guidance_override = None
            guidance_metadata = None
            if route_planner is not None:
                final_goal = xy_from_meta(meta, "final_goal")
                if final_goal is None:
                    final_goal = target[obs_len + pred_len - 1].astype(np.float32)
                try:
                    route_points, computed_route_metrics = route_planner.route(
                        target[obs_len - 1],
                        final_goal,
                        start_snap_distance_m=args.route_start_snap_distance,
                        goal_snap_distance_m=args.route_goal_snap_distance,
                    )
                    guidance_override = guidance_point_along_route(
                        route_points,
                        args.guidance_radius,
                    )
                    if not route_planner.is_safe_world(guidance_override):
                        raise ValueError("route guidance point is not footprint-safe")
                except ValueError as exc:
                    rejection_counts["route_guidance_failed"] += 1
                    reason = str(exc)
                    route_failure_reasons[reason] = route_failure_reasons.get(reason, 0) + 1
                    row["split"] = "route_guidance_failed"
                    quality_rows.append(row)
                    continue

                route_metrics.update(computed_route_metrics)
                route_metrics["route_guidance_valid"] = 1.0
                route_metrics["route_guidance_lookahead_m"] = min(
                    max(float(args.guidance_radius), 0.0),
                    float(computed_route_metrics["route_path_length_m"]),
                )
                q.update(route_metrics)
                row.update(route_metrics)
                response_metrics = route_response_metrics(
                    target,
                    obs_len,
                    pred_len,
                    np.asarray(route_points, dtype=np.float32),
                )
                q.update(response_metrics)
                response_classification = avoidance_classification(q, args)
                q.update(response_classification)
                row.update(response_metrics)
                row.update(response_classification)
                guidance_metadata = {
                    "guidance_policy": "inflated_occupancy_grid_route_lookahead",
                    "guidance_radius": float(args.guidance_radius),
                    "route_planner": "reverse_dijkstra_8_connected",
                    "route_clearance_m": float(route_clearance_m),
                    **computed_route_metrics,
                }

            teacher_classification = teacher_avoidance_classification(
                meta,
                q,
                args,
            )
            q.update(teacher_classification)
            row.update(teacher_classification)

            sample = make_model_sample(
                trajs,
                meta,
                q,
                obs_len,
                pred_len,
                map_data,
                args,
                idx,
                guidance_point_override=guidance_override,
                guidance_metadata=guidance_metadata,
            )
            processed_meta = dict(sample["meta"])
            all_clean_samples.append(sample)
            all_clean_trajs.append(trajs)
            all_clean_meta.append(processed_meta)
            row["split"] = "clean_all"
            is_avoidance = pass_avoidance_filter(q, args)
            if pass_social_filter(q, args) or is_avoidance:
                if pass_between_humans_filter(q, args):
                    between_humans_samples.append(sample)
                    between_humans_trajs.append(trajs)
                    between_humans_meta.append(processed_meta)
                elif args.between_humans_only:
                    rejection_counts["not_between_humans"] += 1
                    row["split"] = "not_between_humans"
                    quality_rows.append(row)
                    continue
                social_clean_samples.append(sample)
                social_clean_trajs.append(trajs)
                social_clean_meta.append(processed_meta)
                if is_avoidance:
                    avoidance_clean_samples.append(sample)
                    avoidance_clean_trajs.append(trajs)
                    avoidance_clean_meta.append(processed_meta)
                    row["split"] = "clean_avoidance"
                else:
                    rejection_counts["not_avoidance_window"] += 1
                    row["split"] = (
                        "clean_between_humans"
                        if q["between_humans"] >= 1.0
                        else "clean_social"
                    )
            else:
                rejection_counts["not_social_window"] += 1
        else:
            rejection_counts["basic_filter"] += 1
            failure_reasons = basic_filter_reasons(q, args)
            row["filter_failure_reasons"] = ";".join(failure_reasons)
            for reason in failure_reasons:
                basic_failure_reasons[reason] = (
                    basic_failure_reasons.get(reason, 0) + 1
                )
            row["split"] = "rejected"
        quality_rows.append(row)

    common_meta = {
        **metadata,
        "postprocess_format": "moai_social_nav_route_guidance_clean_v3",
        "input_path": str(args.input),
        "map_yaml_path": str(args.map_yaml),
        "map_image_path": map_data["image_path"],
        "local_map_size_m": args.map_size_m,
        "local_map_grid_size": args.map_grid_size,
        "guidance_radius": args.guidance_radius,
        "guidance_policy": (
            "inflated_occupancy_grid_route_lookahead"
            if route_planner is not None
            else "metadata_or_circle_line_intersection_to_final_goal"
        ),
        "route_planner": (
            "reverse_dijkstra_8_connected" if route_planner is not None else None
        ),
        "route_robot_radius_m": float(args.route_robot_radius),
        "route_safety_margin_m": float(args.route_safety_margin),
        "route_clearance_m": float(route_clearance_m),
        "target_trajectory_map_check": "inflated_grid_swept_path",
        "route_start_snap_distance_m": float(args.route_start_snap_distance),
        "route_goal_snap_distance_m": float(args.route_goal_snap_distance),
        "route_planning_resolution_m": (
            float(route_planner.resolution)
            if route_planner is not None
            else None
        ),
        "input_keys": [
            "target_past",
            "target_future",
            "neighbor_past",
            "neighbor_future",
            "neighbor_mask",
            "final_goal",
            "guidance_point",
            "local_map",
        ],
        "filters": {
            "min_neighbors": args.min_neighbors,
            "min_obs_disp": args.min_obs_disp,
            "min_future_disp": args.min_future_disp,
            "min_path_length": args.min_path_length,
            "min_path_efficiency": args.min_path_efficiency,
            "min_social_distance": args.min_social_distance,
            "max_social_distance": args.max_social_distance,
            "max_speed": args.max_speed,
            "max_acceleration": args.max_acceleration,
            "max_yaw_rate": args.max_yaw_rate,
            "max_frame_interval_error": args.max_frame_interval_error,
            "collision_distance": args.collision_distance,
            "avoidance_conflict_distance": args.avoidance_conflict_distance,
            "avoidance_min_ttc": args.avoidance_min_ttc,
            "avoidance_max_ttc": args.avoidance_max_ttc,
            "avoidance_min_closing_speed": args.avoidance_min_closing_speed,
            "avoidance_velocity_window": args.avoidance_velocity_window,
            "avoidance_min_reference_speed": args.avoidance_min_reference_speed,
            "avoidance_response_window": args.avoidance_response_window,
            "avoidance_response_extra_time": args.avoidance_response_extra_time,
            "avoidance_min_slowdown_ratio": args.avoidance_min_slowdown_ratio,
            "avoidance_min_route_deviation": args.avoidance_min_route_deviation,
            "guidance_policy": args.guidance_policy,
            "between_humans_only": args.between_humans_only,
            "between_min_path_length": args.between_min_path_length,
            "between_min_side_distance": args.between_min_side_distance,
            "between_max_side_distance": args.between_max_side_distance,
            "between_min_gap": args.between_min_gap,
            "between_max_gap": args.between_max_gap,
            "between_max_forward_gap": args.between_max_forward_gap,
            "between_ahead_margin": args.between_ahead_margin,
            "between_behind_margin": args.between_behind_margin,
        },
    }

    outputs = {
        "clean_all": (all_clean_samples, all_clean_trajs, all_clean_meta),
        "clean_social": (social_clean_samples, social_clean_trajs, social_clean_meta),
        "clean_avoidance": (
            avoidance_clean_samples,
            avoidance_clean_trajs,
            avoidance_clean_meta,
        ),
        "clean_between_humans": (between_humans_samples, between_humans_trajs, between_humans_meta),
    }
    for split, (samples, trajs, metas) in outputs.items():
        payload = {
            "samples": samples,
            "all_trajs": trajs,
            "sample_meta": metas,
            "metadata": {**common_meta, "split": split},
            "quality_summary": summarize([s["quality"] for s in samples]),
        }
        with (args.out_dir / f"{name}_{split}.pkl").open("wb") as f:
            pickle.dump(payload, f)

    write_quality_csv(args.out_dir / f"{name}_quality.csv", quality_rows)
    if not args.skip_quality_report:
        maybe_write_quality_report(args.out_dir, name, quality_rows, args)
    maybe_write_visualizations(
        args.out_dir,
        between_humans_samples or social_clean_samples or all_clean_samples,
        args.visualize,
        args.visualize_neighbor_mode,
        args.visualize_neighbor_index,
    )

    summary = {
        "input_samples": len(all_trajs),
        "clean_all_samples": len(all_clean_samples),
        "clean_social_samples": len(social_clean_samples),
        "clean_avoidance_samples": len(avoidance_clean_samples),
        "clean_between_humans_samples": len(between_humans_samples),
        "rejection_counts": rejection_counts,
        "basic_failure_reasons": dict(sorted(basic_failure_reasons.items())),
        "route_failure_reasons": dict(sorted(route_failure_reasons.items())),
        "clean_all_summary": summarize([s["quality"] for s in all_clean_samples]),
        "clean_social_summary": summarize([s["quality"] for s in social_clean_samples]),
        "clean_avoidance_summary": summarize([s["quality"] for s in avoidance_clean_samples]),
        "clean_between_humans_summary": summarize([s["quality"] for s in between_humans_samples]),
        "outputs": {
            "clean_all": str(args.out_dir / f"{name}_clean_all.pkl"),
            "clean_social": str(args.out_dir / f"{name}_clean_social.pkl"),
            "clean_avoidance": str(args.out_dir / f"{name}_clean_avoidance.pkl"),
            "clean_between_humans": str(args.out_dir / f"{name}_clean_between_humans.pkl"),
            "quality_csv": str(args.out_dir / f"{name}_quality.csv"),
            "quality_report": str(args.out_dir / "quality_report"),
            "visualizations": str(args.out_dir / "visualizations"),
        },
    }
    (args.out_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

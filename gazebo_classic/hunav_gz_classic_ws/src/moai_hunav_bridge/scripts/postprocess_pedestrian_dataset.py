#!/usr/bin/env python3
"""Post-process HuNav pedestrian trajectory PKLs into model-ready samples."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
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


def _read_pgm(path: Path) -> np.ndarray:
    tokens: list[str] = []
    with path.open("rb") as f:
        for raw_line in f:
            line = raw_line.decode("ascii", errors="ignore").strip()
            if not line or line.startswith("#"):
                continue
            if "#" in line:
                line = line.split("#", 1)[0]
            tokens.extend(line.split())
    if not tokens or tokens[0] != "P2":
        raise ValueError(f"Only ASCII P2 PGM is supported: {path}")
    width = int(tokens[1])
    height = int(tokens[2])
    max_value = int(tokens[3])
    pixels = np.asarray([int(v) for v in tokens[4:]], dtype=np.float32)
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
    # ROS occupancy maps are usually bright=free, dark=occupied.
    occupied = (gray < 0.5).astype(np.float32)
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

    crop = np.ones((raw_size, raw_size), dtype=np.float32)
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
    speeds = step / max(dt, 1e-6)
    return {
        "obs_disp": obs_disp,
        "future_disp": fut_disp,
        "total_disp": total_disp,
        "path_length": total_path,
        "path_efficiency": total_disp / total_path if total_path > 1e-6 else 0.0,
        "mean_speed": float(speeds.mean()) if speeds.size else 0.0,
        "max_speed": float(speeds.max()) if speeds.size else 0.0,
        "min_social_distance_obs": min_target_neighbor_distance(trajs, obs_len),
        "neighbor_count": float(max(0, trajs.shape[0] - 1)),
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


def pass_basic_filter(q: dict[str, float], args: argparse.Namespace) -> bool:
    return (
        q["neighbor_count"] >= args.min_neighbors
        and q["obs_disp"] >= args.min_obs_disp
        and q["future_disp"] >= args.min_future_disp
        and q["path_length"] >= args.min_path_length
        and q["path_efficiency"] >= args.min_path_efficiency
        and q["min_social_distance_obs"] >= args.min_social_distance
        and q["max_speed"] <= args.max_speed
    )


def pass_social_filter(q: dict[str, float], args: argparse.Namespace) -> bool:
    return pass_basic_filter(q, args) and q["min_social_distance_obs"] <= args.max_social_distance


def pass_between_humans_filter(q: dict[str, float], args: argparse.Namespace) -> bool:
    return pass_social_filter(q, args) and q.get("between_humans", 0.0) >= 1.0


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
) -> dict[str, Any]:
    target_past = trajs[0, :obs_len].astype(np.float32)
    target_future = trajs[0, obs_len : obs_len + pred_len].astype(np.float32)
    neighbor_past = trajs[1:, :obs_len].astype(np.float32)
    neighbor_mask = np.isfinite(neighbor_past[..., 0]).any(axis=1).astype(np.float32)
    local_map = local_map_patch(target_past[-1], map_data, args.map_size_m, args.map_grid_size)
    final_goal = xy_from_meta(meta, "final_goal")
    if final_goal is None:
        final_goal = target_future[-1].astype(np.float32)
    guidance_point = xy_from_meta(meta, "guidance_point")
    if guidance_point is None:
        guidance_point = guidance_point_from_goal(target_past[-1], final_goal, args.guidance_radius)
    guidance_traj = np.linspace(target_past[-1], guidance_point, pred_len).astype(np.float32)
    return {
        "target_past": target_past,
        "neighbor_past": neighbor_past,
        "neighbor_mask": neighbor_mask,
        "guidance_point": guidance_point.astype(np.float32),
        "guidance_traj": guidance_traj,
        "guidance_mask": np.ones((pred_len,), dtype=np.float32),
        "final_goal": final_goal,
        "local_map": local_map.astype(np.float32),
        "target_future": target_future,
        "meta": dict(meta, source_index=source_index),
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
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
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
    social_count = split_counts.get("clean_social", 0) + split_counts.get("clean_between_humans", 0)
    between_count = split_counts.get("clean_between_humans", 0)
    rejected_count = split_counts.get("rejected", 0)
    q70_count = _count_where(rows, "quality_score", ">=", 0.7)

    clean_rows = [
        row
        for row in rows
        if str(row.get("split")) in {"clean_all", "clean_social", "clean_between_humans"}
    ]
    social_rows = [
        row
        for row in rows
        if str(row.get("split")) in {"clean_social", "clean_between_humans"}
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
        ("Between Humans", f"{between_count}", f"{_rate(between_count, total):.1f}% focused", "#16a34a"),
        ("Rejected", f"{rejected_count}", f"{_rate(rejected_count, total):.1f}% removed", "#dc2626"),
        ("Median Quality", _format_kpi(_median_or_none(rows, "quality_score"), decimals=2), "all samples", "#7c3aed"),
        ("Median Future", _format_kpi(_median_or_none(clean_rows, "future_disp"), "m", 2), "clean robot motion", "#0891b2"),
    ]
    for i, (title, value, subtitle, color) in enumerate(kpis):
        _draw_kpi_card(fig.add_subplot(gs[1, i]), title, value, subtitle, color)

    color_map = {
        "clean_between_humans": "#16a34a",
        "clean_social": "#2563eb",
        "clean_all": "#64748b",
        "not_between_humans": "#f97316",
        "rejected": "#dc2626",
    }

    ax = fig.add_subplot(gs[2, 0:2])
    split_order = ["clean_between_humans", "clean_social", "clean_all", "rejected"]
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
        "Recommended use: train with clean_social; analyze high-pressure cases with clean_between_humans.",
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
    split_order = ["clean_between_humans", "clean_social", "clean_all", "not_between_humans", "rejected"]
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
    parser.add_argument("--min-social-distance", type=float, default=0.7)
    parser.add_argument("--max-social-distance", type=float, default=3.0)
    parser.add_argument("--max-speed", type=float, default=3.5)
    parser.add_argument("--collision-distance", type=float, default=0.65)
    parser.add_argument("--map-size-m", type=float, default=15.0)
    parser.add_argument("--map-grid-size", type=int, default=32)
    parser.add_argument("--guidance-radius", type=float, default=8.0)
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

    metadata = dict(raw.get("metadata", {}))
    obs_len = int(metadata.get("obs_len", 8))
    pred_len = int(metadata.get("pred_len", 12))
    dt = float(metadata.get("dt", 0.4))
    all_trajs = raw.get("all_trajs", [])
    all_meta = raw.get("sample_meta", [{} for _ in all_trajs])

    quality_rows: list[dict[str, Any]] = []
    all_clean_samples: list[dict[str, Any]] = []
    social_clean_samples: list[dict[str, Any]] = []
    between_humans_samples: list[dict[str, Any]] = []
    all_clean_trajs: list[np.ndarray] = []
    social_clean_trajs: list[np.ndarray] = []
    between_humans_trajs: list[np.ndarray] = []
    all_clean_meta: list[dict[str, Any]] = []
    social_clean_meta: list[dict[str, Any]] = []
    between_humans_meta: list[dict[str, Any]] = []

    rejection_counts: dict[str, int] = {
        "nan_target": 0,
        "basic_filter": 0,
        "not_social_window": 0,
        "not_between_humans": 0,
    }

    for idx, trajs_in in enumerate(all_trajs):
        trajs = np.asarray(trajs_in, dtype=np.float32)
        meta = dict(all_meta[idx]) if idx < len(all_meta) else {}
        target = trajs[0, : obs_len + pred_len]
        if not np.isfinite(target).all():
            rejection_counts["nan_target"] += 1
            continue
        q = sample_quality(trajs, obs_len, pred_len, dt)
        q.update(same_time_robot_human_metrics(trajs, obs_len, pred_len, args))
        q.update(between_humans_metrics(trajs, obs_len, pred_len, args))
        q.update(local_map_metrics(target[obs_len - 1], map_data, args))
        q.update(heuristic_quality_scores(q, args))
        row = {"source_index": idx, **meta, **q}
        if pass_basic_filter(q, args):
            sample = make_model_sample(trajs, meta, q, obs_len, pred_len, map_data, args, idx)
            all_clean_samples.append(sample)
            all_clean_trajs.append(trajs)
            all_clean_meta.append(dict(meta, source_index=idx))
            row["split"] = "clean_all"
            if pass_social_filter(q, args):
                if pass_between_humans_filter(q, args):
                    between_humans_samples.append(sample)
                    between_humans_trajs.append(trajs)
                    between_humans_meta.append(dict(meta, source_index=idx))
                elif args.between_humans_only:
                    rejection_counts["not_between_humans"] += 1
                    row["split"] = "not_between_humans"
                    quality_rows.append(row)
                    continue
                social_clean_samples.append(sample)
                social_clean_trajs.append(trajs)
                social_clean_meta.append(dict(meta, source_index=idx))
                row["split"] = "clean_between_humans" if q["between_humans"] >= 1.0 else "clean_social"
            else:
                rejection_counts["not_social_window"] += 1
        else:
            rejection_counts["basic_filter"] += 1
            row["split"] = "rejected"
        quality_rows.append(row)

    common_meta = {
        **metadata,
        "postprocess_format": "moai_social_nav_guidance_point_clean_v1",
        "input_path": str(args.input),
        "map_yaml_path": str(args.map_yaml),
        "map_image_path": map_data["image_path"],
        "local_map_size_m": args.map_size_m,
        "local_map_grid_size": args.map_grid_size,
        "guidance_radius": args.guidance_radius,
        "guidance_policy": "circle_line_intersection_to_final_goal",
        "input_keys": ["target_past", "neighbor_past", "neighbor_mask", "guidance_point", "local_map"],
        "filters": {
            "min_neighbors": args.min_neighbors,
            "min_obs_disp": args.min_obs_disp,
            "min_future_disp": args.min_future_disp,
            "min_path_length": args.min_path_length,
            "min_path_efficiency": args.min_path_efficiency,
            "min_social_distance": args.min_social_distance,
            "max_social_distance": args.max_social_distance,
            "max_speed": args.max_speed,
            "collision_distance": args.collision_distance,
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
        "clean_between_humans_samples": len(between_humans_samples),
        "rejection_counts": rejection_counts,
        "clean_all_summary": summarize([s["quality"] for s in all_clean_samples]),
        "clean_social_summary": summarize([s["quality"] for s in social_clean_samples]),
        "clean_between_humans_summary": summarize([s["quality"] for s in between_humans_samples]),
        "outputs": {
            "clean_all": str(args.out_dir / f"{name}_clean_all.pkl"),
            "clean_social": str(args.out_dir / f"{name}_clean_social.pkl"),
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

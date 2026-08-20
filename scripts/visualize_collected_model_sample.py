#!/usr/bin/env python3
"""Visualize one collected sample from world geometry through model tokens."""

from __future__ import annotations

import argparse
import importlib.util
import math
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
import numpy as np


WORLD_CMAP = ListedColormap(["#f7f8f6", "#aeb7b8", "#263238"])
WORLD_NORM = BoundaryNorm([-0.01, 0.25, 0.75, 1.01], WORLD_CMAP.N)
MODEL_CMAP = ListedColormap(["#aeb7b8", "#f7f8f6", "#263238"])
MODEL_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], MODEL_CMAP.N)
ROBOT_PAST = "#1769aa"
ROBOT_FUTURE = "#d1493f"
ROUTE = "#007c91"
ROUTE_PREFIX = "#e3a008"
GUIDANCE = "#f6c445"
FINAL_GOAL = "#a23e8c"
HUMAN_COLORS = ("#6f4ca5", "#16856c", "#d36c1e", "#5b7083")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--map-yaml", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-neighbors", type=int, default=4)
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def world_extent(map_data: dict[str, Any]) -> tuple[float, float, float, float]:
    occupied = np.asarray(map_data["occupied"])
    resolution = float(map_data["resolution"])
    origin_x, origin_y, _ = map_data["origin"]
    return (
        float(origin_x),
        float(origin_x) + occupied.shape[1] * resolution,
        float(origin_y),
        float(origin_y) + occupied.shape[0] * resolution,
    )


def draw_world_map(axis, grid: np.ndarray, extent: tuple[float, ...]) -> None:
    axis.imshow(
        grid,
        origin="upper",
        extent=extent,
        cmap=WORLD_CMAP,
        norm=WORLD_NORM,
        interpolation="nearest",
        zorder=0,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color="#718086", linewidth=0.6, alpha=0.22)
    axis.set_xlabel("world x [m]")
    axis.set_ylabel("world y [m]")


def finite_xy(points: Any) -> np.ndarray:
    values = np.asarray(points, dtype=float).reshape(-1, 2)
    return values[np.isfinite(values).all(axis=1)]


def draw_trajectory_sample(axis, sample: dict[str, Any], show_final_goal: bool = True) -> None:
    target_past = finite_xy(sample["target_past"])
    target_future = finite_xy(sample["target_future"])
    axis.plot(
        target_past[:, 0],
        target_past[:, 1],
        "o-",
        color=ROBOT_PAST,
        linewidth=2.4,
        markersize=4.5,
        label="robot past: 8 observed",
        zorder=8,
    )
    axis.plot(
        target_future[:, 0],
        target_future[:, 1],
        "o--",
        color=ROBOT_FUTURE,
        linewidth=2.2,
        markersize=4.2,
        label="robot future: 12 labels",
        zorder=8,
    )
    for index, (past, future, mask) in enumerate(
        zip(sample["neighbor_past"], sample["neighbor_future"], sample["neighbor_mask"])
    ):
        if float(mask) <= 0.5:
            continue
        color = HUMAN_COLORS[index % len(HUMAN_COLORS)]
        past_xy = finite_xy(past)
        future_xy = finite_xy(future)
        if len(past_xy):
            axis.plot(
                past_xy[:, 0],
                past_xy[:, 1],
                ".-",
                color=color,
                linewidth=1.7,
                markersize=5,
                label=f"human {index + 1} past" if index == 0 else None,
                zorder=6,
            )
        if len(future_xy):
            axis.plot(
                future_xy[:, 0],
                future_xy[:, 1],
                ".:",
                color=color,
                linewidth=1.5,
                markersize=4,
                alpha=0.82,
                label="human future (recorded QA)" if index == 0 else None,
                zorder=5,
            )
    current = target_past[-1]
    axis.scatter(
        *current,
        marker="X",
        s=90,
        color="#11191c",
        edgecolor="white",
        linewidth=0.8,
        label="current robot pose",
        zorder=12,
    )
    guidance = np.asarray(sample["guidance_point"], dtype=float)
    axis.scatter(
        *guidance,
        marker="D",
        s=105,
        color=GUIDANCE,
        edgecolor="#574100",
        linewidth=1.0,
        label="guidance point (8 m on route)",
        zorder=13,
    )
    if show_final_goal:
        final_goal = np.asarray(sample["final_goal"], dtype=float)
        axis.scatter(
            *final_goal,
            marker="*",
            s=180,
            color=FINAL_GOAL,
            edgecolor="#4c173f",
            linewidth=0.8,
            label="final goal",
            zorder=13,
        )


def route_prefix(route: np.ndarray, distance_m: float, guidance: np.ndarray) -> np.ndarray:
    if len(route) < 2:
        return np.vstack([route, guidance])
    cumulative = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
    )
    end = int(np.searchsorted(cumulative, distance_m, side="left"))
    end = min(max(end, 1), len(route) - 1)
    return np.vstack([route[:end], np.asarray(guidance, dtype=float)])


def transform_sample_for_model(sample: dict[str, Any], adapter, args: argparse.Namespace):
    raw_trajs = adapter.build_all_trajs(sample, obs_len=8, pred_len=12)
    transformed, center, theta = adapter.transform_to_target(raw_trajs, obs_len=8)
    filtered = adapter.neighbor_filtering(
        transformed,
        num_nbr=args.num_neighbors,
        obs_len=8,
        pred_len=12,
        view_range=20.0,
        view_angle=math.pi / 3.0,
        social_range=2.0,
    )
    return filtered, center, theta


def draw_model_frame(
    axis,
    sample: dict[str, Any],
    model_item: dict[str, Any],
    transformed: np.ndarray,
) -> None:
    envs = model_item["envs"].detach().cpu().numpy()
    axis.imshow(
        envs,
        origin="lower",
        extent=(-10.0, 10.0, -10.0, 10.0),
        cmap=MODEL_CMAP,
        norm=MODEL_NORM,
        interpolation="nearest",
        zorder=0,
    )
    target = transformed[0]
    axis.plot(
        target[:8, 0],
        target[:8, 1],
        "o-",
        color=ROBOT_PAST,
        linewidth=2.5,
        markersize=4.5,
        label="MGP/TGP target observations",
        zorder=8,
    )
    axis.plot(
        target[8:, 0],
        target[8:, 1],
        "o--",
        color=ROBOT_FUTURE,
        linewidth=2.1,
        markersize=4,
        label="traj_lbl (TGP supervision)",
        zorder=8,
    )
    for index, neighbor in enumerate(transformed[1:]):
        color = HUMAN_COLORS[index % len(HUMAN_COLORS)]
        past = finite_xy(neighbor[:8])
        if len(past):
            axis.plot(
                past[:, 0],
                past[:, 1],
                ".-",
                color=color,
                linewidth=1.7,
                markersize=5,
                label="filtered human observations" if index == 0 else None,
                zorder=7,
            )
    guidance = model_item["guidance_lbl"].detach().cpu().numpy()
    goal_label = model_item["goal_lbl"].detach().cpu().numpy()
    axis.scatter(
        *guidance,
        marker="D",
        s=105,
        color=GUIDANCE,
        edgecolor="#574100",
        linewidth=1.0,
        label="guidance_lbl (MGP condition)",
        zorder=12,
    )
    axis.scatter(
        *goal_label,
        marker="^",
        s=100,
        color=ROBOT_FUTURE,
        edgecolor="#5c1916",
        linewidth=0.8,
        label="goal_lbl (MGP label / TGP condition)",
        zorder=12,
    )
    axis.scatter(0.0, 0.0, marker="X", s=90, color="#11191c", zorder=13)
    axis.arrow(
        0.0,
        0.0,
        1.2,
        0.0,
        color="#11191c",
        width=0.025,
        head_width=0.22,
        length_includes_head=True,
        zorder=11,
    )
    axis.text(1.35, 0.12, "robot heading +x", fontsize=8.5, color="#11191c")
    axis.set_xlim(-10.0, 10.0)
    axis.set_ylim(-10.0, 10.0)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color="#718086", linewidth=0.6, alpha=0.22)
    axis.set_xlabel("target-frame x [m]")
    axis.set_ylabel("target-frame y [m]")
    axis.set_title("C. Actual model frame | aligned scene tensor envs (32 x 32)", loc="left")
    axis.legend(loc="upper left", fontsize=7.5, framealpha=0.94)


def tensor_shape(value: Any) -> str:
    shape = tuple(value.shape)
    return "scalar" if not shape else " x ".join(str(part) for part in shape)


def draw_token_panel(axis, model_item: dict[str, Any], sample: dict[str, Any]) -> None:
    mgp_mask = model_item["mgp_attn_mask"].detach().cpu().numpy()
    tgp_mask = model_item["tgp_attn_mask"].detach().cpu().numpy()
    mask_image = np.vstack([mgp_mask, tgp_mask])
    axis.imshow(
        mask_image,
        extent=(0, len(mgp_mask), 1.9, 3.15),
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(["#d9dee0", "#1769aa"]),
        vmin=0,
        vmax=1,
    )
    axis.set_xlim(0, len(mgp_mask))
    axis.set_ylim(-6.4, 3.55)
    axis.set_yticks([2.2, 2.85], ["TGP attention", "MGP attention"])
    axis.set_xticks(range(0, len(mgp_mask) + 1, 10))
    axis.set_xlabel("token index (blue = attended, gray = padding)")
    axis.set_title("D. Adapter output passed to SPU-BERT", loc="left")
    for side in ("left", "right", "top"):
        axis.spines[side].set_visible(False)

    ordered_keys = [
        "mgp_spatial_ids",
        "mgp_segment_ids",
        "mgp_temporal_ids",
        "mgp_attn_mask",
        "tgp_spatial_ids",
        "tgp_segment_ids",
        "tgp_temporal_ids",
        "tgp_attn_mask",
        "env_spatial_ids",
        "env_segment_ids",
        "env_temporal_ids",
        "env_attn_mask",
        "envs",
        "envs_params",
        "guidance_lbl",
        "goal_lbl",
        "traj_lbl",
        "scales",
    ]
    left_keys = ordered_keys[:9]
    right_keys = ordered_keys[9:]
    left_text = "\n".join(f"{key:<20} {tensor_shape(model_item[key])}" for key in left_keys)
    right_text = "\n".join(f"{key:<20} {tensor_shape(model_item[key])}" for key in right_keys)
    axis.text(
        0.01,
        0.58,
        left_text,
        transform=axis.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=8.3,
        linespacing=1.35,
        color="#172126",
    )
    axis.text(
        0.52,
        0.58,
        right_text,
        transform=axis.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=8.3,
        linespacing=1.35,
        color="#172126",
    )
    neighbor_count = int(np.asarray(sample["neighbor_mask"], dtype=float).sum())
    note = (
        f"Raw sample: target_past 8 x 2 | neighbor_past {neighbor_count} x 8 x 2 | "
        "local_map 32 x 32\n"
        "Training: guidance_lbl conditions MGP; goal_lbl supervises MGP and conditions TGP; "
        "traj_lbl supervises TGP.\n"
        "Stored-only QA: neighbor_future, final_goal, guidance_traj, meta, quality.\n"
        "Map values: local 0=free, 0.5=unknown, 1=occupied; envs 0=unknown, 1=free, 2=occupied."
    )
    axis.text(
        0.01,
        0.02,
        note,
        transform=axis.transAxes,
        va="bottom",
        ha="left",
        fontsize=8.1,
        linespacing=1.4,
        color="#354147",
        bbox={"boxstyle": "square,pad=0.5", "facecolor": "#f5f7f6", "edgecolor": "#c9d0d2"},
    )


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    postprocess_path = (
        repo_root
        / "gazebo_classic/hunav_gz_classic_ws/src/moai_hunav_bridge/scripts/postprocess_pedestrian_dataset.py"
    )
    adapter_path = (
        repo_root
        / "gazebo_classic/hunav_gz_classic_ws/src/moai_social_bert_refactored/"
        "SPU-BERT/spubert/datasets/moai_social_nav_extended_goal.py"
    )
    postprocess = load_module("moai_postprocess_visualization", postprocess_path)
    adapter = load_module("moai_dataset_adapter_visualization", adapter_path)

    with args.input.open("rb") as stream:
        payload = pickle.load(stream)
    samples = list(payload["samples"])
    if not 0 <= args.sample_index < len(samples):
        raise IndexError(f"sample index {args.sample_index} outside 0..{len(samples) - 1}")
    sample = samples[args.sample_index]
    metadata = dict(payload.get("metadata", {}))

    dataset_args = SimpleNamespace(
        obs_len=8,
        pred_len=12,
        num_nbr=args.num_neighbors,
        view_range=20.0,
        view_angle=math.pi / 3.0,
        social_range=2.0,
        guidance_conditioned=True,
        scene=True,
        patch_size=16,
        map_align_to_target=True,
        map_source_unknown_value=0.5,
        local_map_size_m=float(metadata.get("local_map_size_m", 20.0)),
        local_map_grid_size=int(metadata.get("local_map_grid_size", 32)),
        env_resol=float(metadata.get("local_map_size_m", 20.0))
        / int(metadata.get("local_map_grid_size", 32)),
        dataset_path=str(args.input.resolve()),
        dataset_name="",
        dataset_split="",
    )
    dataset = adapter.MoAISocialNavExtendedGoalDataset("train", dataset_args)
    model_item = dataset[args.sample_index]
    transformed, _, theta = transform_sample_for_model(sample, adapter, args)

    map_data = postprocess.load_map(args.map_yaml)
    clearance = float(metadata.get("route_robot_radius_m", 0.275)) + float(
        metadata.get("route_safety_margin_m", 0.10)
    )
    planner = postprocess.OccupancyGridRoutePlanner(
        map_data,
        clearance_m=clearance,
        planning_resolution_m=float(metadata.get("route_planning_resolution_m", 0.10)),
    )
    current = np.asarray(sample["target_past"][-1], dtype=np.float32)
    final_goal = np.asarray(sample["final_goal"], dtype=np.float32)
    route_points, route_metrics = planner.route(
        current,
        final_goal,
        start_snap_distance_m=float(metadata.get("route_start_snap_distance_m", 0.75)),
        goal_snap_distance_m=float(metadata.get("route_goal_snap_distance_m", 0.0)),
    )
    route = np.asarray(route_points, dtype=float)
    guidance = np.asarray(sample["guidance_point"], dtype=float)
    recomputed_guidance = postprocess.guidance_point_along_route(
        route_points,
        float(metadata.get("guidance_radius", 8.0)),
    )
    guidance_error = float(np.linalg.norm(guidance - recomputed_guidance))
    prefix = route_prefix(route, float(metadata.get("guidance_radius", 8.0)), guidance)

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.subplots_adjust(
        left=0.055,
        right=0.98,
        bottom=0.055,
        top=0.89,
        wspace=0.16,
        hspace=0.24,
    )
    full, local, model_frame, tokens = axes.flat
    extent = world_extent(map_data)

    draw_world_map(full, map_data["occupied"], extent)
    full.plot(
        route[:, 0],
        route[:, 1],
        "--",
        color=ROUTE,
        linewidth=2.0,
        label="collision-free route to final goal",
        zorder=4,
    )
    full.plot(
        prefix[:, 0],
        prefix[:, 1],
        color=ROUTE_PREFIX,
        linewidth=3.4,
        label="first 8 m used to create GP",
        zorder=5,
    )
    draw_trajectory_sample(full, sample, show_final_goal=True)
    map_size_m = float(metadata.get("local_map_size_m", 20.0))
    half = map_size_m * 0.5
    full.add_patch(
        Rectangle(
            (current[0] - half, current[1] - half),
            map_size_m,
            map_size_m,
            fill=False,
            edgecolor="#1769aa",
            linewidth=1.4,
            linestyle=":",
            label="20 m local-map crop",
            zorder=3,
        )
    )
    display_geometry = [
        route,
        finite_xy(sample["target_past"]),
        finite_xy(sample["target_future"]),
    ]
    for past, future, mask in zip(
        sample["neighbor_past"], sample["neighbor_future"], sample["neighbor_mask"]
    ):
        if float(mask) > 0.5:
            display_geometry.extend([finite_xy(past), finite_xy(future)])
    display_points = np.vstack([points for points in display_geometry if len(points)])
    full.set_xlim(float(display_points[:, 0].min()) - 1.2, float(display_points[:, 0].max()) + 1.2)
    full.set_ylim(float(display_points[:, 1].min()) - 2.0, float(display_points[:, 1].max()) + 2.0)
    full.set_title("A. World map | route-aware guidance-point generation", loc="left")
    full.legend(loc="upper left", fontsize=7.4, ncol=2, framealpha=0.94)

    local_extent = (
        float(current[0] - half),
        float(current[0] + half),
        float(current[1] - half),
        float(current[1] + half),
    )
    draw_world_map(local, np.asarray(sample["local_map"]), local_extent)
    local.plot(route[:, 0], route[:, 1], "--", color=ROUTE, linewidth=1.8, zorder=4)
    local.plot(prefix[:, 0], prefix[:, 1], color=ROUTE_PREFIX, linewidth=3.2, zorder=5)
    draw_trajectory_sample(local, sample, show_final_goal=False)
    local.set_xlim(local_extent[0], local_extent[1])
    local.set_ylim(local_extent[2], local_extent[3])
    local.set_title("B. Stored model sample | local_map (20 m, 32 x 32)", loc="left")
    local.legend(loc="upper left", fontsize=7.5, framealpha=0.94)

    draw_model_frame(model_frame, sample, model_item, transformed)
    draw_token_panel(tokens, model_item, sample)

    sample_meta = sample.get("meta", {})
    source_index = sample_meta.get("source_index", "?")
    seed = sample_meta.get("collection_seed", metadata.get("collection_seed", "?"))
    map_name = Path(
        str(sample_meta.get("map_yaml_path", metadata.get("map_yaml_path", "unknown_map")))
    ).stem
    scenario_name = Path(str(sample_meta.get("scenario_name", "unknown_scenario"))).stem
    quality = sample.get("quality", {})
    fig.suptitle(
        f"{map_name} | {scenario_name} | seed {seed}, clean_social sample {args.sample_index} "
        f"(raw source {source_index})",
        fontsize=17,
        weight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.945,
        f"GP policy: {metadata.get('guidance_policy', '?')} | lookahead 8.0 m | "
        f"route {route_metrics['route_path_length_m']:.2f} m | GP reconstruction error {guidance_error:.6f} m | "
        f"target map-safe {int(float(quality.get('target_map_safe', 0)))}",
        ha="center",
        va="top",
        fontsize=10.2,
        color="#354147",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"saved {args.output}")
    print(f"sample_count={len(samples)} sample_index={args.sample_index} source_index={source_index}")
    print(f"route_length_m={route_metrics['route_path_length_m']:.4f}")
    print(f"stored_guidance={guidance.tolist()}")
    print(f"recomputed_guidance={np.asarray(recomputed_guidance).tolist()}")
    print(f"guidance_reconstruction_error_m={guidance_error:.8f}")
    print(f"target_heading_rad={theta:.6f}")
    for key, value in model_item.items():
        print(f"{key}: {tuple(value.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Render a text-free route-guidance figure from one collected sample."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


REPO_ROOT = Path(__file__).resolve().parents[1]
POSTPROCESS_PATH = (
    REPO_ROOT
    / "gazebo_classic/hunav_gz_classic_ws/src/moai_hunav_bridge/scripts/"
    "postprocess_pedestrian_dataset.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--map-yaml", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--mode", choices=("route", "legacy"), default="route")
    parser.add_argument("--candidate-report", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-k-paths-only", action="store_true")
    parser.add_argument("--highlight-selected", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = parse_args()
    with args.input.expanduser().resolve().open("rb") as stream:
        samples = list(pickle.load(stream)["samples"])
    if not 0 <= args.sample_index < len(samples):
        raise IndexError(
            f"sample index {args.sample_index} outside 0..{len(samples) - 1}"
        )
    sample = samples[args.sample_index]

    postprocess = load_module("route_guidance_only_postprocess", POSTPROCESS_PATH)
    map_data = postprocess.load_map(args.map_yaml.expanduser().resolve())
    planner = postprocess.OccupancyGridRoutePlanner(
        map_data,
        clearance_m=0.375,
        planning_resolution_m=0.10,
    )

    robot_past = np.asarray(sample["target_past"], dtype=np.float32)
    current = robot_past[-1]
    final_goal = np.asarray(sample["final_goal"], dtype=np.float32)
    if args.mode == "route":
        guidance = np.asarray(sample["guidance_point"], dtype=np.float32)
        route, _ = planner.route(current, final_goal, 0.75, 0.0)
        route = np.asarray(route, dtype=np.float32)
    else:
        delta = final_goal - current
        distance = float(np.linalg.norm(delta))
        guidance = (
            final_goal.copy()
            if distance <= 1e-8
            else current + delta / distance * min(8.0, distance)
        )
        route = np.stack((current, final_goal))

    occupied = np.asarray(map_data["occupied"], dtype=np.uint8)
    resolution = float(map_data["resolution"])
    origin_x, origin_y, _ = map_data["origin"]
    extent = (
        origin_x,
        origin_x + occupied.shape[1] * resolution,
        origin_y,
        origin_y + occupied.shape[0] * resolution,
    )

    figure, ax = plt.subplots(figsize=(14, 8), facecolor="white")
    ax.imshow(
        occupied,
        origin="upper",
        extent=extent,
        cmap=ListedColormap(["#ffffff", "#4a4e52"]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
        zorder=0,
    )
    ax.plot(
        route[:, 0],
        route[:, 1],
        color="#0b8f68" if args.mode == "route" else "#d43f3a",
        linewidth=4.0,
        linestyle="-" if args.mode == "route" else "--",
        solid_capstyle="round",
        zorder=3,
    )
    ax.plot(
        robot_past[:, 0],
        robot_past[:, 1],
        linestyle="--",
        marker="o",
        markersize=4.5,
        color="#202326",
        linewidth=2.2,
        zorder=5,
    )
    if args.candidate_report is not None:
        report = json.loads(
            args.candidate_report.expanduser().resolve().read_text(encoding="utf-8")
        )
        if args.top_k_paths_only:
            top_goals = np.asarray(report["top_k_candidate_goals"], dtype=np.float32)
            top_paths = np.asarray(report["top_k_predicted_futures"], dtype=np.float32)
            count = min(max(0, int(args.top_k)), len(top_goals), len(top_paths))
            path_colors = ("#0d47a1", "#1976d2", "#5e35b1", "#00acc1", "#3949ab")
            selected_rank = int(report.get("selected_rank", 1)) - 1
            for index in range(count):
                path = top_paths[index]
                is_selected = args.highlight_selected and index == selected_rank
                ax.plot(
                    path[:, 0],
                    path[:, 1],
                    marker="o",
                    markersize=5.5 if is_selected else 3.6,
                    color=(
                        "#d62728"
                        if is_selected
                        else (
                            "#8aa4c7"
                            if args.highlight_selected
                            else path_colors[index % len(path_colors)]
                        )
                    ),
                    linewidth=4.2 if is_selected else 1.7,
                    alpha=(
                        1.0
                        if is_selected
                        else (0.58 if args.highlight_selected else 0.92)
                    ),
                    zorder=8 if is_selected else 6,
                )
            ax.scatter(
                top_goals[:count, 0],
                top_goals[:count, 1],
                s=185,
                marker="X",
                color="#2468b4",
                edgecolor="white",
                linewidth=1.2,
                zorder=7,
            )
            if args.highlight_selected and 0 <= selected_rank < count:
                ax.scatter(
                    *top_goals[selected_rank],
                    s=260,
                    marker="X",
                    color="#d62728",
                    edgecolor="white",
                    linewidth=1.5,
                    zorder=9,
                )
        else:
            candidates = np.asarray(report["candidate_goals"], dtype=np.float32)
            safe_mask = np.asarray(report["candidate_safe_mask"], dtype=bool)
            safe_indices = np.flatnonzero(safe_mask)
            ranked_safe = safe_indices[
                np.argsort(np.linalg.norm(candidates[safe_indices] - guidance, axis=1))
            ]
            top_indices = ranked_safe[: max(0, int(args.top_k))]

            if np.any(safe_mask):
                ax.scatter(
                    candidates[safe_mask, 0],
                    candidates[safe_mask, 1],
                    s=78,
                    color="#42b978",
                    edgecolor="#18794e",
                    linewidth=1.1,
                    alpha=0.90,
                    zorder=6,
                )
            if np.any(~safe_mask):
                ax.scatter(
                    candidates[~safe_mask, 0],
                    candidates[~safe_mask, 1],
                    s=78,
                    color="#ee746c",
                    edgecolor="#9f2d28",
                    linewidth=1.1,
                    alpha=0.92,
                    zorder=6,
                )
            if len(top_indices):
                ax.scatter(
                    candidates[top_indices, 0],
                    candidates[top_indices, 1],
                    s=185,
                    marker="X",
                    color="#2468b4",
                    edgecolor="white",
                    linewidth=1.2,
                    zorder=7,
                )
    ax.scatter(
        *current,
        s=150,
        color="#111111",
        edgecolor="white",
        linewidth=1.2,
        zorder=7,
    )
    ax.scatter(
        *guidance,
        s=250,
        marker="D" if args.mode == "route" else "X",
        color="#ffc107" if args.mode == "route" else "#e4572e",
        edgecolor="#24272a" if args.mode == "route" else "white",
        linewidth=1.5,
        zorder=8,
    )
    ax.scatter(
        *final_goal,
        s=330,
        marker="*",
        color="#8e44ad",
        edgecolor="white",
        linewidth=1.4,
        zorder=9,
    )

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.axis("off")
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        dpi=190,
        bbox_inches="tight",
        pad_inches=0,
        facecolor="white",
    )
    plt.close(figure)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

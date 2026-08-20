#!/usr/bin/env python3
"""Draw a compact robot-pedestrian interaction view for one collected sample."""

from __future__ import annotations

import argparse
import importlib.util
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Circle
import numpy as np


ROBOT_PAST = "#1f5a94"
ROBOT_FUTURE = "#d43f3a"
HUMAN_COLORS = ("#7653a6", "#14866d", "#d17a22", "#5d7182")
MAP_CMAP = ListedColormap(["#fafafa", "#c8ced0", "#4c5256"])
MAP_NORM = BoundaryNorm([-0.01, 0.25, 0.75, 1.01], MAP_CMAP.N)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--map-yaml", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--social-distance", type=float, default=1.2)
    return parser.parse_args()


def load_postprocess(repo_root: Path):
    path = (
        repo_root
        / "gazebo_classic/hunav_gz_classic_ws/src/moai_hunav_bridge/scripts/"
        "postprocess_pedestrian_dataset.py"
    )
    spec = importlib.util.spec_from_file_location("moai_postprocess_interaction", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite_xy(points) -> np.ndarray:
    values = np.asarray(points, dtype=float).reshape(-1, 2)
    return values[np.isfinite(values).all(axis=1)]


def draw_direction(axis, points: np.ndarray, color: str) -> None:
    if len(points) < 2:
        return
    axis.annotate(
        "",
        xy=points[-1],
        xytext=points[-2],
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.0},
        zorder=12,
    )


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    postprocess = load_postprocess(repo_root)

    with args.input.open("rb") as stream:
        payload = pickle.load(stream)
    samples = list(payload["samples"])
    if not 0 <= args.sample_index < len(samples):
        raise IndexError(f"sample index {args.sample_index} outside 0..{len(samples) - 1}")
    sample = samples[args.sample_index]

    robot_past = finite_xy(sample["target_past"])
    robot_future = finite_xy(sample["target_future"])
    robot_all = np.vstack([robot_past, robot_future])

    humans = []
    closest = None
    for index, (past, future, mask) in enumerate(
        zip(sample["neighbor_past"], sample["neighbor_future"], sample["neighbor_mask"])
    ):
        if float(mask) <= 0.5:
            continue
        human_past = finite_xy(past)
        human_future = finite_xy(future)
        human_all = np.vstack([human_past, human_future])
        humans.append((index, human_past, human_future, human_all))
        count = min(len(robot_all), len(human_all))
        distances = np.linalg.norm(robot_all[:count] - human_all[:count], axis=1)
        time_index = int(np.argmin(distances))
        candidate = (
            float(distances[time_index]),
            robot_all[time_index],
            human_all[time_index],
            index,
            time_index,
        )
        if closest is None or candidate[0] < closest[0]:
            closest = candidate

    if closest is None:
        raise RuntimeError("sample has no active pedestrian")

    map_data = postprocess.load_map(args.map_yaml)
    occupied = np.asarray(map_data["occupied"])
    resolution = float(map_data["resolution"])
    origin_x, origin_y, _ = map_data["origin"]
    extent = (
        float(origin_x),
        float(origin_x) + occupied.shape[1] * resolution,
        float(origin_y),
        float(origin_y) + occupied.shape[0] * resolution,
    )

    fig, axis = plt.subplots(figsize=(12.8, 7.4))
    axis.imshow(
        occupied,
        origin="upper",
        extent=extent,
        cmap=MAP_CMAP,
        norm=MAP_NORM,
        interpolation="nearest",
        zorder=0,
    )

    axis.plot(
        robot_past[:, 0],
        robot_past[:, 1],
        "o-",
        color=ROBOT_PAST,
        linewidth=3.0,
        markersize=5.0,
        label="Robot past (8 steps)",
        zorder=8,
    )
    axis.plot(
        robot_future[:, 0],
        robot_future[:, 1],
        "o-",
        color=ROBOT_FUTURE,
        linewidth=3.0,
        markersize=5.0,
        label="Robot future (12 steps)",
        zorder=8,
    )
    axis.scatter(
        *robot_past[-1],
        marker="X",
        s=130,
        color="#111719",
        edgecolor="white",
        linewidth=1.0,
        label="Current robot pose",
        zorder=13,
    )
    draw_direction(axis, robot_future, ROBOT_FUTURE)

    for display_index, (_, past, future, _) in enumerate(humans):
        color = HUMAN_COLORS[display_index % len(HUMAN_COLORS)]
        axis.plot(
            past[:, 0],
            past[:, 1],
            "o-",
            color=color,
            linewidth=2.2,
            markersize=4.5,
            label=f"Pedestrian {display_index + 1} past",
            zorder=7,
        )
        axis.plot(
            future[:, 0],
            future[:, 1],
            "o--",
            color=color,
            linewidth=2.2,
            markersize=4.2,
            alpha=0.9,
            label=f"Pedestrian {display_index + 1} future",
            zorder=7,
        )
        draw_direction(axis, future, color)

    min_distance, robot_closest, human_closest, human_index, time_index = closest
    axis.add_patch(
        Circle(
            robot_closest,
            args.social_distance,
            facecolor="#78b86b",
            edgecolor="#32723b",
            linewidth=1.5,
            alpha=0.14,
            zorder=3,
        )
    )
    axis.plot(
        [robot_closest[0], human_closest[0]],
        [robot_closest[1], human_closest[1]],
        linestyle=":",
        color="#20282b",
        linewidth=2.0,
        zorder=10,
    )
    axis.scatter(
        [robot_closest[0], human_closest[0]],
        [robot_closest[1], human_closest[1]],
        s=80,
        facecolor="white",
        edgecolor="#20282b",
        linewidth=1.7,
        zorder=11,
    )
    midpoint = 0.5 * (robot_closest + human_closest)
    axis.text(
        midpoint[0] + 0.12,
        midpoint[1] + 0.12,
        f"Closest at same time: {min_distance:.2f} m",
        fontsize=11,
        color="#20282b",
        bbox={"boxstyle": "square,pad=0.35", "facecolor": "white", "edgecolor": "#69767a"},
        zorder=14,
    )

    all_points = [robot_all] + [entry[3] for entry in humans]
    display = np.vstack(all_points)
    axis.set_xlim(float(display[:, 0].min()) - 1.4, float(display[:, 0].max()) + 1.4)
    axis.set_ylim(float(display[:, 1].min()) - 1.4, float(display[:, 1].max()) + 1.4)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color="#7d878a", linewidth=0.7, alpha=0.25)
    axis.set_xlabel("World x [m]")
    axis.set_ylabel("World y [m]")

    meta = sample.get("meta", {})
    seed = meta.get("collection_seed", "?")
    quality = sample.get("quality", {})
    axis.set_title(
        f"Recorded crossing interaction | seed {seed}, clean-social sample {args.sample_index}\n"
        f"same-time clearance {min_distance:.2f} m | social threshold {args.social_distance:.2f} m | "
        f"interaction score {float(quality.get('interaction_score', 0.0)):.2f}",
        loc="left",
        fontsize=14,
        weight="bold",
    )
    axis.legend(loc="upper right", fontsize=9, framealpha=0.95, ncol=2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)

    print(f"saved={args.output}")
    print(f"sample_count={len(samples)} sample_index={args.sample_index}")
    print(f"closest_same_time_distance_m={min_distance:.6f}")
    print(f"closest_human_index={human_index} time_index={time_index}")
    print(f"interaction_score={float(quality.get('interaction_score', 0.0)):.6f}")
    print(f"crossing_score={float(quality.get('crossing_score', 0.0)):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Visualize collision-consistent maps, route GP, and pedestrian tracks."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

import generate_training_maps as training_maps


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_ROOT = (
    REPO_ROOT
    / "gazebo_classic"
    / "hunav_gz_classic_ws"
    / "src"
    / "hunav_gazebo_wrapper"
)
POSTPROCESS_PATH = (
    REPO_ROOT
    / "gazebo_classic"
    / "hunav_gz_classic_ws"
    / "src"
    / "moai_hunav_bridge"
    / "scripts"
    / "postprocess_pedestrian_dataset.py"
)
DEFAULT_OUTPUT = REPO_ROOT / "figures" / "training_map_design" / "joint_interaction_scenarios.png"


def load_postprocessor():
    spec = importlib.util.spec_from_file_location(
        "postprocess_pedestrian_dataset",
        POSTPROCESS_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def old_straight_guidance(start: np.ndarray, goal: np.ndarray, lookahead: float) -> np.ndarray:
    delta = goal - start
    distance = float(np.linalg.norm(delta))
    return start + delta / distance * min(lookahead, distance)


def short_name(name: str) -> str:
    prefix = "agents_training_"
    suffix = "_joint_crossing"
    if name.startswith(prefix):
        name = name[len(prefix):]
    if name.endswith(suffix):
        name = name[:-len(suffix)]
    return name.replace("_joint_", "\n").replace("_", " ")


def plot_scenario(ax, scenario, postprocess) -> None:
    map_path = WRAPPER_ROOT / "maps" / f"{scenario.map_name}.yaml"
    map_data = postprocess.load_map(map_path)
    occupied = np.asarray(map_data["occupied"], dtype=np.uint8)
    resolution = float(map_data["resolution"])
    origin_x, origin_y, _ = map_data["origin"]
    extent = [
        origin_x,
        origin_x + occupied.shape[1] * resolution,
        origin_y,
        origin_y + occupied.shape[0] * resolution,
    ]
    ax.imshow(
        occupied,
        cmap=ListedColormap(["#f7f8f5", "#3e4449"]),
        vmin=0,
        vmax=1,
        origin="upper",
        extent=extent,
        interpolation="nearest",
        zorder=0,
    )

    start = np.asarray(scenario.robot_start, dtype=np.float32)
    goal = np.asarray(scenario.robot_goal, dtype=np.float32)
    planner = postprocess.OccupancyGridRoutePlanner(
        map_data,
        clearance_m=0.375,
        planning_resolution_m=0.10,
    )
    route, metrics = planner.route(start, goal, 0.75, 0.0)
    guidance = postprocess.guidance_point_along_route(route, 8.0)
    route = np.asarray(route, dtype=np.float32)
    old_guidance = old_straight_guidance(start, goal, 8.0)

    ax.plot(
        [start[0], goal[0]],
        [start[1], goal[1]],
        color="#c44e52",
        linestyle="--",
        linewidth=1.5,
        alpha=0.9,
        zorder=2,
    )
    ax.plot(route[:, 0], route[:, 1], color="#168a83", linewidth=2.4, zorder=3)
    ax.scatter(*start, s=50, color="#2f6fb0", edgecolor="white", linewidth=0.8, zorder=6)
    ax.scatter(*goal, s=55, marker="s", color="#3f8f4f", edgecolor="white", linewidth=0.8, zorder=6)
    ax.scatter(
        *old_guidance,
        s=70,
        marker="x",
        color="#c44e52",
        linewidth=2.2,
        zorder=6,
    )
    ax.scatter(
        *guidance,
        s=115,
        marker="*",
        color="#f0bd28",
        edgecolor="#3e4449",
        linewidth=0.7,
        zorder=7,
    )

    pedestrian_colors = ("#7656a8", "#de7b34", "#1f7898")
    for index, profile in enumerate(scenario.agents):
        color = pedestrian_colors[index % len(pedestrian_colors)]
        ax.annotate(
            "",
            xy=profile.end,
            xytext=profile.start,
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "linewidth": 1.5,
                "alpha": 0.9,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=4,
        )
        ax.scatter(
            *profile.start,
            s=28,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
        )

    straight_distance = float(np.linalg.norm(goal - start))
    detour = float(np.linalg.norm(guidance - old_guidance))
    ax.set_title(short_name(scenario.name), fontsize=10, pad=5)
    ax.text(
        0.02,
        0.02,
        f"route {metrics['route_path_length_m']:.1f} m / straight {straight_distance:.1f} m\nGP shift {detour:.2f} m",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#202428",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.5},
        zorder=8,
    )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.grid(False)
    ax.tick_params(labelsize=7, length=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    postprocess = load_postprocessor()
    scenarios = training_maps.JOINT_SCENARIOS
    figure, axes = plt.subplots(2, 3, figsize=(15, 10.0))
    for ax, scenario in zip(axes.flat, scenarios):
        plot_scenario(ax, scenario, postprocess)

    legend = [
        Line2D([0], [0], color="#c44e52", linestyle="--", label="Old straight direction"),
        Line2D([0], [0], color="#168a83", linewidth=2.4, label="Collision-free global path"),
        Line2D([0], [0], marker="x", color="#c44e52", linestyle="None", label="Old 8 m GP"),
        Line2D([0], [0], marker="*", markerfacecolor="#f0bd28", markeredgecolor="#3e4449", color="none", markersize=11, label="Route-aware 8 m GP"),
        Line2D([0], [0], color="#7656a8", marker=">", label="Pedestrian track"),
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    figure.suptitle(
        "Joint Interaction Training Designs: Static Detour + Pedestrian Encounter",
        fontsize=15,
    )
    figure.tight_layout(rect=(0.0, 0.07, 1.0, 0.95), h_pad=4.5, w_pad=1.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()

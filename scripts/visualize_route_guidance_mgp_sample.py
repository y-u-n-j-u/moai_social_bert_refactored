#!/usr/bin/env python3
"""Plot route-aware GP and actual MGP/TGP inference for one collected sample."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = (
    REPO_ROOT
    / "gazebo_classic"
    / "hunav_gz_classic_ws"
    / "src"
    / "moai_social_bert_refactored"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--map-yaml", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_MODEL_ROOT
        / "configs/spubert/moai_social_nav_ext_scene_guided_gazebo5952_continue_b21.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_MODEL_ROOT / "output/spubert_gazebo_5952_continue_b21/model_best.pth",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--tgp-top-k", type=int, default=5)
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def straight_guidance(current: np.ndarray, final_goal: np.ndarray, lookahead: float) -> np.ndarray:
    delta = final_goal - current
    distance = float(np.linalg.norm(delta))
    return current + delta / distance * min(lookahead, distance)


def heading_from_past(past: np.ndarray) -> float:
    delta = past[-1] - past[-2]
    if float(np.linalg.norm(delta)) <= 1e-6:
        return 0.0
    return float(math.atan2(float(delta[1]), float(delta[0])))


def local_to_world(points: np.ndarray, origin: np.ndarray, theta: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    cosine = math.cos(theta)
    sine = math.sin(theta)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    return points.dot(rotation.T) + origin


def load_inference(
    args: argparse.Namespace,
    samples: list[dict],
    planner,
):
    visualizer_path = args.model_root / "scripts/visualize_gazebo_guided_results.py"
    visualizer = load_module("guided_mgp_sample_visualizer", visualizer_path)
    runtime_args = visualizer.load_runtime_namespace(
        str(args.config),
        command="test",
        cli_dry_run=False,
    )
    runtime_args.dataset_path = str(args.input)
    runtime_args.dataset_split = ""
    runtime_args.cuda = bool(args.cuda)
    runtime_args.train_mode = "fs"
    runtime_args.finetune_checkpoint = ""
    runtime_args.pretrain_checkpoint = ""
    runtime_args.shuffle = False
    runtime_args.test_batch_size = 32

    old_cwd = Path.cwd()
    try:
        os.chdir(args.model_root)
        model, device, test_loader = visualizer.load_model_and_test_loader(
            runtime_args,
            args.checkpoint,
        )
        global_index = 0
        with visualizer.torch.no_grad():
            for raw_batch in test_loader:
                data = visualizer._to_device(raw_batch, device)
                batch_size = int(data["mgp_spatial_ids"].size(0))
                batch_samples = samples[global_index : global_index + batch_size]
                origins = [
                    np.asarray(item["target_past"], dtype=np.float32)[-1]
                    for item in batch_samples
                ]
                headings = [
                    heading_from_past(
                        np.asarray(item["target_past"], dtype=np.float32)
                    )
                    for item in batch_samples
                ]

                def candidate_safety_fn(candidate_goals):
                    local = candidate_goals.detach().cpu().numpy()
                    mask = np.zeros(local.shape[:2], dtype=np.bool_)
                    for batch_index, points in enumerate(local):
                        for candidate_index, point in enumerate(points):
                            if not np.isfinite(point[:2]).all():
                                continue
                            world = local_to_world(
                                point[:2],
                                origins[batch_index],
                                headings[batch_index],
                            )[0]
                            mask[batch_index, candidate_index] = planner.is_safe_world(
                                world
                            )
                    return visualizer.torch.as_tensor(
                        mask,
                        dtype=visualizer.torch.bool,
                        device=candidate_goals.device,
                    )

                outputs = model.inference_guided_candidates(
                    **visualizer.guided_inference_kwargs(
                        data,
                        d_sample=runtime_args.d_sample,
                        reject_unknown=runtime_args.reject_unknown_goals,
                    ),
                    top_k=max(1, int(args.tgp_top_k)),
                    candidate_safety_fn=candidate_safety_fn,
                )
                if not global_index <= args.sample_index < global_index + batch_size:
                    global_index += batch_size
                    continue

                batch_index = args.sample_index - global_index
                origin = origins[batch_index]
                theta = headings[batch_index]
                candidate_goals = visualizer.to_numpy(
                    outputs["candidate_goals"][batch_index]
                )
                candidate_safe = visualizer.to_numpy(
                    outputs["candidate_safe_mask"][batch_index]
                ).astype(bool)
                guided_goals = visualizer.to_numpy(
                    outputs["guided_candidate_goals"][batch_index]
                )
                guided_indices = visualizer.to_numpy(
                    outputs["guided_candidate_indices"][batch_index]
                )
                guided_valid = visualizer.to_numpy(
                    outputs["guided_candidate_goal_valid"][batch_index]
                ).astype(bool)
                guided_paths = visualizer.to_numpy(
                    outputs["guided_pred_trajs"][batch_index]
                )

                attempts = []
                selected_rank = None
                for rank in range(len(guided_paths)):
                    goal_world = local_to_world(guided_goals[rank], origin, theta)[0]
                    path_world = local_to_world(guided_paths[rank], origin, theta)
                    goal_safe = bool(guided_valid[rank] and planner.is_safe_world(goal_world))
                    path_safe = bool(
                        goal_safe
                        and all(planner.is_safe_world(point) for point in path_world)
                    )
                    attempts.append(
                        {
                            "rank": rank + 1,
                            "candidate_index": int(guided_indices[rank]),
                            "goal_safe": goal_safe,
                            "path_safe": path_safe,
                        }
                    )
                    if path_safe and selected_rank is None:
                        selected_rank = rank

                if selected_rank is None:
                    selected_rank = 0

                selected_goal = guided_goals[selected_rank]
                predicted_path = guided_paths[selected_rank]
                chosen = attempts[selected_rank]
                record = {
                    "candidate_goals": candidate_goals,
                    "candidate_safe": candidate_safe,
                    "selected_goal": selected_goal,
                    "selected_index": int(guided_indices[selected_rank]),
                    "selected_rank": selected_rank + 1,
                    "pred_traj": predicted_path,
                    "selected_goal_valid": bool(chosen["goal_safe"]),
                    "trajectory_map_safe": bool(chosen["path_safe"]),
                    "execution_valid": bool(chosen["path_safe"]),
                    "candidate_attempts": attempts,
                    "guided_candidate_indices": guided_indices,
                    "guided_candidate_goals": guided_goals,
                    "guided_pred_trajs": guided_paths,
                }
                metrics = {
                    "candidate_safe_rate": float(candidate_safe.mean()),
                    "attempted_candidate_count": len(attempts),
                    "selected_candidate_rank": selected_rank + 1,
                }
                return record, metrics
        raise IndexError(f"sample index {args.sample_index} was not loaded")
    finally:
        os.chdir(old_cwd)


def plot_map(ax, map_data: dict) -> tuple[float, float, float, float]:
    occupied = np.asarray(map_data["occupied"], dtype=np.uint8)
    resolution = float(map_data["resolution"])
    origin_x, origin_y, _ = map_data["origin"]
    extent = (
        origin_x,
        origin_x + occupied.shape[1] * resolution,
        origin_y,
        origin_y + occupied.shape[0] * resolution,
    )
    ax.imshow(
        occupied,
        origin="upper",
        extent=extent,
        cmap=ListedColormap(["#f7f8f5", "#44494d"]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
        zorder=0,
    )
    return extent


def main() -> int:
    args = parse_args()
    args.input = args.input.expanduser().resolve()
    args.map_yaml = args.map_yaml.expanduser().resolve()
    args.model_root = args.model_root.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    with args.input.open("rb") as stream:
        payload = pickle.load(stream)
    samples = list(payload["samples"])
    if not 0 <= args.sample_index < len(samples):
        raise IndexError(f"sample index {args.sample_index} outside 0..{len(samples) - 1}")
    sample = samples[args.sample_index]

    postprocess = load_module("route_gp_postprocess", POSTPROCESS_PATH)
    map_data = postprocess.load_map(args.map_yaml)
    planner = postprocess.OccupancyGridRoutePlanner(
        map_data,
        clearance_m=0.375,
        planning_resolution_m=0.10,
    )
    record, inference_metrics = load_inference(args, samples, planner)

    robot_past = np.asarray(sample["target_past"], dtype=np.float32)
    teacher_future = np.asarray(sample["target_future"], dtype=np.float32)
    current = robot_past[-1]
    final_goal = np.asarray(sample["final_goal"], dtype=np.float32)
    route, route_metrics = planner.route(current, final_goal, 0.75, 0.0)
    route = np.asarray(route, dtype=np.float32)
    route_gp = np.asarray(sample["guidance_point"], dtype=np.float32)
    recomputed_gp = np.asarray(
        postprocess.guidance_point_along_route(route.tolist(), 8.0),
        dtype=np.float32,
    )
    old_gp = straight_guidance(current, final_goal, 8.0)

    theta = heading_from_past(robot_past)
    candidates = local_to_world(record["candidate_goals"], current, theta)
    selected_goal = local_to_world(record["selected_goal"], current, theta)[0]
    predicted_future = local_to_world(record["pred_traj"], current, theta)
    top_k_goals = local_to_world(record["guided_candidate_goals"], current, theta)
    top_k_futures = np.stack(
        [
            local_to_world(path, current, theta)
            for path in record["guided_pred_trajs"]
        ]
    )
    candidate_safe = np.asarray(record["candidate_safe"], dtype=bool)
    safe_candidates = candidates[candidate_safe]
    unsafe_candidates = candidates[~candidate_safe]

    gp_error = float(np.linalg.norm(route_gp - recomputed_gp))
    gp_shift = float(np.linalg.norm(route_gp - old_gp))
    old_gp_safe = bool(planner.is_safe_world(old_gp))
    route_gp_safe = bool(planner.is_safe_world(route_gp))

    plt.rcParams.update(
        {
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 10,
            "font.size": 10,
        }
    )
    figure = plt.figure(figsize=(16, 9.2), facecolor="white")
    grid = figure.add_gridspec(1, 2, width_ratios=(4.5, 1.35), wspace=0.03)
    ax = figure.add_subplot(grid[0, 0])
    info = figure.add_subplot(grid[0, 1])
    extent = plot_map(ax, map_data)

    ax.plot(
        [current[0], final_goal[0]],
        [current[1], final_goal[1]],
        linestyle="--",
        linewidth=1.3,
        color="#c44e52",
        alpha=0.65,
        zorder=2,
    )
    ax.plot(route[:, 0], route[:, 1], color="#14867c", linewidth=2.4, zorder=3)
    ax.plot(
        robot_past[:, 0],
        robot_past[:, 1],
        linestyle="--",
        marker="o",
        markersize=3.5,
        color="#343a40",
        linewidth=1.5,
        zorder=5,
    )
    ax.plot(
        teacher_future[:, 0],
        teacher_future[:, 1],
        marker="o",
        markersize=4,
        color="#d33f3f",
        linewidth=2.5,
        zorder=6,
    )
    ax.plot(
        predicted_future[:, 0],
        predicted_future[:, 1],
        marker="o",
        markersize=3.5,
        color="#2867b2",
        linewidth=2.4,
        zorder=7,
    )

    if len(safe_candidates):
        ax.scatter(
            safe_candidates[:, 0],
            safe_candidates[:, 1],
            s=58,
            color="#42b978",
            edgecolor="#18794e",
            linewidth=1.0,
            alpha=0.88,
            zorder=8,
        )
    if len(unsafe_candidates):
        ax.scatter(
            unsafe_candidates[:, 0],
            unsafe_candidates[:, 1],
            s=58,
            color="#ee8b83",
            edgecolor="#a5312d",
            linewidth=1.0,
            alpha=0.88,
            zorder=8,
        )

    ax.scatter(*current, s=115, color="black", edgecolor="white", linewidth=1.0, zorder=12)
    ax.scatter(
        *old_gp,
        s=150,
        marker="X",
        color="#e4572e",
        edgecolor="white",
        linewidth=1.0,
        zorder=11,
    )
    ax.scatter(
        *route_gp,
        s=175,
        marker="D",
        color="#ffc107",
        edgecolor="#30343b",
        linewidth=1.1,
        zorder=12,
    )
    ax.scatter(
        *selected_goal,
        s=180,
        marker="X",
        color="#1261a0",
        edgecolor="white",
        linewidth=1.2,
        zorder=13,
    )
    ax.add_patch(
        Circle(
            selected_goal,
            radius=0.375,
            facecolor="#2867b2",
            edgecolor="#1261a0",
            linewidth=1.2,
            alpha=0.14,
            zorder=9,
        )
    )
    ax.scatter(
        *final_goal,
        s=240,
        marker="*",
        color="#8e44ad",
        edgecolor="white",
        linewidth=1.0,
        zorder=12,
    )
    ax.annotate(
        "Old straight GP\ncollision",
        xy=old_gp,
        xytext=(old_gp[0] + 0.8, old_gp[1] - 2.0),
        arrowprops={"arrowstyle": "->", "color": "#a5312d", "linewidth": 1.3},
        color="#8f2d28",
        fontsize=9,
        weight="bold",
        zorder=15,
    )
    ax.annotate(
        "Route-aware GP\nsafe",
        xy=route_gp,
        xytext=(route_gp[0] - 4.1, route_gp[1] + 2.0),
        arrowprops={"arrowstyle": "->", "color": "#9c7400", "linewidth": 1.3},
        color="#6f5600",
        fontsize=9,
        weight="bold",
        zorder=15,
    )

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.set_title("Collected sample: route-aware GP + footprint-filtered MGP/TGP", loc="left")
    ax.grid(color="#c9ced3", linewidth=0.6, alpha=0.35)

    info.axis("off")
    meta = sample.get("meta", {})
    source_index = meta.get("source_index", "?")
    safe_count = int(candidate_safe.sum())
    info.text(0.0, 0.97, "GUIDANCE CHANGE", fontsize=12, weight="bold", color="#263238")
    info.text(
        0.0,
        0.91,
        f"Old GP safe       {old_gp_safe}\n"
        f"Route GP safe     {route_gp_safe}\n"
        f"8 m GP shift      {gp_shift:.2f} m\n"
        f"GP rebuild error  {gp_error:.6f} m\n"
        f"Route length      {route_metrics['route_path_length_m']:.2f} m",
        va="top",
        linespacing=1.55,
        family="monospace",
    )
    info.text(0.0, 0.66, "MODEL INPUT", fontsize=12, weight="bold", color="#263238")
    info.text(
        0.0,
        0.60,
        "Robot history     8 poses\n"
        "Local map        20 m / 32 x 32\n"
        "Final goal       purple star\n"
        "Guidance         yellow route GP\n"
        "Pedestrians      hidden in plot",
        va="top",
        linespacing=1.55,
        family="monospace",
    )
    info.text(0.0, 0.39, "ACTUAL INFERENCE", fontsize=12, weight="bold", color="#263238")
    info.text(
        0.0,
        0.33,
        f"MGP candidates    {len(candidates)}\n"
        f"Footprint-safe    {safe_count}\n"
        f"Selected rank     {int(record['selected_rank'])}\n"
        f"Selected index    {int(record['selected_index']) + 1}\n"
        f"Goal footprint    {record['selected_goal_valid']}\n"
        f"TGP footprint     {record['trajectory_map_safe']}",
        va="top",
        linespacing=1.45,
        family="monospace",
    )

    handles = [
        Line2D([0], [0], color="#14867c", linewidth=2.4, label="Footprint-safe global route"),
        Line2D([0], [0], marker="o", color="#d33f3f", label="Teacher robot future"),
        Line2D([0], [0], marker="o", color="#2867b2", label="Predicted TGP future"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#42b978", markeredgecolor="#18794e", label="Footprint-safe MGP"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#ee8b83", markeredgecolor="#a5312d", label="Rejected MGP goal"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#1261a0", markeredgecolor="white", markersize=9, label="Selected safe MGP"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#ffc107", markeredgecolor="#30343b", label="Route-aware GP"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#8e44ad", markeredgecolor="white", markersize=12, label="Final goal"),
    ]
    info.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.03, -0.01), frameon=False, fontsize=9)

    scenario = Path(str(meta.get("scenario_name", "unknown"))).stem
    figure.suptitle(
        f"{scenario} | sample {args.sample_index} (raw source {source_index}) | "
        f"checkpoint: {args.checkpoint.parent.name}",
        fontsize=13,
        y=0.985,
    )
    figure.subplots_adjust(top=0.93, bottom=0.07, left=0.06, right=0.98)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    report = {
        "input": str(args.input),
        "sample_index": args.sample_index,
        "source_index": source_index,
        "map_yaml": str(args.map_yaml),
        "checkpoint": str(args.checkpoint),
        "current_robot": current.tolist(),
        "final_goal": final_goal.tolist(),
        "old_straight_guidance": old_gp.tolist(),
        "old_straight_guidance_safe": old_gp_safe,
        "route_guidance": route_gp.tolist(),
        "route_guidance_safe": route_gp_safe,
        "route_guidance_rebuild_error_m": gp_error,
        "guidance_shift_m": gp_shift,
        "route_path_length_m": float(route_metrics["route_path_length_m"]),
        "route_planner": "inflated_occupancy_grid_reverse_dijkstra_8_connected",
        "footprint_radius_m": 0.375,
        "candidate_count": len(candidates),
        "candidate_safe_count": safe_count,
        "selected_index": int(record["selected_index"]),
        "selected_rank": int(record["selected_rank"]),
        "selected_goal": selected_goal.tolist(),
        "selected_goal_valid": bool(record["selected_goal_valid"]),
        "trajectory_map_safe": bool(record["trajectory_map_safe"]),
        "execution_valid": bool(record["execution_valid"]),
        "candidate_goals": candidates.tolist(),
        "candidate_safe_mask": candidate_safe.tolist(),
        "candidate_attempts": record["candidate_attempts"],
        "top_k_candidate_indices": [
            int(index) for index in record["guided_candidate_indices"]
        ],
        "top_k_candidate_goals": top_k_goals.tolist(),
        "top_k_predicted_futures": top_k_futures.tolist(),
        "predicted_future": predicted_future.tolist(),
        "teacher_future": teacher_future.tolist(),
        "dataset_inference_metrics": inference_metrics,
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output)
    print(report_path)
    print(json.dumps({key: report[key] for key in (
        "old_straight_guidance_safe",
        "route_guidance_safe",
        "guidance_shift_m",
        "candidate_count",
        "candidate_safe_count",
        "selected_goal_valid",
        "trajectory_map_safe",
        "execution_valid",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

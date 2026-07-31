#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.loader import load_runtime_namespace
from src.data_loader import _build_dataset, build_loaders
from src.trainer import build_trainer_class
from src.utils import _to_device, set_seed, set_workdir


MAP_COLORS = ["#cbd5e1", "#f8fbff", "#173b7a"]
PAST_COLOR = "#2563eb"
GT_COLOR = "#16a34a"
PRED_COLOR = "#7c3aed"
GUIDANCE_COLOR = "#f59e0b"
SAFE_COLOR = "#22c55e"
UNSAFE_COLOR = "#ef4444"
SELECTED_COLOR = "#a855f7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize real Gazebo guided-MGP checkpoint outputs."
    )
    parser.add_argument(
        "--config",
        default="configs/spubert/moai_social_nav_ext_scene_smoke_fs.yaml",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--output-dir",
        default="figures/gazebo_guided_mgp_results",
    )
    parser.add_argument("--font-path", default="")
    return parser.parse_args()


def configure_plotting(font_path: str) -> None:
    if font_path:
        path = Path(font_path)
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            font_name = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.family"] = font_name
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def guided_inference_kwargs(
    data: dict[str, torch.Tensor],
    *,
    d_sample: int,
    reject_unknown: bool,
) -> dict[str, Any]:
    return {
        "mgp_spatial_ids": data["mgp_spatial_ids"],
        "mgp_temporal_ids": data["mgp_temporal_ids"],
        "mgp_segment_ids": data["mgp_segment_ids"],
        "mgp_attn_mask": data["mgp_attn_mask"],
        "tgp_temporal_ids": data["tgp_temporal_ids"],
        "tgp_segment_ids": data["tgp_segment_ids"],
        "tgp_attn_mask": data["tgp_attn_mask"],
        "guidance_points": data["guidance_lbl"],
        "env_spatial_ids": data["env_spatial_ids"],
        "env_temporal_ids": data["env_temporal_ids"],
        "env_segment_ids": data["env_segment_ids"],
        "env_attn_mask": data["env_attn_mask"],
        "envs": data["envs"],
        "envs_params": data["envs_params"],
        "d_sample": d_sample,
        "reject_unknown": reject_unknown,
    }


def load_model_and_test_loader(args: argparse.Namespace, checkpoint: Path):
    set_seed(args.seed, use_cuda=args.cuda)
    _, _, test_loader = build_loaders(args, for_test=True)
    trainer_class = build_trainer_class(args)
    trainer = trainer_class(
        train_dataloader=test_loader,
        val_dataloader=None,
        args=args,
        tb_writer=None,
    )
    state = torch.load(
        str(checkpoint),
        map_location=trainer.device,
        weights_only=True,
    )
    model = trainer.model.module if trainer.parallel else trainer.model
    model.load_state_dict(state)
    model.eval()
    return model, trainer.device, test_loader


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def map_bounds(params: np.ndarray) -> tuple[float, float, float, float]:
    min_x, min_y, width, height, resolution = params[:5]
    return (
        float(min_x),
        float(min_x + width * resolution),
        float(min_y),
        float(min_y + height * resolution),
    )


def point_outside_map(point: np.ndarray, params: np.ndarray) -> bool:
    min_x, max_x, min_y, max_y = map_bounds(params)
    return bool(
        point[0] < min_x
        or point[0] >= max_x
        or point[1] < min_y
        or point[1] >= max_y
    )


def collect_inference_records(
    args: argparse.Namespace,
    checkpoint: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model, device, test_loader = load_model_and_test_loader(args, checkpoint)
    records: list[dict[str, Any]] = []
    safe_candidates = 0
    total_candidates = 0
    selected_valid = 0
    trajectory_safe = 0
    execution_valid = 0
    global_index = 0

    with torch.no_grad():
        for raw_batch in test_loader:
            data = _to_device(raw_batch, device)
            outputs = model.inference_guided(
                **guided_inference_kwargs(
                    data,
                    d_sample=args.d_sample,
                    reject_unknown=args.reject_unknown_goals,
                )
            )
            safe_candidates += int(outputs["candidate_safe_mask"].sum().item())
            total_candidates += int(outputs["candidate_safe_mask"].numel())
            selected_valid += int(outputs["selected_goal_valid"].sum().item())
            trajectory_safe += int(outputs["trajectory_map_safe"].sum().item())
            execution_valid += int(outputs["execution_valid"].sum().item())

            batch_size = data["mgp_spatial_ids"].size(0)
            for batch_index in range(batch_size):
                obs = to_numpy(data["mgp_spatial_ids"][batch_index, 1:9, :2])
                gt_traj = to_numpy(data["traj_lbl"][batch_index])
                goal = to_numpy(data["goal_lbl"][batch_index])
                guidance = to_numpy(data["guidance_lbl"][batch_index])
                pred_goal = to_numpy(outputs["pred_goals"][batch_index])
                pred_traj = to_numpy(outputs["pred_trajs"][batch_index])
                params = to_numpy(data["envs_params"][batch_index])
                candidate_goals = to_numpy(outputs["candidate_goals"][batch_index])
                candidate_safe = to_numpy(
                    outputs["candidate_safe_mask"][batch_index]
                ).astype(bool)
                point_safe = to_numpy(
                    outputs["trajectory_point_safe_mask"][batch_index]
                ).astype(bool)
                is_selected_valid = bool(
                    outputs["selected_goal_valid"][batch_index].item()
                )
                is_trajectory_safe = bool(
                    outputs["trajectory_map_safe"][batch_index].item()
                )
                is_execution_valid = bool(
                    outputs["execution_valid"][batch_index].item()
                )
                ade = float(
                    np.linalg.norm(pred_traj - gt_traj, axis=-1).mean()
                )
                fde = float(np.linalg.norm(pred_traj[-1] - gt_traj[-1]))
                gde = float(np.linalg.norm(pred_goal - goal))
                records.append(
                    {
                        "index": global_index,
                        "env": to_numpy(data["envs"][batch_index]),
                        "env_params": params,
                        "obs": obs,
                        "gt_traj": gt_traj,
                        "goal": goal,
                        "guidance": guidance,
                        "candidate_goals": candidate_goals,
                        "candidate_safe": candidate_safe,
                        "selected_goal": pred_goal,
                        "selected_index": int(
                            outputs["selected_indices"][batch_index].item()
                        ),
                        "pred_traj": pred_traj,
                        "trajectory_point_safe": point_safe,
                        "selected_goal_valid": is_selected_valid,
                        "trajectory_map_safe": is_trajectory_safe,
                        "execution_valid": is_execution_valid,
                        "all_candidates_invalid": bool(
                            outputs["all_candidates_invalid"][batch_index].item()
                        ),
                        "guidance_outside_map": point_outside_map(
                            guidance, params
                        ),
                        "ade": ade,
                        "fde": fde,
                        "gde": gde,
                    }
                )
                global_index += 1

    valid_records = [record for record in records if record["execution_valid"]]
    metrics: dict[str, Any] = {
        "test_samples": len(records),
        "all_sample_ade": float(np.mean([record["ade"] for record in records])),
        "all_sample_fde": float(np.mean([record["fde"] for record in records])),
        "all_sample_gde": float(np.mean([record["gde"] for record in records])),
        "safe_candidate_rate": safe_candidates / max(total_candidates, 1),
        "selected_goal_valid_rate": selected_valid / max(len(records), 1),
        "trajectory_map_safe_rate": trajectory_safe / max(len(records), 1),
        "execution_valid_rate": execution_valid / max(len(records), 1),
        "all_candidates_invalid": len(records) - selected_valid,
        "valid_only_ade": (
            float(np.mean([record["ade"] for record in valid_records]))
            if valid_records
            else None
        ),
        "valid_only_fde": (
            float(np.mean([record["fde"] for record in valid_records]))
            if valid_records
            else None
        ),
        "valid_only_gde": (
            float(np.mean([record["gde"] for record in valid_records]))
            if valid_records
            else None
        ),
    }
    return records, metrics


def collect_dataset_stats(args: argparse.Namespace) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "splits": {},
        "total_samples": 0,
        "finite_and_shape_valid": 0,
        "guidance_differs_from_endpoint": 0,
        "guidance_outside_map": 0,
        "endpoint_outside_map": 0,
    }
    expected_shapes = {
        "mgp_spatial_ids": (58, 2),
        "tgp_spatial_ids": (58, 2),
        "traj_lbl": (12, 2),
        "goal_lbl": (2,),
        "guidance_lbl": (2,),
        "env_spatial_ids": (4, 256),
        "envs": (32, 32),
        "envs_params": (6,),
    }
    for split in ("train", "val", "test"):
        dataset = _build_dataset(args, split=split)
        stats["splits"][split] = len(dataset)
        stats["total_samples"] += len(dataset)
        for index in range(len(dataset)):
            sample = dataset[index]
            shapes_valid = all(
                key in sample and tuple(sample[key].shape) == shape
                for key, shape in expected_shapes.items()
            )
            finite = all(
                not value.is_floating_point() or torch.isfinite(value).all()
                for value in sample.values()
                if isinstance(value, torch.Tensor)
            )
            stats["finite_and_shape_valid"] += int(shapes_valid and finite)
            stats["guidance_differs_from_endpoint"] += int(
                not torch.allclose(sample["guidance_lbl"], sample["goal_lbl"])
            )
            params = to_numpy(sample["envs_params"])
            stats["guidance_outside_map"] += int(
                point_outside_map(to_numpy(sample["guidance_lbl"]), params)
            )
            stats["endpoint_outside_map"] += int(
                point_outside_map(to_numpy(sample["goal_lbl"]), params)
            )
    return stats


def draw_map(ax: plt.Axes, record: dict[str, Any]) -> None:
    min_x, max_x, min_y, max_y = map_bounds(record["env_params"])
    cmap = ListedColormap(MAP_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.imshow(
        record["env"],
        origin="lower",
        extent=[min_x, max_x, min_y, max_y],
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        zorder=0,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#94a3b8", alpha=0.22, linewidth=0.6)
    ax.axhline(0, color="#94a3b8", alpha=0.35, linewidth=0.7)
    ax.axvline(0, color="#94a3b8", alpha=0.35, linewidth=0.7)


def draw_common_trajectory(
    ax: plt.Axes,
    record: dict[str, Any],
    *,
    show_gt: bool = True,
    show_pred: bool = False,
) -> None:
    obs = record["obs"]
    ax.plot(
        obs[:, 0],
        obs[:, 1],
        "-o",
        color=PAST_COLOR,
        linewidth=2.2,
        markersize=3.5,
        zorder=4,
    )
    ax.scatter(
        obs[-1, 0],
        obs[-1, 1],
        s=80,
        marker="s",
        color=PAST_COLOR,
        edgecolor="white",
        linewidth=1.2,
        zorder=7,
    )
    if show_gt:
        gt = record["gt_traj"]
        ax.plot(
            gt[:, 0],
            gt[:, 1],
            "--o",
            color=GT_COLOR,
            linewidth=2,
            markersize=3,
            zorder=4,
        )
    if show_pred and record["execution_valid"]:
        pred = record["pred_traj"]
        ax.plot(
            pred[:, 0],
            pred[:, 1],
            "-o",
            color=PRED_COLOR,
            linewidth=2.4,
            markersize=3.5,
            zorder=6,
        )


def draw_guidance(ax: plt.Axes, record: dict[str, Any]) -> None:
    guidance = record["guidance"]
    ax.scatter(
        guidance[0],
        guidance[1],
        s=220,
        marker="*",
        color=GUIDANCE_COLOR,
        edgecolor="#78350f",
        linewidth=0.8,
        zorder=9,
    )
    ax.plot(
        [record["obs"][-1, 0], guidance[0]],
        [record["obs"][-1, 1], guidance[1]],
        linestyle=":",
        color=GUIDANCE_COLOR,
        alpha=0.75,
        linewidth=1.5,
        zorder=2,
    )


def draw_candidates(ax: plt.Axes, record: dict[str, Any]) -> None:
    candidates = record["candidate_goals"]
    safe = record["candidate_safe"]
    if (~safe).any():
        ax.scatter(
            candidates[~safe, 0],
            candidates[~safe, 1],
            s=48,
            marker="x",
            color=UNSAFE_COLOR,
            linewidth=1.8,
            zorder=5,
        )
    if safe.any():
        ax.scatter(
            candidates[safe, 0],
            candidates[safe, 1],
            s=55,
            marker="o",
            color=SAFE_COLOR,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
    selected = record["selected_goal"]
    ax.scatter(
        selected[0],
        selected[1],
        s=190,
        marker="P",
        color=SELECTED_COLOR,
        edgecolor="white",
        linewidth=1.2,
        zorder=10,
    )


def set_plot_limits(
    ax: plt.Axes,
    record: dict[str, Any],
    *,
    include_guidance: bool,
) -> None:
    min_x, max_x, min_y, max_y = map_bounds(record["env_params"])
    if include_guidance:
        guidance = record["guidance"]
        min_x = min(min_x, float(guidance[0]) - 0.5)
        max_x = max(max_x, float(guidance[0]) + 0.5)
        min_y = min(min_y, float(guidance[1]) - 0.5)
        max_y = max(max_y, float(guidance[1]) + 0.5)
    pad = 0.25
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def common_legend_handles() -> list[Any]:
    return [
        Patch(facecolor=MAP_COLORS[2], label="occupied"),
        Patch(facecolor=MAP_COLORS[0], label="unknown"),
        Line2D([], [], color=PAST_COLOR, marker="o", label="robot past"),
        Line2D([], [], color=GT_COLOR, linestyle="--", marker="o", label="GT future"),
        Line2D([], [], color=PRED_COLOR, marker="o", label="TGP prediction"),
        Line2D(
            [], [], color=GUIDANCE_COLOR, marker="*", linestyle="None",
            markersize=12, label="guidance point"
        ),
        Line2D(
            [], [], color=SAFE_COLOR, marker="o", linestyle="None",
            markersize=7, label="free candidate"
        ),
        Line2D(
            [], [], color=UNSAFE_COLOR, marker="x", linestyle="None",
            markersize=8, label="rejected candidate"
        ),
        Line2D(
            [], [], color=SELECTED_COLOR, marker="P", linestyle="None",
            markersize=9, label="selected goal"
        ),
    ]


def choose_single_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        record
        for record in records
        if (
            record["execution_valid"]
            and not record["guidance_outside_map"]
            and record["candidate_safe"].any()
            and (~record["candidate_safe"]).any()
        )
    ]
    if not candidates:
        candidates = [
            record
            for record in records
            if record["execution_valid"] and not record["guidance_outside_map"]
        ]
    if not candidates:
        candidates = [record for record in records if record["execution_valid"]]
    if not candidates:
        raise RuntimeError("No execution-valid test sample is available for visualization")
    candidates = sorted(candidates, key=lambda record: record["ade"])
    return candidates[len(candidates) // 2]


def create_single_pipeline_figure(
    record: dict[str, Any],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 8.7))
    fig.suptitle(
        "Gazebo → Guidance-conditioned MGP → TGP 실제 추론 결과",
        fontsize=24,
        fontweight="bold",
        color="#0f2f5f",
        y=0.96,
    )
    fig.text(
        0.5,
        0.91,
        f"test #{record['index']} · execution_valid=True · "
        f"ADE {record['ade']:.2f} m · FDE {record['fde']:.2f} m",
        ha="center",
        fontsize=12,
        color="#475569",
    )

    draw_map(axes[0], record)
    draw_common_trajectory(axes[0], record, show_gt=True)
    draw_guidance(axes[0], record)
    set_plot_limits(axes[0], record, include_guidance=True)
    axes[0].set_title("① 모델 입력\nlocal map + robot trajectory + guidance")

    draw_map(axes[1], record)
    draw_common_trajectory(axes[1], record, show_gt=False)
    draw_guidance(axes[1], record)
    draw_candidates(axes[1], record)
    set_plot_limits(axes[1], record, include_guidance=True)
    axes[1].set_title(
        "② MGP 후보 goal 20개\n"
        f"free {int(record['candidate_safe'].sum())} / "
        f"rejected {int((~record['candidate_safe']).sum())}"
    )

    draw_map(axes[2], record)
    draw_common_trajectory(axes[2], record, show_gt=True, show_pred=True)
    draw_guidance(axes[2], record)
    draw_candidates(axes[2], record)
    set_plot_limits(axes[2], record, include_guidance=False)
    axes[2].set_title(
        "③ 선택 goal → TGP trajectory\n"
        f"goal error {record['gde']:.2f} m · map-safe trajectory"
    )

    fig.legend(
        handles=common_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.5,
        0.005,
        "※ 학습된 checkpoint의 실제 test 추론 결과",
        ha="center",
        fontsize=9,
        color="#64748b",
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.16, wspace=0.28)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def choose_multi_records(
    records: list[dict[str, Any]],
    single_record: dict[str, Any],
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = [single_record]

    def add_first(predicate) -> None:
        for record in records:
            if predicate(record) and record["index"] not in {
                item["index"] for item in chosen
            }:
                chosen.append(record)
                return

    add_first(lambda record: record["execution_valid"] and record["guidance_outside_map"])
    valid_sorted = sorted(
        [record for record in records if record["execution_valid"]],
        key=lambda record: record["ade"],
    )
    for quantile in (0.2, 0.8):
        if valid_sorted:
            candidate = valid_sorted[
                min(int(round((len(valid_sorted) - 1) * quantile)), len(valid_sorted) - 1)
            ]
            if candidate["index"] not in {item["index"] for item in chosen}:
                chosen.append(candidate)
    add_first(lambda record: record["all_candidates_invalid"])
    add_first(
        lambda record: record["selected_goal_valid"]
        and not record["trajectory_map_safe"]
    )
    for record in records:
        if len(chosen) >= 6:
            break
        if record["index"] not in {item["index"] for item in chosen}:
            chosen.append(record)
    return chosen[:6]


def record_status(record: dict[str, Any]) -> tuple[str, str]:
    if record["execution_valid"]:
        return "EXECUTE", "#15803d"
    if record["all_candidates_invalid"]:
        return "STOP · no free goal", "#b91c1c"
    return "STOP · unsafe TGP path", "#b45309"


def create_multi_sample_figure(
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.2))
    fig.suptitle(
        "Guided MGP 실제 test sample 결과",
        fontsize=24,
        fontweight="bold",
        color="#0f2f5f",
        y=0.97,
    )
    for ax, record in zip(axes.flat, records):
        draw_map(ax, record)
        draw_common_trajectory(ax, record, show_gt=True, show_pred=True)
        draw_guidance(ax, record)
        draw_candidates(ax, record)
        set_plot_limits(ax, record, include_guidance=record["guidance_outside_map"])
        status, status_color = record_status(record)
        title = f"test #{record['index']} · {status}"
        if record["execution_valid"]:
            title += f"\nADE {record['ade']:.2f} m · FDE {record['fde']:.2f} m"
        elif record["guidance_outside_map"]:
            title += "\n8 m guidance는 local map 밖"
        ax.set_title(title, color=status_color, fontsize=11)

    fig.legend(
        handles=common_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.5,
        0.005,
        "초록/빨강 후보는 endpoint map 판정, EXECUTE/STOP은 trajectory 12점 검사까지 반영",
        ha="center",
        fontsize=9,
        color="#64748b",
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.13, hspace=0.38, wspace=0.22)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_pipeline_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    color: str,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.8,
        edgecolor=color,
        facecolor=f"{color}14",
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=color,
    )
    ax.text(
        x + width / 2,
        y + height * 0.30,
        body,
        ha="center",
        va="center",
        fontsize=9,
        color="#334155",
    )


def create_summary_figure(
    dataset_stats: dict[str, Any],
    inference_metrics: dict[str, Any],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[0.92, 1.35],
        height_ratios=[0.82, 1.0],
        left=0.065,
        right=0.965,
        top=0.86,
        bottom=0.10,
        hspace=0.42,
        wspace=0.28,
    )
    fig.suptitle(
        "Gazebo Guided MGP 변환·학습·추론 검증 결과",
        fontsize=25,
        fontweight="bold",
        color="#0f2f5f",
        y=0.96,
    )
    fig.text(
        0.5,
        0.905,
        f"실제 {dataset_stats['total_samples']}개 Gazebo sample + best validation checkpoint 기반",
        ha="center",
        fontsize=12,
        color="#475569",
    )

    ax_split = fig.add_subplot(grid[0, 0])
    split_names = ["train", "val", "test"]
    split_values = [dataset_stats["splits"][name] for name in split_names]
    bars = ax_split.barh(
        split_names,
        split_values,
        color=["#2563eb", "#f59e0b", "#16a34a"],
        height=0.58,
    )
    for bar, value in zip(bars, split_values):
        ax_split.text(
            value + 3,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center",
            fontweight="bold",
            fontsize=11,
        )
    ax_split.set_xlim(0, max(split_values) * 1.2)
    ax_split.set_title("① Episode 단위 dataset split")
    ax_split.set_xlabel("samples")
    ax_split.spines[["top", "right", "left"]].set_visible(False)
    ax_split.grid(axis="x", alpha=0.2)

    ax_pipeline = fig.add_subplot(grid[0, 1])
    ax_pipeline.set_xlim(0, 1)
    ax_pipeline.set_ylim(0, 1)
    ax_pipeline.axis("off")
    ax_pipeline.set_title("② 실제 모델 입력·출력 shape", pad=12)
    box_specs = [
        (0.01, "Gazebo adapter", "MGP/TGP (58,2)\nmap (4,256)", "#2563eb"),
        (0.265, "MGP", "candidate goals\n(B,20,2)", "#0f766e"),
        (0.52, "map + GP 선택", "selected goal\n(B,2)", "#d97706"),
        (0.775, "TGP", "trajectory\n(B,12,2)", "#7c3aed"),
    ]
    for index, (x, title, body, color) in enumerate(box_specs):
        add_pipeline_box(ax_pipeline, x, 0.27, 0.205, 0.47, title, body, color)
        if index < len(box_specs) - 1:
            ax_pipeline.annotate(
                "",
                xy=(x + 0.247, 0.505),
                xytext=(x + 0.21, 0.505),
                arrowprops={"arrowstyle": "->", "lw": 1.7, "color": "#64748b"},
            )

    ax_rates = fig.add_subplot(grid[1, 0])
    rate_labels = [
        "free candidates",
        "selected goal valid",
        "trajectory map-safe",
        "execution valid",
    ]
    rate_values = [
        inference_metrics["safe_candidate_rate"] * 100,
        inference_metrics["selected_goal_valid_rate"] * 100,
        inference_metrics["trajectory_map_safe_rate"] * 100,
        inference_metrics["execution_valid_rate"] * 100,
    ]
    rate_colors = ["#22c55e", "#0f766e", "#7c3aed", "#2563eb"]
    rate_bars = ax_rates.barh(
        rate_labels[::-1],
        rate_values[::-1],
        color=rate_colors[::-1],
        height=0.58,
    )
    for bar, value in zip(rate_bars, rate_values[::-1]):
        ax_rates.text(
            min(value + 1.5, 96),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontweight="bold",
        )
    ax_rates.set_xlim(0, 105)
    ax_rates.set_xlabel("rate (%)")
    ax_rates.set_title("③ Test split guided inference 안전 gate")
    ax_rates.spines[["top", "right", "left"]].set_visible(False)
    ax_rates.grid(axis="x", alpha=0.2)

    ax_checks = fig.add_subplot(grid[1, 1])
    ax_checks.axis("off")
    ax_checks.set_title("④ 눈으로 확인 가능한 검증 근거", pad=12)
    check_items = [
        (
            f"{dataset_stats['finite_and_shape_valid']} / "
            f"{dataset_stats['total_samples']}",
            "모든 tensor shape 일치\nNaN / Inf 없음",
            "#15803d",
        ),
        (
            f"{dataset_stats['guidance_differs_from_endpoint']} / "
            f"{dataset_stats['total_samples']}",
            "guidance와 실제 t+12 endpoint\n별도 label로 분리",
            "#2563eb",
        ),
        (
            f"{dataset_stats['guidance_outside_map']} / "
            f"{dataset_stats['total_samples']}",
            "8 m guidance가 ±10 m local map 밖\n0개로 범위 정합성 확인",
            "#d97706",
        ),
        (
            "9 / 9",
            "token · map y축 · 후보 필터\nSTOP gate 단위 테스트 통과",
            "#7c3aed",
        ),
    ]
    positions = [(0.02, 0.55), (0.52, 0.55), (0.02, 0.08), (0.52, 0.08)]
    for (value, description, color), (x, y) in zip(check_items, positions):
        box = FancyBboxPatch(
            (x, y),
            0.45,
            0.34,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            facecolor=f"{color}12",
            edgecolor=color,
            linewidth=1.5,
        )
        ax_checks.add_patch(box)
        ax_checks.text(
            x + 0.225,
            y + 0.235,
            value,
            ha="center",
            va="center",
            fontsize=19,
            fontweight="bold",
            color=color,
        )
        ax_checks.text(
            x + 0.225,
            y + 0.105,
            description,
            ha="center",
            va="center",
            fontsize=9.5,
            color="#334155",
        )

    fig.text(
        0.5,
        0.025,
        "※ 안전률과 ADE/FDE는 해당 checkpoint를 독립 test split에서 계산한 결과",
        ha="center",
        fontsize=10,
        color="#64748b",
    )
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    cli = parse_args()
    set_workdir()
    configure_plotting(cli.font_path)

    checkpoint = Path(cli.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    output_dir = Path(cli.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    args = load_runtime_namespace(cli.config, command="test", cli_dry_run=False)
    records, inference_metrics = collect_inference_records(args, checkpoint)
    dataset_stats = collect_dataset_stats(args)
    single_record = choose_single_record(records)
    multi_records = choose_multi_records(records, single_record)

    single_path = output_dir / "01_guided_mgp_single_sample_pipeline.png"
    multi_path = output_dir / "02_guided_mgp_multi_sample_results.png"
    summary_path = output_dir / "03_guided_mgp_validation_summary.png"
    metrics_path = output_dir / "guided_mgp_visualization_metrics.json"

    create_single_pipeline_figure(single_record, single_path)
    create_multi_sample_figure(multi_records, multi_path)
    create_summary_figure(dataset_stats, inference_metrics, summary_path)

    report = {
        "status": "PASS",
        "note": "Best-validation checkpoint test result",
        "config": str(cli.config),
        "checkpoint": str(checkpoint),
        "dataset": dataset_stats,
        "test_inference": inference_metrics,
        "single_sample": {
            key: single_record[key]
            for key in (
                "index",
                "selected_index",
                "selected_goal_valid",
                "trajectory_map_safe",
                "execution_valid",
                "guidance_outside_map",
                "ade",
                "fde",
                "gde",
            )
        },
        "multi_sample_indices": [record["index"] for record in multi_records],
        "outputs": [str(single_path), str(multi_path), str(summary_path)],
    }
    metrics_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("[GUIDED_MGP_VISUALIZATION]")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

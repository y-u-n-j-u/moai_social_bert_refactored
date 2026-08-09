#!/usr/bin/env python3
"""Summarize and visualize live guided SPU-BERT runtime diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostics", type=Path, help="Runtime JSONL file")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--examples-per-reason", type=int, default=3)
    parser.add_argument("--map-yaml", type=Path, default=None)
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            record["_line_number"] = line_number
            records.append(record)
    if not records:
        raise RuntimeError(f"No diagnostic records found in {path}")
    return records


def finite(values: Iterable[Optional[float]]) -> List[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def stats(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    items = finite(values)
    if not items:
        return {"min": None, "median": None, "mean": None, "max": None}
    array = np.asarray(items, dtype=float)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def build_summary(
    records: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    reasons = Counter(str(record.get("reason", "unknown")) for record in records)
    valid = int(reasons.get("valid", 0))
    total = len(records)
    global_paths = [
        event
        for event in events
        if event.get("event") == "global_path"
        and bool(event.get("compute_path_success"))
    ]
    fallbacks = [event for event in events if event.get("event") == "fallback"]
    holds = [event for event in events if event.get("event") == "guided_hold"]
    goal_generations = {
        int(event["goal_generation"])
        for event in global_paths
        if event.get("goal_generation") is not None
    }
    fallback_generations = {
        int(event["goal_generation"])
        for event in fallbacks
        if event.get("goal_generation") is not None
    }
    summary = {
        "total_predictions": total,
        "valid_predictions": valid,
        "rejected_predictions": total - valid,
        "valid_rate": valid / total,
        "reason_counts": dict(sorted(reasons.items())),
        "min_human_clearance_m": stats(
            record.get("metrics", {}).get("min_human_clearance_m") for record in records
        ),
        "max_step_m": stats(record.get("metrics", {}).get("max_step_m") for record in records),
        "goal_progress_m": stats(
            record.get("metrics", {}).get("goal_progress_m") for record in records
        ),
        "footprint_collision_count": stats(
            record.get("metrics", {}).get("footprint_collision_count") for record in records
        ),
        "goal_count": len(goal_generations),
        "fallback_goal_count": len(fallback_generations),
        "model_only_goal_count": len(goal_generations - fallback_generations),
        "fallback_goal_rate": (
            len(fallback_generations) / len(goal_generations)
            if goal_generations
            else None
        ),
        "fallback_reason_counts": dict(sorted(Counter(
            str(event.get("reason", "unknown")) for event in fallbacks
        ).items())),
        "hold_event_count": len(holds),
        "hold_reason_counts": dict(sorted(Counter(
            str(event.get("reason", "unknown")) for event in holds
        ).items())),
        "max_hold_rejection_streak": max(
            (int(event.get("rejection_streak", 0)) for event in holds),
            default=0,
        ),
        "recovered_after_hold_predictions": sum(
            bool(record.get("valid"))
            and int(record.get("rejection_streak_before", 0)) > 0
            for record in records
        ),
    }

    top_k_records = [
        record
        for record in records
        if isinstance(record.get("candidate_attempts"), list)
        and record["candidate_attempts"]
    ]
    if top_k_records:
        rank_one_valid = sum(
            bool(record["candidate_attempts"][0].get("valid"))
            for record in top_k_records
        )
        rescued = [
            record
            for record in top_k_records
            if bool(record.get("valid"))
            and not bool(record["candidate_attempts"][0].get("valid"))
        ]
        selected_ranks = Counter(
            int(record.get("selected_candidate_rank", 1))
            for record in top_k_records
            if bool(record.get("valid"))
        )
        attempt_reasons = Counter(
            str(attempt.get("reason", "unknown"))
            for record in top_k_records
            for attempt in record["candidate_attempts"]
        )
        summary["top_k_selection"] = {
            "records": len(top_k_records),
            "configured_top_k": int(top_k_records[0].get("tgp_top_k", 1)),
            "rank_one_valid_predictions": rank_one_valid,
            "rank_one_valid_rate": rank_one_valid / len(top_k_records),
            "top_k_valid_predictions": sum(
                bool(record.get("valid")) for record in top_k_records
            ),
            "top_k_valid_rate": sum(
                bool(record.get("valid")) for record in top_k_records
            ) / len(top_k_records),
            "rescued_predictions": len(rescued),
            "selected_rank_counts": {
                str(rank): count for rank, count in sorted(selected_ranks.items())
            },
            "attempt_count": stats(
                record.get("attempted_candidate_count") for record in top_k_records
            ),
            "attempt_reason_counts": dict(sorted(attempt_reasons.items())),
        }
    return summary


def resolve_map_yaml(
    explicit: Optional[Path], record: Dict[str, Any], diagnostics: Path
) -> Optional[Path]:
    if explicit is not None:
        return explicit.resolve()
    raw = str(record.get("map_yaml_path", ""))
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_file():
        return candidate
    container_root = "/home/hunav_gz_classic_ws/"
    if raw.startswith(container_root):
        workspace = diagnostics.resolve().parent.parent
        relative = raw[len(container_root):]
        candidate = workspace / relative
        if candidate.is_file():
            return candidate
        install_prefix = "install/hunav_gazebo_wrapper/share/hunav_gazebo_wrapper/"
        if relative.startswith(install_prefix):
            candidate = (
                workspace / "src/hunav_gazebo_wrapper"
                / relative[len(install_prefix):]
            )
            if candidate.is_file():
                return candidate
    return None


def load_map(path: Optional[Path]) -> Optional[Tuple[np.ndarray, Tuple[float, float, float, float]]]:
    if path is None or not path.is_file():
        return None
    meta: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip("'\"")
    image_path = Path(meta["image"])
    if not image_path.is_absolute():
        image_path = path.parent / image_path
    image = plt.imread(image_path)
    if image.ndim == 3:
        image = image[..., 0]
    resolution = float(meta.get("resolution", 0.05))
    origin = [float(item.strip()) for item in meta.get("origin", "[0,0,0]")[1:-1].split(",")]
    height, width = image.shape[:2]
    extent = (
        origin[0], origin[0] + width * resolution,
        origin[1], origin[1] + height * resolution,
    )
    return np.flipud(image), extent


def as_xy(values: Any) -> np.ndarray:
    array = np.asarray(values or [], dtype=float)
    return array.reshape((-1, 2)) if array.size else np.empty((0, 2), dtype=float)


def plot_record(
    record: Dict[str, Any], output: Path,
    map_data: Optional[Tuple[np.ndarray, Tuple[float, float, float, float]]],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    if map_data is not None:
        image, extent = map_data
        ax.imshow(image, cmap="gray", origin="lower", extent=extent, vmin=0, vmax=255, alpha=0.65)

    robot = as_xy([record.get("robot")])
    path = as_xy(record.get("path"))
    candidates = as_xy(record.get("candidate_goals"))
    history = as_xy(record.get("robot_history"))
    combined = np.vstack([robot, path]) if path.size else robot
    valid = bool(record.get("valid"))
    reason = str(record.get("reason", "unknown"))
    metrics = record.get("metrics", {})

    if history.size:
        ax.plot(history[:, 0], history[:, 1], "k--", lw=1.2, alpha=0.7, label="robot history")
    if combined.size:
        ax.plot(
            combined[:, 0], combined[:, 1], "-o",
            color="#1565c0" if valid else "#d32f2f", lw=2.4, ms=3.5,
            label="TGP path",
        )
    if candidates.size:
        ax.scatter(candidates[:, 0], candidates[:, 1], s=28, color="#24a148", alpha=0.65, label="MGP candidates")

    selected = as_xy([record.get("selected_goal")])
    guidance = as_xy([record.get("guidance_point")])
    final_goal = as_xy([record.get("final_goal")])
    if robot.size:
        ax.scatter(robot[:, 0], robot[:, 1], s=90, color="black", marker="o", label="robot", zorder=8)
    if selected.size:
        ax.scatter(selected[:, 0], selected[:, 1], s=90, color="#0057b8", marker="D", label="selected goal", zorder=8)
    if guidance.size:
        ax.scatter(guidance[:, 0], guidance[:, 1], s=120, color="#f6c000", marker="s", edgecolor="black", label="guidance point", zorder=8)
    if final_goal.size:
        ax.scatter(final_goal[:, 0], final_goal[:, 1], s=150, color="#8e24aa", marker="*", label="final goal", zorder=8)

    for human in record.get("humans", []):
        current = as_xy([human.get("current")])
        predicted = as_xy(human.get("predicted"))
        if current.size:
            ax.scatter(current[:, 0], current[:, 1], s=65, color="#ef6c00", marker="^", zorder=6)
        if predicted.size:
            ax.plot(predicted[:, 0], predicted[:, 1], ":", color="#ef6c00", lw=1.2, alpha=0.8)

    collision_steps = metrics.get("footprint_collision_steps", []) or []
    if path.size and collision_steps:
        indices = [index for index in collision_steps if 0 <= int(index) < len(path)]
        if indices:
            points = path[indices]
            ax.scatter(points[:, 0], points[:, 1], s=130, marker="x", color="#ffcc00", linewidth=3, label="map collision", zorder=9)

    jump_index = int(metrics.get("max_step_index", -1) or -1)
    if reason == "kinematic_jump" and combined.size and 0 <= jump_index < len(path):
        segment = combined[jump_index:jump_index + 2]
        ax.plot(segment[:, 0], segment[:, 1], color="#7b1fa2", lw=6, alpha=0.7, label="max jump")

    human_position = as_xy([metrics.get("min_human_position")])
    robot_position = as_xy([metrics.get("min_human_robot_position")])
    if reason == "predicted_human_clearance" and human_position.size and robot_position.size:
        pair = np.vstack([robot_position, human_position])
        ax.plot(pair[:, 0], pair[:, 1], color="#c62828", lw=3, label="minimum human clearance")

    clearance = metrics.get("min_human_clearance_m")
    max_step = metrics.get("max_step_m")
    allowed = metrics.get("allowed_step_m")
    progress = metrics.get("goal_progress_m")
    details = []
    if clearance is not None:
        details.append(f"human clearance={clearance:.3f} m")
    if max_step is not None and allowed is not None:
        details.append(f"max step={max_step:.3f}/{allowed:.3f} m")
    if progress is not None:
        details.append(f"goal progress={progress:.3f} m")
    ax.set_title(
        "Guided SPU-BERT runtime sample\n"
        f"reason={reason} | line={record.get('_line_number')}\n"
        + " | ".join(details),
        fontsize=11,
    )
    ax.set_xlabel("map x [m]")
    ax.set_ylabel("map y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8, ncol=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def severity(record: Dict[str, Any]) -> float:
    reason = record.get("reason")
    metric = record.get("metrics", {})
    if reason == "predicted_human_clearance":
        return -float(metric.get("min_human_clearance_m") or 0.0)
    if reason == "kinematic_jump":
        return float(metric.get("max_step_m") or 0.0)
    if reason in {"robot_footprint_collision", "model_map_check_failed"}:
        return float(metric.get("footprint_collision_count") or 0.0)
    if reason == "insufficient_goal_progress":
        return -float(metric.get("goal_progress_m") or 0.0)
    return 0.0


def write_summary(summary: Dict[str, Any], output_dir: Path) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "reason_counts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["reason", "count", "fraction"])
        total = summary["total_predictions"]
        for reason, count in summary["reason_counts"].items():
            writer.writerow([reason, count, count / total])

    labels = list(summary["reason_counts"])
    counts = [summary["reason_counts"][label] for label in labels]
    colors = ["#24a148" if label == "valid" else "#d95f02" for label in labels]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    bars = ax.bar(labels, counts, color=colors)
    ax.bar_label(bars)
    ax.set_ylabel("prediction count")
    ax.set_title(f"Runtime validation reasons (valid rate {100 * summary['valid_rate']:.1f}%)")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(output_dir / "reason_counts.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    diagnostics = args.diagnostics.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else diagnostics.with_suffix("").with_name(diagnostics.stem + "_analysis")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    events = load_records(diagnostics)
    records = [event for event in events if event.get("event") == "prediction"]
    if not records:
        raise RuntimeError(f"No prediction records found in {diagnostics}")
    summary = build_summary(records, events)
    write_summary(summary, output_dir)
    map_yaml = resolve_map_yaml(args.map_yaml, records[0], diagnostics)
    map_data = load_map(map_yaml)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get("reason", "unknown")), []).append(record)
    for reason, items in grouped.items():
        items.sort(key=severity, reverse=True)
        for index, record in enumerate(items[: max(args.examples_per_reason, 0)], 1):
            plot_record(
                record,
                output_dir / "examples" / f"{reason}_{index:02d}.png",
                map_data,
            )

    print(f"Analyzed {len(records)} predictions")
    print(f"Valid rate: {100 * summary['valid_rate']:.2f}%")
    print(f"Summary: {output_dir / 'summary.json'}")
    print(f"Figures: {output_dir}")


if __name__ == "__main__":
    main()

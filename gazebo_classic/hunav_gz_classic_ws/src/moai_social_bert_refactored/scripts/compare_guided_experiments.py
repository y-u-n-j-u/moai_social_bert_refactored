#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Gazebo-only and ETH/UCY-pretrained guided models."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_metrics(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))["test_inference"]


def relative_change(before: float, after: float) -> float:
    return (after - before) / before * 100.0


def main() -> int:
    args = parse_args()
    baseline = load_metrics(args.baseline)
    pretrained = load_metrics(args.pretrained)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    error_keys = ("all_sample_ade", "all_sample_fde", "all_sample_gde")
    rate_keys = (
        "safe_candidate_rate",
        "selected_goal_valid_rate",
        "trajectory_map_safe_rate",
        "execution_valid_rate",
    )
    report = {
        "baseline": baseline,
        "ethucy_pretrained": pretrained,
        "relative_error_change_percent": {
            key: relative_change(baseline[key], pretrained[key])
            for key in error_keys
        },
        "rate_change_percentage_points": {
            key: (pretrained[key] - baseline[key]) * 100.0
            for key in rate_keys
        },
    }
    (output_dir / "gazebo_baseline_vs_ethucy_pretrained.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    fig.suptitle(
        "Gazebo-only vs ETH/UCY pretrain → Gazebo fine-tune",
        fontsize=19,
        fontweight="bold",
        color="#0f2f5f",
    )
    names = ["ADE", "FDE", "GDE"]
    x = np.arange(len(names))
    width = 0.36
    axes[0].bar(
        x - width / 2,
        [baseline[key] for key in error_keys],
        width,
        label="Gazebo-only",
        color="#2563eb",
    )
    axes[0].bar(
        x + width / 2,
        [pretrained[key] for key in error_keys],
        width,
        label="ETH/UCY pretrained",
        color="#f59e0b",
    )
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("error (m), lower is better")
    axes[0].set_title("Trajectory and goal errors")
    axes[0].legend(frameon=False)

    rate_names = ["candidate", "goal", "trajectory", "execution"]
    axes[1].bar(
        x=np.arange(len(rate_names)) - width / 2,
        height=[baseline[key] * 100 for key in rate_keys],
        width=width,
        label="Gazebo-only",
        color="#2563eb",
    )
    axes[1].bar(
        x=np.arange(len(rate_names)) + width / 2,
        height=[pretrained[key] * 100 for key in rate_keys],
        width=width,
        label="ETH/UCY pretrained",
        color="#f59e0b",
    )
    axes[1].set_xticks(np.arange(len(rate_names)), rate_names)
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("rate (%), higher is better")
    axes[1].set_title("Map-safety gates")
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(
        output_dir / "05_gazebo_baseline_vs_ethucy_pretrained.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

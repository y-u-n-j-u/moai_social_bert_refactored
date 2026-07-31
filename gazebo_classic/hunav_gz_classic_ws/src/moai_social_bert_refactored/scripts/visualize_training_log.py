#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRAIN_PATTERN = re.compile(
    r"^\[TRAIN\] epoch=(?P<epoch>\d+) "
    r"train=(?P<train>[0-9.eE+-]+) "
    r"val=(?P<val>[0-9.eE+-]+)"
    r"(?: ade=(?P<ade>[0-9.eE+-]+) "
    r"fde=(?P<fde>[0-9.eE+-]+) "
    r"gde=(?P<gde>[0-9.eE+-]+))?$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot metrics printed by train.py.")
    parser.add_argument("--log", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def parse_metrics(log_path: Path) -> list[dict[str, float | int]]:
    text = log_path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    records: list[dict[str, float | int]] = []
    for line in text.splitlines():
        match = TRAIN_PATTERN.match(line.strip())
        if not match:
            continue
        record: dict[str, float | int] = {
            "epoch": int(match.group("epoch")),
            "train_loss": float(match.group("train")),
            "val_loss": float(match.group("val")),
        }
        for name in ("ade", "fde", "gde"):
            value = match.group(name)
            if value is not None:
                record[name] = float(value)
        records.append(record)
    if not records:
        raise ValueError(f"No [TRAIN] epoch records found in {log_path}")
    return records


def create_figure(records: list[dict[str, float | int]], output_path: Path) -> None:
    epochs = [int(record["epoch"]) for record in records]
    has_test_metrics = all(
        all(name in record for name in ("ade", "fde", "gde"))
        for record in records
    )
    if has_test_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
        loss_axis, metric_axis = axes
    else:
        fig, loss_axis = plt.subplots(1, 1, figsize=(8, 5.6))
        axes = [loss_axis]
        metric_axis = None
    fig.suptitle(
        "Training curves",
        fontsize=20,
        fontweight="bold",
        color="#0f2f5f",
    )

    loss_axis.plot(
        epochs,
        [float(record["train_loss"]) for record in records],
        marker="o",
        markersize=3,
        linewidth=2,
        label="train loss",
        color="#2563eb",
    )
    loss_axis.plot(
        epochs,
        [float(record["val_loss"]) for record in records],
        marker="o",
        markersize=3,
        linewidth=2,
        label="validation loss",
        color="#f59e0b",
    )
    loss_axis.set_title("Loss")
    loss_axis.set_xlabel("epoch")
    loss_axis.set_ylabel("loss")
    loss_axis.legend(frameon=False)

    if metric_axis is not None:
        for name, color in (("ade", "#7c3aed"), ("fde", "#dc2626"), ("gde", "#0f766e")):
            metric_axis.plot(
                epochs,
                [float(record[name]) for record in records],
                marker="o",
                markersize=3,
                linewidth=2,
                label=name.upper(),
                color=color,
            )
        metric_axis.set_title("Independent test metrics")
        metric_axis.set_xlabel("epoch")
        metric_axis.set_ylabel("error (m)")
        metric_axis.legend(frameon=False)

    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    log_path = Path(args.log).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = parse_metrics(log_path)
    best = min(records, key=lambda record: float(record["val_loss"]))
    report = {
        "epochs_completed": len(records),
        "first_epoch": records[0],
        "best_validation_epoch": best,
        "last_epoch": records[-1],
    }
    (output_dir / "training_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    create_figure(records, output_dir / "04_training_curves.png")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

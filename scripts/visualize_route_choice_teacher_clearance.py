#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Rectangle
import numpy as np
from PIL import Image


BLOCK_BOUNDS = (0.8, 4.8, -1.9, 3.5)
REQUIRED_CLEARANCE_M = 0.375


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-yaml", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_simple_yaml(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if value.startswith("["):
            values[key] = [float(part.strip()) for part in value[1:-1].split(",")]
        else:
            values[key] = value.strip('"\'')
    return values


def load_map(path: Path) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    info = load_simple_yaml(path)
    image_path = path.parent / str(info["image"])
    pixels = np.asarray(Image.open(image_path).convert("L"))
    occupied = pixels < 128
    resolution = float(info["resolution"])
    origin_x, origin_y, _ = info["origin"]
    extent = (
        float(origin_x),
        float(origin_x) + occupied.shape[1] * resolution,
        float(origin_y),
        float(origin_y) + occupied.shape[0] * resolution,
    )
    return occupied, extent


def load_recording(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return pickle.load(stream)


def reconstruct_row(recording: dict[str, Any], row: int) -> np.ndarray:
    by_frame: dict[int, np.ndarray] = {}
    for trajectories, meta in zip(
        recording["all_trajs"], recording["sample_meta"]
    ):
        start_frame = int(meta["start_frame"])
        if row >= trajectories.shape[0]:
            continue
        for offset, point in enumerate(trajectories[row]):
            by_frame.setdefault(start_frame + offset, np.asarray(point, dtype=float))
    return np.asarray([by_frame[index] for index in sorted(by_frame)], dtype=float)


def densify(points: np.ndarray, spacing: float = 0.01) -> np.ndarray:
    dense: list[np.ndarray] = []
    for start, end in zip(points[:-1], points[1:]):
        distance = float(np.linalg.norm(end - start))
        count = max(1, int(np.ceil(distance / spacing)))
        dense.extend(start + (end - start) * (index / count) for index in range(count))
    dense.append(points[-1])
    return np.asarray(dense)


def block_distance(points: np.ndarray) -> np.ndarray:
    x_min, x_max, y_min, y_max = BLOCK_BOUNDS
    dx = np.maximum(np.maximum(x_min - points[:, 0], 0.0), points[:, 0] - x_max)
    dy = np.maximum(np.maximum(y_min - points[:, 1], 0.0), points[:, 1] - y_max)
    return np.hypot(dx, dy)


def closest_to_block(points: np.ndarray) -> tuple[np.ndarray, float]:
    dense = densify(points)
    distances = block_distance(dense)
    index = int(np.argmin(distances))
    return dense[index], float(distances[index])


def draw_map(axis, occupied: np.ndarray, extent: tuple[float, ...]) -> None:
    axis.imshow(
        occupied,
        origin="upper",
        extent=extent,
        cmap=ListedColormap(["#f4f7f6", "#273238"]),
        interpolation="nearest",
        zorder=0,
    )
    x_min, x_max, y_min, y_max = BLOCK_BOUNDS
    axis.add_patch(
        Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            edgecolor="#111719",
            linewidth=1.2,
            zorder=2,
        )
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color="#aeb9bc", alpha=0.18, linewidth=0.6)
    axis.set_xlabel("map x [m]")
    axis.set_ylabel("map y [m]")


def main() -> int:
    args = parse_args()
    occupied, extent = load_map(args.map_yaml)
    v2_recording = load_recording(args.v2)
    v3_recording = load_recording(args.v3)
    v2_path = reconstruct_row(v2_recording, 0)
    v3_path = reconstruct_row(v3_recording, 0)
    human_paths = [
        reconstruct_row(v3_recording, row)
        for row in range(1, v3_recording["all_trajs"][0].shape[0])
    ]
    v2_closest, v2_clearance = closest_to_block(v2_path)
    v3_closest, v3_clearance = closest_to_block(v3_path)

    fig, (full, zoom) = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for axis in (full, zoom):
        draw_map(axis, occupied, extent)
        axis.plot(
            v2_path[:, 0],
            v2_path[:, 1],
            color="#d64b45",
            linewidth=2.4,
            linestyle="--",
            label="V2: soft margin (rejected)",
            zorder=4,
        )
        axis.plot(
            v3_path[:, 0],
            v3_path[:, 1],
            color="#16846b",
            linewidth=2.8,
            label="V3: 0.40 m virtual footprint (passed)",
            zorder=5,
        )

    for index, human_path in enumerate(human_paths):
        full.plot(
            human_path[:, 0],
            human_path[:, 1],
            color="#2878b5",
            alpha=0.65,
            linewidth=1.5,
            label="pedestrian trajectories" if index == 0 else None,
            zorder=3,
        )

    start = v3_path[0]
    goal = np.asarray(v3_recording["sample_meta"][0]["final_goal"], dtype=float)
    guidance = np.asarray(v3_recording["sample_meta"][0]["guidance_point"], dtype=float)
    full.scatter(*start, marker="o", s=70, color="#222222", label="start", zorder=7)
    full.scatter(*goal, marker="X", s=90, color="#222222", label="final goal", zorder=7)
    full.scatter(
        *guidance,
        marker="*",
        s=180,
        color="#f2ba2f",
        edgecolor="#6e5200",
        linewidth=0.7,
        label="route guidance point",
        zorder=8,
    )
    full.set_xlim(-9.5, 9.8)
    full.set_ylim(-5.2, 7.2)
    full.set_title("Route-choice teacher trajectory on the training map")
    full.legend(loc="lower center", ncol=2, frameon=True, fontsize=9)

    corner = np.asarray([BLOCK_BOUNDS[0], BLOCK_BOUNDS[3]], dtype=float)
    zoom.add_patch(
        Circle(
            corner,
            REQUIRED_CLEARANCE_M,
            fill=False,
            linestyle=":",
            linewidth=2.0,
            edgecolor="#a36a00",
            label="required 0.375 m centre clearance",
            zorder=6,
        )
    )
    zoom.scatter(*v2_closest, s=70, color="#d64b45", zorder=8)
    zoom.scatter(*v3_closest, s=70, color="#16846b", zorder=8)
    zoom.annotate(
        f"V2 {v2_clearance:.3f} m",
        v2_closest,
        xytext=(-72, -32),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#d64b45"},
        color="#a62d29",
        fontsize=11,
        weight="bold",
    )
    zoom.annotate(
        f"V3 {v3_clearance:.3f} m",
        v3_closest,
        xytext=(24, 30),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#16846b"},
        color="#0e6653",
        fontsize=11,
        weight="bold",
    )
    zoom.set_xlim(-0.35, 1.7)
    zoom.set_ylim(2.95, 4.65)
    zoom.set_title("Upper-left obstacle corner: swept-path clearance")
    zoom.legend(loc="upper right", fontsize=9, frameon=True)

    fig.suptitle(
        "Safe-teacher V2 vs V3 | Same seed 204 and profile y=+1.4",
        fontsize=16,
        weight="bold",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, facecolor="white")
    plt.close(fig)
    print(
        f"saved {args.out}; V2 clearance={v2_clearance:.4f} m, "
        f"V3 clearance={v3_clearance:.4f} m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

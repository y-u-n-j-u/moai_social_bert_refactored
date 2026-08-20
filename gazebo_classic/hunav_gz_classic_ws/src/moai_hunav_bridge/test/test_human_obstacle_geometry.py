from __future__ import annotations

import importlib.util
from math import cos, hypot, pi, sin
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_PATH = PACKAGE_ROOT / "moai_hunav_bridge" / "human_obstacle_geometry.py"
SPEC = importlib.util.spec_from_file_location("human_obstacle_geometry", GEOMETRY_PATH)
GEOMETRY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GEOMETRY)


def nearest_sample_distance(
    offsets: list[tuple[float, float]],
    query_x: float,
    query_y: float,
) -> float:
    return min(hypot(x - query_x, y - query_y) for x, y in offsets)


def test_filled_disc_has_no_hollow_safety_band() -> None:
    radius = 0.80
    spacing = 0.20
    offsets = GEOMETRY.filled_disc_offsets(radius, spacing, 16)

    assert offsets[0] == (0.0, 0.0)
    assert abs(max(hypot(x, y) for x, y in offsets) - radius) < 1e-9

    for radial_index in range(17):
        query_radius = radius * radial_index / 16
        for angle_index in range(32):
            angle = 2.0 * pi * angle_index / 32
            distance = nearest_sample_distance(
                offsets,
                query_radius * cos(angle),
                query_radius * sin(angle),
            )
            assert distance <= 0.16


def test_zero_ring_points_preserves_center_only_mode() -> None:
    assert GEOMETRY.filled_disc_offsets(0.80, 0.20, 0) == [(0.0, 0.0)]

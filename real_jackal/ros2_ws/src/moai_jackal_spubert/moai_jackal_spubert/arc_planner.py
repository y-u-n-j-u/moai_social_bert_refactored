"""Model-free local planner geometry: constant-curvature arcs from the robot pose.

Pure functions (no ROS, numpy or model imports). The bridge validates every arc
with the same ``validate_candidate_path`` used for SPU-BERT output, so obstacle,
pedestrian and progress rules are identical; this module only proposes shapes.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

XY = Tuple[float, float]


def curvature_fan(maximum: float, step: float) -> List[float]:
    """0, +step, -step, ... up to ``maximum`` (1/m); straight first."""
    if not (maximum >= 0.0 and step > 0.0):
        return [0.0]
    count = int(round(maximum / step + 1e-9))
    result = [0.0]
    for k in range(1, count + 1):
        result.extend((k * step, -k * step))
    return result


def arc_points(start: XY, yaw: float, curvature: float, length: float, count: int) -> Optional[List[XY]]:
    """``count`` points spaced ``length/count`` along an arc that starts along ``yaw``.

    Positive curvature turns left. Returns None for non-finite or degenerate input.
    """
    values = (start[0], start[1], yaw, curvature, length)
    if count < 1 or not all(math.isfinite(float(v)) for v in values) or not length > 0.0:
        return None
    spacing = float(length) / count
    x, y, heading = float(start[0]), float(start[1]), float(yaw)
    result: List[XY] = []
    for _ in range(count):
        turn = curvature * spacing
        if abs(turn) < 1e-9:
            x += spacing * math.cos(heading)
            y += spacing * math.sin(heading)
        else:
            chord = 2.0 * math.sin(turn / 2.0) / curvature
            x += chord * math.cos(heading + turn / 2.0)
            y += chord * math.sin(heading + turn / 2.0)
            heading += turn
        result.append((x, y))
    return result


def arc_cost(end: XY, target: XY, curvature: float, length: float,
             maximum_length: float, heading_offset: float = 0.0,
             curvature_weight: float = 0.15, length_weight: float = 0.10,
             heading_weight: float = 0.25) -> float:
    """Lower is better: near the target, gentle, straight-ahead, and long.

    ``heading_offset`` is how far the arc's start direction is turned away from
    the robot's heading (the tracker pivots in place for large offsets).
    """
    return (
        math.hypot(end[0] - target[0], end[1] - target[1])
        + curvature_weight * abs(curvature)
        + length_weight * max(maximum_length - length, 0.0)
        + heading_weight * abs(heading_offset)
    )

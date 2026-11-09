"""Fallback paths that end at the goal samples (MGP) of the model, used when every model candidate was rejected.

The model's own path (TGP) to a goal sample may wiggle through a wall; a smooth curve that leaves the robot along its heading and
runs to the same goal sample usually does not. These curves are only candidates: the bridge validates each of them with the same
checks as every other path and picks among the valid ones with the same selector as for model candidates.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

XY = Tuple[float, float]


def curve_to_goal(
    start: XY,
    yaw: float,
    goal: XY,
    count: int,
    spacing: Optional[float] = None,
    tangent_m: float = 1.0,
    max_angle_rad: float = 1.4,
    minimum_distance_m: float = 0.3,
    max_length_m: float = 4.0,
    lateral_m: float = 0.0,
) -> Optional[List[XY]]:
    """Points of a quadratic Bezier from ``start`` to ``goal``: one every ``spacing`` along the curve, at most ``count``.

    With ``spacing=None`` the ``count`` points are spread over the whole curve (at most ``max_length_m`` of it), so the path ends AT the
    goal sample whenever the curve is not longer than ``max_length_m``.

    The curve leaves ``start`` along ``yaw``. A goal that lies more than ``max_angle_rad`` off the heading (behind or beside the robot)
    gives None, as does a goal closer than ``minimum_distance_m``. When the curve is shorter than ``count * spacing`` the last point is
    the goal itself, so the path really ends at the goal sample.
    """
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    distance = math.hypot(dx, dy)
    if distance < minimum_distance_m or count < 1 or (spacing is not None and spacing <= 0.0):
        return None
    off_heading = abs((math.atan2(dy, dx) - yaw + math.pi) % (2.0 * math.pi) - math.pi)
    if off_heading > max_angle_rad:
        return None
    reach = min(0.5 * distance, max(tangent_m, 0.0))
    # lateral_m > 0 moves the control point to the left of the heading (< 0 to the right): a rounder curve that keeps away from a wall on the other side
    control = (start[0] + reach * math.cos(yaw) - lateral_m * math.sin(yaw), start[1] + reach * math.sin(yaw) + lateral_m * math.cos(yaw))
    dense = 240
    curve = []
    for step in range(dense + 1):
        t = step / dense
        a, b, c = (1.0 - t) ** 2, 2.0 * (1.0 - t) * t, t * t
        curve.append((a * start[0] + b * control[0] + c * goal[0], a * start[1] + b * control[1] + c * goal[1]))
    cumulative = [0.0]
    for previous, point in zip(curve, curve[1:]):
        cumulative.append(cumulative[-1] + math.hypot(point[0] - previous[0], point[1] - previous[1]))
    total = cumulative[-1]
    if spacing is None:
        spacing = min(total, max(max_length_m, 0.1)) / count
    points: List[XY] = []
    index = 0
    for k in range(1, count + 1):
        wanted = k * spacing
        if wanted > total:
            break
        while index < len(cumulative) - 1 and cumulative[index + 1] < wanted:
            index += 1
        span = cumulative[index + 1] - cumulative[index]
        ratio = 0.0 if span <= 1e-12 else (wanted - cumulative[index]) / span
        points.append((curve[index][0] + ratio * (curve[index + 1][0] - curve[index][0]),
                       curve[index][1] + ratio * (curve[index + 1][1] - curve[index][1])))
    if len(points) < count and (not points or math.hypot(goal[0] - points[-1][0], goal[1] - points[-1][1]) > 0.05):
        points.append((float(goal[0]), float(goal[1])))
    return points or None


def curve_shapes(tangent_m: float = 1.0, lateral_m: float = 0.6) -> List[Tuple[float, float]]:
    """(tangent, lateral) shapes tried for one goal sample, the plain one first: shorter / longer start, pushed left / right."""
    shapes = [(tangent_m, 0.0)]
    for tangent in (tangent_m, 0.5 * tangent_m, 1.6 * tangent_m):
        for lateral in (lateral_m, -lateral_m):
            shapes.append((tangent, lateral))
    for lateral in (0.5 * lateral_m, -0.5 * lateral_m):
        shapes.append((0.5 * tangent_m, lateral))
    seen, unique = set(), []
    for shape in shapes:
        key = (round(shape[0], 3), round(shape[1], 3))
        if key not in seen:
            seen.add(key)
            unique.append(shape)
    return unique


def pick_goal_samples(
    goals: Sequence[XY],
    safe: Sequence[bool],
    guidance: XY,
    maximum: int = 5,
    minimum_separation_m: float = 0.4,
) -> List[Tuple[int, XY]]:
    """Safe goal samples nearest to the guidance point first, no two closer than ``minimum_separation_m``: (index, goal)."""
    order = sorted(
        (i for i, g in enumerate(goals) if i < len(safe) and safe[i] and all(math.isfinite(v) for v in g[:2])),
        key=lambda i: math.hypot(goals[i][0] - guidance[0], goals[i][1] - guidance[1]),
    )
    picked: List[Tuple[int, XY]] = []
    for i in order:
        point = (float(goals[i][0]), float(goals[i][1]))
        if all(math.hypot(point[0] - q[0], point[1] - q[1]) >= minimum_separation_m for _, q in picked):
            picked.append((i, point))
        if len(picked) >= maximum:
            break
    return picked

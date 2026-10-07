"""Pure helpers for choosing among valid path candidates (no ROS, numpy or model).

The bridge used to publish the first valid candidate by model rank. When the
top candidate was rejected on one cycle and accepted on the next, the published
path flipped sideways and the tracker chased a target that jumped between left
and right. These helpers add continuity with the previously published path and a
limit on how sharply a path may leave the robot's heading.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

XY = Tuple[float, float]


@dataclass(frozen=True)
class SelectionConfig:
    rank_weight: float = 0.0            # cost per model rank step
    continuity_weight: float = 0.0      # cost per metre of lateral jump vs the previous path
    heading_weight: float = 0.0         # cost per radian of initial heading deviation
    heading_limit_rad: float = 0.0      # drop candidates beyond this if others remain (0 = off)
    lookahead_points: int = 6
    previous_max_age_s: float = 1.5

    @property
    def active(self) -> bool:
        return (self.rank_weight > 0.0 or self.continuity_weight > 0.0
                or self.heading_weight > 0.0 or self.heading_limit_rad > 0.0)


def point_to_polyline_distance(point: XY, polyline: Sequence[XY]) -> float:
    if not polyline:
        return math.inf
    px, py = float(point[0]), float(point[1])
    if len(polyline) == 1:
        return math.hypot(px - polyline[0][0], py - polyline[0][1])
    best = math.inf
    for (ax, ay), (bx, by) in zip(polyline, polyline[1:]):
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        t = 0.0 if seg2 <= 1e-12 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg2))
        best = min(best, math.hypot(px - (ax + t * dx), py - (ay + t * dy)))
    return best


def continuity_cost(candidate: Sequence[XY], previous: Optional[Sequence[XY]], count: int) -> float:
    """Mean distance of the candidate's first ``count`` points to the previous path (m)."""
    if not previous or len(previous) < 2 or not candidate:
        return 0.0
    head = candidate[: max(count, 1)]
    return sum(point_to_polyline_distance(p, previous) for p in head) / len(head)


def heading_deviation(robot: XY, yaw: float, candidate: Sequence[XY], index: int = 3) -> float:
    """Angle (rad, >= 0) between the robot heading and the direction to the ``index``-th point."""
    if not candidate:
        return 0.0
    p = candidate[min(index, len(candidate) - 1)]
    dx, dy = p[0] - robot[0], p[1] - robot[1]
    if math.hypot(dx, dy) < 1e-6:
        return 0.0
    diff = math.atan2(dy, dx) - yaw
    return abs((diff + math.pi) % (2.0 * math.pi) - math.pi)


def choose_candidate(
    paths: Sequence[Sequence[XY]],
    robot: XY,
    yaw: float,
    previous: Optional[Sequence[XY]],
    previous_age_s: float,
    config: SelectionConfig,
    extra_costs: Optional[Sequence[float]] = None,
) -> int:
    """Index into ``paths`` (given in model-rank order) of the candidate to publish.

    With no weights set this returns 0, the first valid candidate, exactly as before.
    """
    if not paths:
        raise ValueError("no candidates")
    extra = list(extra_costs) if extra_costs is not None else [0.0] * len(paths)
    if len(extra) != len(paths):
        raise ValueError("extra_costs must match paths")
    if len(paths) == 1 or not (config.active or any(c > 0.0 for c in extra)):
        return 0
    usable = list(range(len(paths)))
    if config.heading_limit_rad > 0.0:
        inside = [i for i in usable
                  if heading_deviation(robot, yaw, paths[i]) <= config.heading_limit_rad]
        usable = inside or usable  # never leave the robot without a candidate
    use_previous = previous if previous_age_s <= config.previous_max_age_s else None
    best, best_cost = usable[0], math.inf
    for rank_order, i in enumerate(usable):
        cost = (
            config.rank_weight * i
            + config.continuity_weight * continuity_cost(paths[i], use_previous, config.lookahead_points)
            + config.heading_weight * heading_deviation(robot, yaw, paths[i])
            + extra[i]
        )
        if cost < best_cost - 1e-12:
            best, best_cost = i, cost
    return best

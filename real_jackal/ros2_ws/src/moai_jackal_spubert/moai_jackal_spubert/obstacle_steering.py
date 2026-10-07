"""Continuous sideways steering away from obstacles that lie in the robot's way.

Pure functions (no ROS). The tracker adds the returned turn rate to its path-following
turn rate, so the robot curves around an obstacle from a distance instead of holding
straight until the path planner is forced to swerve.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class SteeringConfig:
    lookahead_m: float = 3.0       # obstacles farther than this are ignored
    min_x_m: float = 0.20          # ignore points this close to the sensor (self-hits)
    half_width_m: float = 0.75     # strip the body sweeps plus margin; corridor walls lie outside
    gain: float = 0.50             # rad/s of turn at an obstacle touching the robot
    max_rate: float = 0.40         # rad/s cap
    centre_band_m: float = 0.10    # |centroid y| below this: side is ambiguous, use the preference

    @property
    def active(self) -> bool:
        return self.gain > 0.0 and self.max_rate > 0.0 and self.lookahead_m > self.min_x_m


@dataclass(frozen=True)
class Steering:
    rate: float            # rad/s, left positive; 0 when nothing is in the way
    nearest_x_m: float     # distance to the nearest obstacle point in the strip (inf if none)
    centroid_y_m: float    # lateral position of the obstacle (0 if none)
    side: int              # +1 steer left, -1 steer right, 0 none


def avoidance_steering(
    ranges: Iterable[float],
    angle_min: float,
    angle_increment: float,
    config: SteeringConfig,
    prefer_side: int = 0,
) -> Steering:
    """Turn rate that moves the robot away from the obstacle ahead, if there is one.

    ``prefer_side`` (+1 left / -1 right / 0 none) breaks the tie for an obstacle dead ahead.
    """
    none = Steering(0.0, math.inf, 0.0, 0)
    if not config.active:
        return none
    near = math.inf
    weight_sum = 0.0
    y_weighted = 0.0
    angle = float(angle_min)
    for value in ranges:
        r = float(value)
        if math.isfinite(r) and r > 0.0:
            px, py = r * math.cos(angle), r * math.sin(angle)
            if config.min_x_m < px < config.lookahead_m and abs(py) <= config.half_width_m:
                w = 1.0 / px
                weight_sum += w
                y_weighted += w * py
                near = min(near, px)
        angle += float(angle_increment)
    if weight_sum <= 0.0:
        return none
    centroid = y_weighted / weight_sum
    if abs(centroid) >= config.centre_band_m:
        side = -1 if centroid > 0.0 else 1
    else:
        side = 1 if prefer_side >= 0 else -1
    urgency = min(max((config.lookahead_m - near) / max(config.lookahead_m - 0.3, 1e-6), 0.0), 1.0)
    # an obstacle well off to one side needs less of a turn than one right on the axis
    centred = 1.0 - 0.6 * min(abs(centroid) / config.half_width_m, 1.0)
    rate = min(config.gain * urgency * centred, config.max_rate)
    return Steering(side * rate, near, centroid, side)


def combine_with_path_turn(path_turn: float, steering: Steering) -> float:
    """Add the avoidance turn without doubling a turn the path already makes away from it."""
    if steering.side == 0:
        return path_turn
    want = abs(steering.rate)
    already = max(0.0, steering.side * path_turn)   # path turn already going the right way
    return path_turn + steering.side * max(0.0, want - already)


@dataclass(frozen=True)
class GapConfig:
    """Follow-the-gap: pick the side of an obstacle that has room to pass and head there."""
    horizon_m: float = 3.5        # look this far ahead for obstacles and walls
    block_horizon_m: float = 3.0  # an obstacle on the straight line closer than this triggers the search
    min_x_m: float = 0.20
    view_half_m: float = 2.5      # lateral extent considered
    straight_half_m: float = 0.60  # the straight line must be free this wide (body 0.215 + room to pass)
    min_gap_m: float = 0.85       # narrowest opening worth driving through
    edge_margin_m: float = 0.65   # keep this far from both edges of the opening when it is wide enough
    bin_m: float = 0.05
    switch_penalty_m: float = 0.5  # reluctance to change to the other side of the obstacle

    @property
    def active(self) -> bool:
        return self.min_gap_m > 0.0 and self.horizon_m > self.min_x_m


@dataclass(frozen=True)
class GapResult:
    blocked: bool                  # something is on the straight line ahead
    heading: Optional[float]       # rad, left positive: where to head; None if blocked with no opening
    side: int                      # +1 gap lies to the left, -1 right, 0 none
    target_y_m: float
    gap_width_m: float
    near_x_m: float


def gap_heading(
    ranges: Iterable[float],
    angle_min: float,
    angle_increment: float,
    config: GapConfig,
    prefer_side: int = 0,
) -> GapResult:
    free = GapResult(False, None, 0, 0.0, 0.0, math.inf)
    if not config.active:
        return free
    pts = []
    angle = float(angle_min)
    for value in ranges:
        r = float(value)
        if math.isfinite(r) and r > 0.0:
            px, py = r * math.cos(angle), r * math.sin(angle)
            if config.min_x_m < px < config.horizon_m and abs(py) <= config.view_half_m:
                pts.append((px, py))
        angle += float(angle_increment)
    near = min((x for x, y in pts if abs(y) <= config.straight_half_m and x < config.block_horizon_m),
               default=math.inf)
    if not math.isfinite(near):
        return free
    count = int(round(2.0 * config.view_half_m / config.bin_m))
    occupied = [False] * count
    for _, y in pts:
        occupied[min(max(int((y + config.view_half_m) / config.bin_m), 0), count - 1)] = True
    openings = []
    start = None
    for i in range(count + 1):
        if i < count and not occupied[i]:
            if start is None:
                start = i
        elif start is not None:
            openings.append((start * config.bin_m - config.view_half_m, i * config.bin_m - config.view_half_m))
            start = None
    best = None
    for lo, hi in openings:
        width = hi - lo
        # space beyond the outermost wall or obstacle is merely unseen, not a way through:
        # an opening must be closed on both sides inside the view
        if width + config.bin_m < config.min_gap_m or lo <= -config.view_half_m + 1e-9 or hi >= config.view_half_m - 1e-9:
            continue
        lo2, hi2 = lo + config.edge_margin_m, hi - config.edge_margin_m
        y_t = (lo + hi) / 2.0 if lo2 > hi2 else min(max(0.0, lo2), hi2)
        side = 1 if y_t > 0 else -1 if y_t < 0 else (prefer_side or 1)
        cost = abs(y_t) + (config.switch_penalty_m if prefer_side and side != prefer_side else 0.0)
        if best is None or cost < best[0]:
            best = (cost, y_t, width, side)
    if best is None:
        return GapResult(True, None, 0, 0.0, 0.0, near)
    _, y_t, width, side = best
    aim_x = min(max(near * 0.7, 1.0), 2.5)
    return GapResult(True, math.atan2(y_t, aim_x), side, y_t, width, near)

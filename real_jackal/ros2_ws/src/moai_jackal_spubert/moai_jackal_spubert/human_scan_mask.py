"""Treat tracked pedestrians as people, not as walls, in the scan-based obstacle slow-down and stop rules."""
from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

XY = Tuple[float, float]


def mask_people_in_ranges(
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    people: Iterable[XY],
    radius_m: float,
) -> List[float]:
    """Ranges with the beams that end within ``radius_m`` of a person replaced by +inf.

    Used only for the obstacle stop / slow-down distances; people are handled by their own rules
    (``human_stop_distance`` and ``human_slow_distance``). The sector hard stop keeps the raw scan.
    """
    people = [(float(x), float(y)) for x, y in people if math.isfinite(x) and math.isfinite(y)]
    if radius_m <= 0.0 or not people:
        return list(ranges)
    out: List[float] = []
    for index, value in enumerate(ranges):
        value = float(value)
        if math.isfinite(value):
            angle = angle_min + index * angle_increment
            x, y = value * math.cos(angle), value * math.sin(angle)
            if any(math.hypot(x - px, y - py) < radius_m for px, py in people):
                value = math.inf
        out.append(value)
    return out


def nearest_person_ahead(people: Iterable[XY], half_angle_rad: float) -> float:
    """Distance to the nearest person inside the forward sector (inf if none)."""
    best = math.inf
    for x, y in people:
        if x > 0.0 and abs(math.atan2(y, x)) <= half_angle_rad:
            best = min(best, math.hypot(x, y))
    return best


def human_slow_scale(distance_m: float, stop_m: float, slow_m: float) -> float:
    """1 beyond ``slow_m``, 0 at ``stop_m``, linear in between; 1 when the rule is off (slow_m <= stop_m)."""
    if slow_m <= stop_m or distance_m >= slow_m:
        return 1.0
    return min(max((distance_m - stop_m) / (slow_m - stop_m), 0.0), 1.0)


class PeopleHold:
    """Keep the last people for ``hold_s`` when the tracks vanish while a person was close.

    A person who is lost inside ``range_m`` (left the camera view, a merged or split track) must not make the stop and
    slow-down rules disappear at the very moment he is closest. Only an empty track list is bridged; any new track
    replaces the held people at once. 0 = off.
    """

    def __init__(self, hold_s: float, range_m: float) -> None:
        self.hold_s = float(hold_s)
        self.range_m = float(range_m)
        self._points: List[XY] = []
        self._until = -math.inf

    def update(self, now_s: float, points: Sequence[XY]) -> List[XY]:
        points = list(points)
        if points:
            self._points = points
            close = min(math.hypot(x, y) for x, y in points) <= self.range_m
            self._until = now_s + self.hold_s if close and self.hold_s > 0.0 else -math.inf
            return points
        if now_s <= self._until:
            return list(self._points)
        self._points = []
        return []

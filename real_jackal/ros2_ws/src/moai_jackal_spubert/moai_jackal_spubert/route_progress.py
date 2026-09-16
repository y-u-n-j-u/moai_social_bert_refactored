"""Measure progress along one ordered route without jumping between its arms.

This is a routing check, not a substitute for footprint or pedestrian checks.
All coordinates must come from the same frame and route snapshot.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import math
from typing import Sequence, Tuple

XY = Tuple[float, float]
_EPS = 1e-9


@dataclass(frozen=True)
class RouteProjection:
    arc_m: float
    distance_m: float
    point: XY
    segment: int


@dataclass(frozen=True)
class RouteProgress:
    error: str
    arc_progress_m: float
    effective_progress_m: float
    start_distance_m: float
    end_distance_m: float
    maximum_distance_m: float


class RouteProgressReference:
    """Associate a candidate with a bounded, continuous part of a Nav2 route.

    A 5 cm ambiguity band (at the default 10 cm map resolution) rejects
    competing distant route branches instead of choosing one due to mm noise.
    Ordinary neighboring corner segments remain eligible. Sequential matching
    permits projection transitions around ordinary corners, but bounds them
    by both route arc/chord geometry and cumulative physical candidate travel.
    """

    def __init__(
        self, route: Sequence[XY], current: XY, *,
        max_distance_m: float = 1.5, ambiguity_distance_m: float = 0.05,
    ):
        self.max_distance_m = float(max_distance_m)
        self.ambiguity_distance_m = float(ambiguity_distance_m)
        if (not math.isfinite(self.max_distance_m) or self.max_distance_m <= 0
                or not math.isfinite(self.ambiguity_distance_m)
                or self.ambiguity_distance_m < 0):
            raise ValueError("invalid_route_progress_parameters")
        self.current = float(current[0]), float(current[1])
        points = [(float(x), float(y)) for x, y in route]
        if not all(math.isfinite(v) for p in [self.current, *points] for v in p):
            raise ValueError("nonfinite_progress_route")
        self.segments = []
        self.starts, self.ends = [], []
        total = 0.0
        for index, (start, end) in enumerate(zip(points, points[1:])):
            length = math.dist(start, end)
            if length <= _EPS:
                continue
            self.segments.append((start, end, length, index))
            self.starts.append(total)
            total += length
            self.ends.append(total)
        if not self.segments:
            raise ValueError("empty_progress_route")
        self.length_m = total
        self.anchor = self._choose(self._projections(self.current, 0.0, total))
        if self.anchor.distance_m > self.max_distance_m:
            raise ValueError("progress_route_too_far")

    def _projections(self, point: XY, low: float, high: float):
        first = bisect_left(self.ends, low - _EPS)
        last = bisect_right(self.starts, high + _EPS)
        projections = []
        for i in range(first, last):
            start, end, length, original_index = self.segments[i]
            dx, dy = end[0] - start[0], end[1] - start[1]
            fraction = max(0.0, min(1.0, (
                (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
            ) / (length * length)))
            arc = self.starts[i] + fraction * length
            # Do not manufacture a projection by clamping it to the search
            # window. A projection outside this branch window is ineligible.
            if arc < low - _EPS or arc > high + _EPS:
                continue
            projected = start[0] + fraction * dx, start[1] + fraction * dy
            projections.append(RouteProjection(
                arc, math.dist(point, projected), projected, original_index,
            ))
        return projections

    def _choose(self, projections, previous_arc=None):
        if not projections:
            raise ValueError("route_projection_discontinuous")
        nearest = min(p.distance_m for p in projections)
        contenders = [p for p in projections
                      if p.distance_m <= nearest + self.ambiguity_distance_m + _EPS]
        best = min(projections, key=lambda p: (
            p.distance_m,
            abs(p.arc_m - previous_arc) if previous_arc is not None else p.arc_m,
        ))
        for other in contenders:
            # A normal 90-degree corner has arc/chord <= sqrt(2). A nearby
            # crossing or another U arm has a much larger arc separation.
            if abs(other.arc_m - best.arc_m) > (
                2.0 * math.dist(other.point, best.point)
                + 2.0 * self.ambiguity_distance_m + _EPS
            ):
                raise ValueError("ambiguous_progress_route")
        return best

    def measure(self, path: Sequence[XY], *, sample_spacing_m: float = 0.05) -> RouteProgress:
        spacing = float(sample_spacing_m)
        if not math.isfinite(spacing) or spacing <= 0:
            raise ValueError("invalid_route_sample_spacing")
        previous_point, projection = self.current, self.anchor
        maximum_distance = projection.distance_m
        traveled = 0.0
        error = ""
        try:
            for raw_end in path:
                end = float(raw_end[0]), float(raw_end[1])
                if not all(math.isfinite(v) for v in end):
                    raise ValueError("nonfinite_route_candidate")
                start = previous_point
                steps = max(1, int(math.ceil(math.dist(start, end) / spacing)))
                for step in range(1, steps + 1):
                    point = (start[0] + (end[0] - start[0]) * step / steps,
                             start[1] + (end[1] - start[1]) * step / steps)
                    movement = math.dist(previous_point, point)
                    if movement <= _EPS:
                        previous_point = point
                        continue
                    traveled += movement
                    # Off-route motion can switch its nearest projection from
                    # one corner leg to the next with tiny physical movement.
                    # Include the association offsets in this search window;
                    # then validate the transition geometry and total travel.
                    allowance = 2.0 * (movement + projection.distance_m + self.max_distance_m)
                    candidates = self._projections(
                        point, max(0.0, projection.arc_m - allowance),
                        min(self.length_m, projection.arc_m + allowance),
                    )
                    next_projection = self._choose(candidates, projection.arc_m)
                    arc_change = abs(next_projection.arc_m - projection.arc_m)
                    chord = math.dist(next_projection.point, projection.point)
                    if arc_change > 2.0 * (chord + self.ambiguity_distance_m) + _EPS:
                        raise ValueError("route_projection_discontinuous")
                    total_arc_change = abs(next_projection.arc_m - self.anchor.arc_m)
                    if total_arc_change > 2.0 * (
                        traveled + self.anchor.distance_m + next_projection.distance_m
                    ) + _EPS:
                        raise ValueError("route_projection_discontinuous")
                    projection = next_projection
                    maximum_distance = max(maximum_distance, projection.distance_m)
                    if projection.distance_m > self.max_distance_m + _EPS:
                        raise ValueError("route_deviation_exceeded")
                    previous_point = point
        except ValueError as exc:
            error = str(exc)
        arc_progress = projection.arc_m - self.anchor.arc_m
        # Decrease in (remaining route arc + distance needed to rejoin route).
        # Adjacent corner projections can switch with very little motion.
        # Never credit more than net physical displacement, preserving the
        # minimum endpoint movement implied by the old final-goal dot gate.
        # Path length alone would incorrectly credit an out-and-back wobble.
        effective = min(
            arc_progress + self.anchor.distance_m - projection.distance_m,
            math.dist(self.current, previous_point),
        )
        return RouteProgress(error, arc_progress, effective, self.anchor.distance_m,
                             projection.distance_m, maximum_distance)

    def diagnostics(self):
        return {"route_length_m": self.length_m, "anchor_arc_m": self.anchor.arc_m,
                "anchor_segment": self.anchor.segment,
                "anchor_distance_m": self.anchor.distance_m,
                "max_distance_m": self.max_distance_m,
                "ambiguity_distance_m": self.ambiguity_distance_m}

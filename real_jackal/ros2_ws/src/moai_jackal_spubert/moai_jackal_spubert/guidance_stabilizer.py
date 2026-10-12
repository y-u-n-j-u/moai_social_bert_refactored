"""Keep the guidance point (GP) from jumping back and sideways between planning cycles: pure functions."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

XY = Tuple[float, float]


@dataclass(frozen=True)
class StabilizerConfig:
    max_step_m: float = 0.8       # the GP may move at most this far between two cycles
    backtrack_m: float = 0.3      # ...and at most this far backwards along the route
    lateral_reset_m: float = 2.5  # the previous GP this far from the route: the route changed, start over

    @property
    def active(self) -> bool:
        return self.max_step_m > 0.0


def route_s(route: Sequence[XY], point: XY) -> Tuple[float, float]:
    """(arc length from the route start of the projection of ``point``, distance to the route)."""
    best_d, best_s, run = math.inf, 0.0, 0.0
    for (x1, y1), (x2, y2) in zip(route, route[1:]):
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length <= 1e-12:
            continue
        f = max(0.0, min(1.0, ((point[0] - x1) * dx + (point[1] - y1) * dy) / (length * length)))
        px, py = x1 + f * dx, y1 + f * dy
        d = math.hypot(point[0] - px, point[1] - py)
        if d < best_d:
            best_d, best_s = d, run + f * length
        run += length
    return best_s, best_d


def point_at_s(route: Sequence[XY], s: float) -> XY:
    run = 0.0
    for (x1, y1), (x2, y2) in zip(route, route[1:]):
        length = math.hypot(x2 - x1, y2 - y1)
        if length > 1e-12 and run + length >= s:
            f = (s - run) / length
            return x1 + f * (x2 - x1), y1 + f * (y2 - y1)
        run += length
    return route[-1]


def stabilize(
    prev: Optional[XY],
    new: XY,
    route: Sequence[XY],
    is_safe: Callable[[XY], bool],
    config: StabilizerConfig,
) -> XY:
    """``new`` is always safe (it came from the safety search); the result is ``new`` unless a calmer, safe point exists."""
    if not config.active or prev is None or len(route) < 2:
        return new
    s_prev, d_prev = route_s(route, prev)
    if d_prev > config.lateral_reset_m:
        return new
    s_new, _ = route_s(route, new)
    s = min(max(s_new, s_prev - config.backtrack_m), s_prev + config.max_step_m)
    candidate = point_at_s(route, s)
    # a lateral jump (the route moved sideways) is also spread over several cycles
    step = math.hypot(candidate[0] - prev[0], candidate[1] - prev[1])
    if step > config.max_step_m:
        k = config.max_step_m / step
        candidate = (prev[0] + k * (candidate[0] - prev[0]), prev[1] + k * (candidate[1] - prev[1]))
    if is_safe(candidate):
        return candidate
    # The calm point is blocked: take the farthest safe point between it and the previous GP before giving up to
    # ``new``, which can be several metres away from where the GP was a moment ago.
    for fraction in (0.75, 0.5, 0.25):
        middle = (prev[0] + fraction * (candidate[0] - prev[0]), prev[1] + fraction * (candidate[1] - prev[1]))
        if is_safe(middle):
            return middle
    return new

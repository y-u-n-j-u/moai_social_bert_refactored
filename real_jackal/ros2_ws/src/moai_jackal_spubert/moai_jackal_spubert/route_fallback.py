"""Pure geometry for the global-path fallback (no ROS, numpy or model imports)."""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

XY = Tuple[float, float]


def route_prefix_points(
    route: Sequence[XY],
    current: XY,
    count: int,
    spacing: float,
    max_route_offset: float = 1.0,
) -> Optional[List[XY]]:
    """Up to ``count`` points, ``spacing`` apart, walking ahead along ``route``.

    Starts from the point on the route nearest to ``current``. Returns None
    when the route is unusable or the robot is farther than
    ``max_route_offset`` from it (a stale route must not steer the robot
    across open space). The caller decides whether the prefix is safe; this
    is geometry only. Truncated at the route end.
    """
    pts = [(float(x), float(y)) for x, y in route]
    if len(pts) < 2 or count < 1 or not spacing > 0.0:
        return None
    if not all(math.isfinite(v) for p in pts for v in p) or not all(
        math.isfinite(float(v)) for v in current
    ):
        return None
    cx, cy = float(current[0]), float(current[1])
    best_d2, best_i, best_t = math.inf, 0, 0.0
    for i, (a, b) in enumerate(zip(pts, pts[1:])):
        dx, dy = b[0] - a[0], b[1] - a[1]
        seg2 = dx * dx + dy * dy
        if seg2 <= 1e-12:
            continue
        t = max(0.0, min(1.0, ((cx - a[0]) * dx + (cy - a[1]) * dy) / seg2))
        d2 = (cx - (a[0] + t * dx)) ** 2 + (cy - (a[1] + t * dy)) ** 2
        if d2 < best_d2:
            best_d2, best_i, best_t = d2, i, t
    if not math.isfinite(best_d2) or math.sqrt(best_d2) > max_route_offset:
        return None
    a, b = pts[best_i], pts[best_i + 1]
    start = (a[0] + best_t * (b[0] - a[0]), a[1] + best_t * (b[1] - a[1]))
    tail = [start, *pts[best_i + 1 :]]
    result: List[XY] = []
    target = spacing
    walked = 0.0
    for p, q in zip(tail, tail[1:]):
        seg = math.hypot(q[0] - p[0], q[1] - p[1])
        if seg <= 1e-12:
            continue
        while target <= walked + seg + 1e-9 and len(result) < count:
            r = (target - walked) / seg
            result.append((p[0] + r * (q[0] - p[0]), p[1] + r * (q[1] - p[1])))
            target += spacing
        walked += seg
        if len(result) >= count:
            break
    return result or None


def shift_path_laterally(
    points: Sequence[XY],
    start: XY,
    offset: float,
    taper_m: float = 0.4,
) -> Optional[List[XY]]:
    """Slide ``points`` sideways by ``offset`` metres (positive = left of travel).

    The shift ramps in linearly over the first ``taper_m`` of arc length from
    ``start`` so the path stays attached to the robot instead of jumping
    sideways. Each point moves along the normal of the local path direction.
    Returns None for unusable input; offset 0 returns the points unchanged.
    """
    pts = [(float(x), float(y)) for x, y in points]
    if not pts or not math.isfinite(float(offset)):
        return None
    if offset == 0.0:
        return list(pts)
    chain = [(float(start[0]), float(start[1])), *pts]
    if not all(math.isfinite(v) for p in chain for v in p):
        return None
    result: List[XY] = []
    walked = 0.0
    for index, point in enumerate(pts):
        previous = chain[index]
        walked += math.hypot(point[0] - previous[0], point[1] - previous[1])
        after = pts[index + 1] if index + 1 < len(pts) else point
        before = chain[index]
        dx, dy = after[0] - before[0], after[1] - before[1]
        norm = math.hypot(dx, dy)
        if norm <= 1e-9:
            return None
        ramp = 1.0 if taper_m <= 0.0 else min(walked / taper_m, 1.0)
        shift = float(offset) * ramp
        result.append((point[0] - dy / norm * shift, point[1] + dx / norm * shift))
    return result


def lateral_offsets(maximum: float, step: float) -> List[float]:
    """0, +step, -step, +2*step, ... up to ``maximum`` (nearest-to-centre first)."""
    if not (maximum >= 0.0 and step > 0.0):
        return [0.0]
    count = int(round(maximum / step + 1e-9))
    result = [0.0]
    for k in range(1, count + 1):
        result.extend((k * step, -k * step))
    return result

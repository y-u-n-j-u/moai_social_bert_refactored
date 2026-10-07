"""Rate-limited following of a frame transform (no ROS, numpy or model imports).

The Nav2 route and the goal are planned in the map frame and handed to the controller in
odom. AMCL keeps correcting map->odom while the robot drives (field data 2026-10-07:
>5 cm in 14% of the updates, excursions of 0.2-0.7 m), and every correction used to shift
the whole route sideways in odom, so the tracker kept steering after a path that moved.
These helpers let the applied transform follow the latest one at a bounded speed.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

Pose2 = Tuple[float, float, float]  # x, y, yaw


def wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def slew_transform(
    applied: Optional[Pose2],
    latest: Pose2,
    dt: float,
    max_linear_m_s: float,
    max_angular_rad_s: float,
) -> Pose2:
    """Move ``applied`` toward ``latest`` by at most the allowed distance and angle.

    ``applied=None`` (first use) or a non-positive rate returns ``latest`` unchanged.
    Non-finite input returns the previous applied transform (or latest if none).
    """
    if not all(math.isfinite(v) for v in latest):
        return applied if applied is not None else latest
    if applied is None or max_linear_m_s <= 0.0 or max_angular_rad_s <= 0.0:
        return latest
    step = max(float(dt), 0.0)
    dx, dy = latest[0] - applied[0], latest[1] - applied[1]
    distance = math.hypot(dx, dy)
    allowed = max_linear_m_s * step
    if distance <= allowed or distance < 1e-12:
        x, y = latest[0], latest[1]
    else:
        scale = allowed / distance
        x, y = applied[0] + dx * scale, applied[1] + dy * scale
    dyaw = wrap_angle(latest[2] - applied[2])
    limit = max_angular_rad_s * step
    yaw = latest[2] if abs(dyaw) <= limit else wrap_angle(applied[2] + math.copysign(limit, dyaw))
    return (x, y, yaw)


def is_large_jump(applied: Pose2, latest: Pose2, linear_m: float) -> bool:
    return math.hypot(latest[0] - applied[0], latest[1] - applied[1]) > linear_m

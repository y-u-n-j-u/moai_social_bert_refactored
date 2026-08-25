from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple


XY = Tuple[float, float]


def same_frame_id(first: str, second: str) -> bool:
    """Compare ROS frame IDs while tolerating a legacy leading slash."""
    return str(first).strip().lstrip("/") == str(second).strip().lstrip("/")


def clamp(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(value), float(upper)))


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(
        2.0 * (float(w) * float(z) + float(x) * float(y)),
        1.0 - 2.0 * (float(y) * float(y) + float(z) * float(z)),
    )


def transform_xy(point: XY, translation: XY, yaw: float) -> XY:
    """Apply a 2-D rigid transform from a source frame into a target frame."""
    ct = math.cos(float(yaw))
    st = math.sin(float(yaw))
    return (
        float(translation[0]) + ct * float(point[0]) - st * float(point[1]),
        float(translation[1]) + st * float(point[0]) + ct * float(point[1]),
    )


def interpolate_polyline(points: Sequence[XY], max_spacing: float) -> list[XY]:
    spacing = float(max_spacing)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("max_spacing must be positive and finite")
    normalized = [(float(x), float(y)) for x, y in points]
    if not normalized:
        return []
    output = [normalized[0]]
    for start, end in zip(normalized, normalized[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.hypot(dx, dy)
        steps = max(1, int(math.ceil(distance / spacing)))
        output.extend(
            (
                start[0] + dx * index / steps,
                start[1] + dy * index / steps,
            )
            for index in range(1, steps + 1)
        )
    return output


def estimate_velocity(
    history: Sequence[XY],
    sample_dt: float,
    maximum_speed: float = 2.5,
) -> XY:
    """Estimate a bounded velocity from the oldest and newest useful samples."""
    if len(history) < 2 or sample_dt <= 0.0:
        return 0.0, 0.0
    latest = history[-1]
    earliest_index = max(0, len(history) - 4)
    earliest = history[earliest_index]
    elapsed = max((len(history) - 1 - earliest_index) * float(sample_dt), sample_dt)
    vx = (float(latest[0]) - float(earliest[0])) / elapsed
    vy = (float(latest[1]) - float(earliest[1])) / elapsed
    speed = math.hypot(vx, vy)
    limit = max(float(maximum_speed), 0.0)
    if speed > limit > 0.0:
        scale = limit / speed
        vx *= scale
        vy *= scale
    return vx, vy


@dataclass(frozen=True)
class CandidateCheck:
    valid: bool
    reason: str
    footprint_collision_count: int
    maximum_step_m: float
    goal_progress_m: float
    minimum_human_distance_m: float
    minimum_human_clearance_m: float


def validate_candidate_path(
    *,
    current: XY,
    path: Sequence[XY],
    final_goal: XY,
    map_provider,
    footprint_radius: float,
    prediction_dt: float,
    maximum_model_speed: float,
    maximum_step_ratio: float,
    minimum_goal_progress: float,
    human_histories: Mapping[int, Sequence[XY]],
    human_sample_dt: float,
    minimum_human_center_distance: float,
    human_radius: float,
    human_safety_margin: float,
) -> CandidateCheck:
    """Apply the runtime checks that are independent of ROS message types."""
    if not path:
        return CandidateCheck(False, "empty_path", 0, 0.0, 0.0, math.inf, math.inf)
    if not all(math.isfinite(x) and math.isfinite(y) for x, y in path):
        return CandidateCheck(False, "nonfinite_path", 0, math.inf, 0.0, math.inf, math.inf)

    spacing = max(float(map_provider.resolution) * 0.5, 0.02)
    swept = interpolate_polyline([current, *path], spacing)
    collision_count = sum(
        map_provider.path_collision_cost([point], radius=footprint_radius, weight=1.0) > 0.0
        for point in swept
    )

    previous = current
    steps = []
    for point in path:
        steps.append(math.hypot(float(point[0]) - previous[0], float(point[1]) - previous[1]))
        previous = float(point[0]), float(point[1])
    maximum_step = max(steps, default=0.0)
    allowed_step = (
        max(float(maximum_model_speed), 0.01)
        * max(float(prediction_dt), 0.01)
        * max(float(maximum_step_ratio), 1.0)
    )

    gx = float(final_goal[0]) - float(current[0])
    gy = float(final_goal[1]) - float(current[1])
    goal_distance = max(math.hypot(gx, gy), 1e-6)
    endpoint = path[-1]
    progress = (
        (float(endpoint[0]) - float(current[0])) * gx
        + (float(endpoint[1]) - float(current[1])) * gy
    ) / goal_distance

    minimum_center = math.inf
    required_human_distance = max(
        float(minimum_human_center_distance),
        float(footprint_radius) + float(human_radius) + float(human_safety_margin),
    )
    for step_index, robot_point in enumerate(path, start=1):
        horizon = step_index * float(prediction_dt)
        for history in human_histories.values():
            if not history:
                continue
            vx, vy = estimate_velocity(history, human_sample_dt)
            hx = float(history[-1][0]) + vx * horizon
            hy = float(history[-1][1]) + vy * horizon
            minimum_center = min(
                minimum_center,
                math.hypot(float(robot_point[0]) - hx, float(robot_point[1]) - hy),
            )
    minimum_clearance = (
        minimum_center - required_human_distance
        if math.isfinite(minimum_center)
        else math.inf
    )

    if collision_count:
        reason = "robot_footprint_collision"
    elif maximum_step > allowed_step:
        reason = "kinematic_jump"
    elif progress < float(minimum_goal_progress):
        reason = "insufficient_goal_progress"
    elif minimum_clearance < 0.0:
        reason = "predicted_human_clearance"
    else:
        reason = "valid"
    return CandidateCheck(
        valid=reason == "valid",
        reason=reason,
        footprint_collision_count=int(collision_count),
        maximum_step_m=float(maximum_step),
        goal_progress_m=float(progress),
        minimum_human_distance_m=float(minimum_center),
        minimum_human_clearance_m=float(minimum_clearance),
    )


def closest_path_index(points: Sequence[XY], current: XY) -> int:
    if not points:
        return 0
    return min(
        range(len(points)),
        key=lambda index: math.hypot(
            float(points[index][0]) - float(current[0]),
            float(points[index][1]) - float(current[1]),
        ),
    )


def lookahead_point(points: Sequence[XY], current: XY, distance: float) -> Optional[XY]:
    if not points:
        return None
    start = closest_path_index(points, current)
    previous = (float(current[0]), float(current[1]))
    remaining = max(float(distance), 0.0)
    for point in points[start:]:
        target = float(point[0]), float(point[1])
        segment = math.hypot(target[0] - previous[0], target[1] - previous[1])
        if segment >= remaining and segment > 1e-9:
            ratio = remaining / segment
            return (
                previous[0] + ratio * (target[0] - previous[0]),
                previous[1] + ratio * (target[1] - previous[1]),
            )
        remaining -= segment
        previous = target
    return float(points[-1][0]), float(points[-1][1])


def finite_ranges_in_sector(
    ranges: Iterable[float],
    angle_min: float,
    angle_increment: float,
    half_angle: float,
) -> list[float]:
    selected = []
    angle = float(angle_min)
    limit = abs(float(half_angle))
    for value in ranges:
        if abs(normalize_angle(angle)) <= limit and math.isfinite(float(value)):
            selected.append(float(value))
        angle += float(angle_increment)
    return selected

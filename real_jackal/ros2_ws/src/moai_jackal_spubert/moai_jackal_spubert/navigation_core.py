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
    for history in human_histories.values():
        if not history:
            continue
        vx, vy = estimate_velocity(history, human_sample_dt)
        human_x, human_y = float(history[-1][0]), float(history[-1][1])
        previous = float(current[0]), float(current[1])
        dt = float(prediction_dt)
        for robot_point in path:
            # The robot follows a linear segment during this interval, while
            # the human follows the same constant-velocity prediction as before.
            # Minimize their relative distance over the whole interval, including
            # t=0; checking only the future waypoints can miss a crossing.
            relative_x, relative_y = previous[0] - human_x, previous[1] - human_y
            delta_x = float(robot_point[0]) - previous[0] - vx * dt
            delta_y = float(robot_point[1]) - previous[1] - vy * dt
            relative_speed_sq = delta_x * delta_x + delta_y * delta_y
            fraction = (
                clamp(-(relative_x * delta_x + relative_y * delta_y) / relative_speed_sq, 0.0, 1.0)
                if relative_speed_sq > 1e-18 else 0.0
            )
            minimum_center = min(minimum_center, math.hypot(
                relative_x + fraction * delta_x,
                relative_y + fraction * delta_y,
            ))
            previous = float(robot_point[0]), float(robot_point[1])
            human_x += vx * dt
            human_y += vy * dt
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


def lookahead_point(
    points: Sequence[XY], current: XY, distance: float, *, require_forward_progress: bool = False,
) -> Optional[XY]:
    """Project onto the polyline, then walk forward by the requested arc length.

    Counting the robot-to-nearest-vertex distance can first walk backwards on a
    sparse path. Projection also keeps the target continuous when the nearest
    vertex changes. Equal-distance projections prefer the earlier segment.
    Execution can require remaining forward arc length so an exhausted short
    plan never commands a turn back toward its endpoint.
    """
    if not points:
        return None
    normalized = [(float(x), float(y)) for x, y in points]
    current_x, current_y = float(current[0]), float(current[1])
    if (
        not all(math.isfinite(value) for point in normalized for value in point)
        or not all(math.isfinite(value) for value in (current_x, current_y, float(distance)))
    ):
        return None
    if len(normalized) == 1:
        return None if require_forward_progress else normalized[0]

    best_distance_sq = math.inf
    best_segment = 0
    projection = normalized[0]
    for index, (start, end) in enumerate(zip(normalized, normalized[1:])):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-18:
            continue
        fraction = clamp(
            ((current_x - start[0]) * dx + (current_y - start[1]) * dy) / length_sq,
            0.0,
            1.0,
        )
        candidate = start[0] + fraction * dx, start[1] + fraction * dy
        distance_sq = (current_x - candidate[0]) ** 2 + (current_y - candidate[1]) ** 2
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_segment = index
            projection = candidate
    if not math.isfinite(best_distance_sq):
        return None if require_forward_progress else normalized[-1]

    if require_forward_progress:
        tail = [projection, *normalized[best_segment + 1 :]]
        remaining_arc = sum(
            math.hypot(end[0] - start[0], end[1] - start[1])
            for start, end in zip(tail, tail[1:])
        )
        if remaining_arc <= 1e-9:
            return None

    previous = projection
    remaining = max(float(distance), 0.0)
    for target in normalized[best_segment + 1 :]:
        segment = math.hypot(target[0] - previous[0], target[1] - previous[1])
        if segment >= remaining and segment > 1e-9:
            ratio = remaining / segment
            return (
                previous[0] + ratio * (target[0] - previous[0]),
                previous[1] + ratio * (target[1] - previous[1]),
            )
        remaining -= segment
        previous = target
    return normalized[-1]


def goal_approach_speed_limit(
    goal_distance: float,
    goal_tolerance: float,
    slow_distance: float,
    maximum_speed: float,
    minimum_speed: float,
) -> float:
    """Cap approach speed without asymptotically stopping outside tolerance.

    This is an upper bound, not a minimum command: heading and obstacle checks
    can still reduce the actual requested speed to zero.
    """
    if float(goal_distance) <= float(goal_tolerance):
        return 0.0
    maximum = max(float(maximum_speed), 0.0)
    if float(slow_distance) <= float(goal_tolerance):
        return maximum
    fraction = clamp(
        (float(goal_distance) - float(goal_tolerance))
        / (float(slow_distance) - float(goal_tolerance)),
        0.0,
        1.0,
    )
    minimum = clamp(minimum_speed, 0.0, maximum)
    return minimum + (maximum - minimum) * fraction


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


def nearest_range_in_sector(
    ranges: Iterable[float], angle_min: float, angle_increment: float, half_angle: float,
) -> Tuple[float, Optional[float]]:
    """Return the existing sector minimum and its bearing for diagnostics.

    Filtering and angle accumulation match ``finite_ranges_in_sector``. With
    no finite return the distance stays infinite and the bearing is unknown.
    """
    minimum = math.inf
    bearing = None
    angle = float(angle_min)
    limit = abs(float(half_angle))
    for value in ranges:
        normalized = normalize_angle(angle)
        if abs(normalized) <= limit and math.isfinite(float(value)) and float(value) < minimum:
            minimum = float(value)
            bearing = normalized
        angle += float(angle_increment)
    return minimum, bearing

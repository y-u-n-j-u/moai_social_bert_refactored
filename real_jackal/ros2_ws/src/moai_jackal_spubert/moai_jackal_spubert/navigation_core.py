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
    # Diagnostic-only: sample centers, not the occupied cell/contact location.
    current_footprint_collision: Optional[bool] = None
    first_collision_point: Optional[XY] = None
    first_collision_sample_index: Optional[int] = None
    first_collision_segment_index: Optional[int] = None
    first_collision_distance_m: Optional[float] = None
    collision_sample_spacing_m: Optional[float] = None
    swept_sample_count: int = 0


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
    # Preserve the exact samples, radius, call order and full collision count.
    collision_count = 0
    first_collision_index = None
    current_collision = False
    for index, point in enumerate(swept):
        collision = map_provider.path_collision_cost(
            [point], radius=footprint_radius, weight=1.0
        ) > 0.0
        if index == 0:
            current_collision = bool(collision)
        if collision:
            collision_count += 1
            if first_collision_index is None:
                first_collision_index = index
    first_segment = None
    first_distance = None
    if first_collision_index is not None:
        first_distance = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                             for a, b in zip(swept[:first_collision_index],
                                             swept[1:first_collision_index + 1]))
        if first_collision_index > 0:
            end_sample = 0
            for segment, (a, b) in enumerate(zip([current, *path], path)):
                end_sample += max(1, int(math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / spacing)))
                if first_collision_index <= end_sample:
                    first_segment = segment
                    break

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
        current_footprint_collision=current_collision,
        first_collision_point=(None if first_collision_index is None else swept[first_collision_index]),
        first_collision_sample_index=first_collision_index,
        first_collision_segment_index=first_segment,
        first_collision_distance_m=first_distance,
        collision_sample_spacing_m=spacing,
        swept_sample_count=len(swept),
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


@dataclass(frozen=True)
class LookaheadResult:
    projection: XY
    target: XY
    segment_index: int
    target_segment_index: int
    progress_m: float
    target_progress_m: float
    path_length_m: float
    cross_track_error_m: float
    search_start_m: float
    search_end_m: float


def projected_lookahead(
    points: Sequence[XY],
    current: XY,
    distance: float,
    *,
    minimum_progress: float = 0.0,
    maximum_progress: Optional[float] = None,
) -> Optional[LookaheadResult]:
    """Project onto a polyline, then advance by arc length, not robot-to-vertex distance.

    Progress bounds constrain the projection only, not the lookahead target.
    Equal-distance projections select the earliest segment. Degenerate or
    nonfinite input has no usable tracking target and fails closed.
    """
    normalized = [(float(x), float(y)) for x, y in points]
    scalars = [*current, distance, minimum_progress]
    if maximum_progress is not None:
        scalars.append(maximum_progress)
    if not all(math.isfinite(value) for value in scalars):
        return None
    if not all(math.isfinite(value) for point in normalized for value in point):
        return None
    segments = []
    total = 0.0
    for index, (start, end) in enumerate(zip(normalized, normalized[1:])):
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        if length <= 1e-9:
            continue
        segments.append((index, start, end, total, length))
        total += length
    if not segments or not math.isfinite(total):
        return None

    lower = clamp(minimum_progress, 0.0, total)
    upper = total if maximum_progress is None else clamp(maximum_progress, lower, total)
    best = None
    best_distance = math.inf
    for index, start, end, offset, length in segments:
        if offset + length < lower or offset > upper:
            continue
        ux, uy = (end[0] - start[0]) / length, (end[1] - start[1]) / length
        along = (current[0] - start[0]) * ux + (current[1] - start[1]) * uy
        along = clamp(along, max(0.0, lower - offset), min(length, upper - offset))
        projection = (start[0] + ux * along, start[1] + uy * along)
        error = math.hypot(current[0] - projection[0], current[1] - projection[1])
        if error < best_distance - 1e-12:
            best_distance = error
            best = (index, projection, offset + along)
    if best is None:
        return None

    index, projection, progress = best
    target_progress = min(progress + max(float(distance), 0.0), total)
    for target_index, start, end, offset, length in segments:
        if target_progress <= offset + length + 1e-12:
            ratio = clamp((target_progress - offset) / length, 0.0, 1.0)
            target = (start[0] + ratio * (end[0] - start[0]),
                      start[1] + ratio * (end[1] - start[1]))
            return LookaheadResult(
                projection, target, index, target_index, progress, target_progress,
                total, best_distance, lower, upper,
            )
    return None


def lookahead_point(points: Sequence[XY], current: XY, distance: float) -> Optional[XY]:
    result = projected_lookahead(points, current, distance)
    return None if result is None else result.target


class PathProgressTracker:
    """Forward progress on one geometry; reset when geometry or frame changes.

    After initial acquisition, search at most one lookahead distance plus the
    robot's displacement ahead of prior progress. This bounds jumps to a later
    branch at intersections. It is a forward-path tracker, not reverse driving
    or recovery logic; a replacement path acquires its projection afresh.
    """

    def __init__(self) -> None:
        self.points: Tuple[XY, ...] = ()
        self.frame_id = ""
        self.version = 0
        self.progress: Optional[float] = None
        self._last_position: Optional[XY] = None

    def set_path(self, points: Sequence[XY], frame_id: str) -> bool:
        normalized = tuple((float(x), float(y)) for x, y in points)
        if normalized == self.points and frame_id == self.frame_id:
            return False
        self.points = normalized
        self.frame_id = frame_id
        self.version += 1
        self.progress = None
        self._last_position = None
        return True

    def update(self, current: XY, distance: float) -> Optional[LookaheadResult]:
        maximum = None
        if self.progress is not None and self._last_position is not None:
            displacement = math.hypot(current[0] - self._last_position[0],
                                      current[1] - self._last_position[1])
            maximum = self.progress + displacement + max(distance, 0.0)
        result = projected_lookahead(
            self.points, current, distance,
            minimum_progress=self.progress if self.progress is not None else 0.0,
            maximum_progress=maximum,
        )
        if result is not None:
            self.progress = result.progress_m
            self._last_position = current
        return result


def corridor_obstacle_distance(
    ranges: Iterable[float],
    angle_min: float,
    angle_increment: float,
    half_width: float,
) -> float:
    """Distance ahead to the nearest scan point inside the straight-ahead corridor.

    The corridor is the strip ``0 < x`` and ``|y| <= half_width`` in the scan frame
    (x forward, y left), i.e. what the robot body sweeps when driving straight.
    Points beside it (a box edge in a narrow passage) do not count. Returns inf
    when the corridor is clear.
    """
    nearest = math.inf
    angle = float(angle_min)
    width = abs(float(half_width))
    for value in ranges:
        r = float(value)
        if math.isfinite(r) and r > 0.0:
            a = normalize_angle(angle)
            x = r * math.cos(a)
            if x > 0.0 and abs(r * math.sin(a)) <= width and x < nearest:
                nearest = x
        angle += float(angle_increment)
    return nearest


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

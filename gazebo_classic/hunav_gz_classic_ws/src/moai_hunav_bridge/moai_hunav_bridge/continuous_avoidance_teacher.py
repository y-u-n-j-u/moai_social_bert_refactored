from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin
from typing import Iterable, Sequence


@dataclass(frozen=True)
class MovingHuman:
    agent_id: int
    x: float
    y: float
    vx: float
    vy: float


@dataclass(frozen=True)
class AvoidanceCommand:
    linear_speed: float
    angular_speed: float
    intervention: bool
    predicted_clearance: float
    agent_id: int | None


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _angle_difference(target: float, current: float) -> float:
    return (target - current + pi) % (2.0 * pi) - pi


def _unit_direction(direction: tuple[float, float], yaw: float) -> tuple[float, float]:
    length = hypot(direction[0], direction[1])
    if length <= 1e-6:
        return cos(yaw), sin(yaw)
    return direction[0] / length, direction[1] / length


def choose_avoidance_side(
    *,
    path_direction: tuple[float, float],
    robot_yaw: float,
    human_position: tuple[float, float],
    human_velocity: tuple[float, float],
    path_reference: tuple[float, float],
) -> int:
    """Choose the side behind a crossing human in the path frame."""
    direction = _unit_direction(path_direction, robot_yaw)
    normal = (-direction[1], direction[0])
    crossing_speed = (
        human_velocity[0] * normal[0]
        + human_velocity[1] * normal[1]
    )
    if abs(crossing_speed) > 0.05:
        return -1 if crossing_speed > 0.0 else 1

    human_lateral = (
        (human_position[0] - path_reference[0]) * normal[0]
        + (human_position[1] - path_reference[1]) * normal[1]
    )
    return -1 if human_lateral >= 0.0 else 1


def lane_tracking_command(
    *,
    robot_yaw: float,
    path_direction: tuple[float, float],
    current_lateral_offset: float,
    avoidance_side: int,
    lateral_offset: float = 1.5,
    lane_lookahead: float = 2.5,
    preferred_speed: float = 0.7,
    minimum_forward_speed: float = 0.35,
    maximum_forward_speed: float = 0.8,
    maximum_angular_speed: float = 1.0,
    heading_gain: float = 1.5,
) -> tuple[float, float]:
    """Track a temporary lane offset while retaining positive forward speed."""
    direction = _unit_direction(path_direction, robot_yaw)
    path_heading = atan2(direction[1], direction[0])
    side = -1 if avoidance_side < 0 else 1
    desired_lateral_offset = side * abs(lateral_offset)
    desired_heading = path_heading + atan2(
        desired_lateral_offset - current_lateral_offset,
        max(lane_lookahead, 1e-3),
    )
    angular_speed = _clamp(
        heading_gain * _angle_difference(desired_heading, robot_yaw),
        -maximum_angular_speed,
        maximum_angular_speed,
    )
    linear_speed = _clamp(
        preferred_speed,
        minimum_forward_speed,
        maximum_forward_speed,
    )
    return linear_speed, angular_speed


def _candidate_clearance(
    *,
    robot_position: tuple[float, float],
    robot_yaw: float,
    linear_speed: float,
    angular_speed: float,
    humans: Sequence[MovingHuman],
    horizon: float,
    step: float,
    steering_duration: float,
) -> tuple[float, int | None, float, float, float]:
    x, y = robot_position
    yaw = robot_yaw
    elapsed = 0.0
    minimum_clearance = float("inf")
    minimum_agent_id: int | None = None
    while elapsed <= horizon + 1e-9:
        for human in humans:
            hx = human.x + human.vx * elapsed
            hy = human.y + human.vy * elapsed
            clearance = hypot(hx - x, hy - y)
            if clearance < minimum_clearance:
                minimum_clearance = clearance
                minimum_agent_id = human.agent_id
        if elapsed >= horizon:
            break
        dt = min(step, horizon - elapsed)
        applied_turn = angular_speed if elapsed < steering_duration else 0.0
        yaw += applied_turn * dt
        x += linear_speed * cos(yaw) * dt
        y += linear_speed * sin(yaw) * dt
        elapsed += dt

    return minimum_clearance, minimum_agent_id, x, y, yaw


def select_continuous_avoidance_command(
    *,
    robot_position: tuple[float, float],
    robot_yaw: float,
    nominal_linear_speed: float,
    nominal_angular_speed: float,
    humans: Iterable[MovingHuman],
    path_direction: tuple[float, float],
    safety_distance: float = 1.2,
    trigger_distance: float = 1.35,
    horizon: float = 3.5,
    step: float = 0.1,
    activation_distance: float = 5.0,
    preferred_speed: float = 0.7,
    minimum_forward_speed: float = 0.35,
    maximum_forward_speed: float = 0.8,
    maximum_angular_speed: float = 1.0,
    steering_duration: float = 1.0,
    minimum_avoidance_angular_speed: float = 0.2,
) -> AvoidanceCommand:
    """Choose a positive-speed, time-aware command around moving humans."""
    human_list = tuple(
        human
        for human in humans
        if activation_distance <= 0.0
        or hypot(
            human.x - robot_position[0],
            human.y - robot_position[1],
        ) <= activation_distance
    )
    if not human_list:
        return AvoidanceCommand(
            nominal_linear_speed,
            nominal_angular_speed,
            False,
            float("inf"),
            None,
        )

    direction = _unit_direction(path_direction, robot_yaw)
    path_heading = atan2(direction[1], direction[0])
    reference_speed = _clamp(
        max(abs(nominal_linear_speed), preferred_speed),
        minimum_forward_speed,
        maximum_forward_speed,
    )
    reference_turn = _clamp(
        nominal_angular_speed,
        -maximum_angular_speed,
        maximum_angular_speed,
    )
    nominal = _candidate_clearance(
        robot_position=robot_position,
        robot_yaw=robot_yaw,
        linear_speed=reference_speed,
        angular_speed=reference_turn,
        humans=human_list,
        horizon=horizon,
        step=step,
        steering_duration=steering_duration,
    )
    if nominal[0] >= trigger_distance:
        return AvoidanceCommand(
            nominal_linear_speed,
            nominal_angular_speed,
            False,
            nominal[0],
            nominal[1],
        )

    speed_candidates = sorted(
        {
            reference_speed,
            max(minimum_forward_speed, reference_speed * 0.9),
            max(minimum_forward_speed, reference_speed * 0.8),
        },
        reverse=True,
    )
    turn_candidates = {
        maximum_angular_speed * index / 5.0
        for index in range(-5, 6)
        if index != 0
    }
    if abs(reference_turn) >= minimum_avoidance_angular_speed:
        turn_candidates.add(reference_turn)

    best: tuple[float, float, float, int | None, float] | None = None
    safest: tuple[float, float, float, int | None, float] | None = None
    start_x, start_y = robot_position
    for linear_speed in speed_candidates:
        for angular_speed in sorted(turn_candidates):
            clearance, agent_id, end_x, end_y, end_yaw = _candidate_clearance(
                robot_position=robot_position,
                robot_yaw=robot_yaw,
                linear_speed=linear_speed,
                angular_speed=angular_speed,
                humans=human_list,
                horizon=horizon,
                step=step,
                steering_duration=steering_duration,
            )
            dx = end_x - start_x
            dy = end_y - start_y
            progress = dx * direction[0] + dy * direction[1]
            lateral = abs(dx * direction[1] - dy * direction[0])
            heading_error = abs(_angle_difference(path_heading, end_yaw))
            score = (
                4.0 * progress
                - 1.8 * lateral
                - 0.8 * heading_error
                - 0.25 * abs(angular_speed)
                + 1.0 * linear_speed
            )
            candidate = (score, linear_speed, angular_speed, agent_id, clearance)
            if safest is None or clearance > safest[4] or (
                abs(clearance - safest[4]) <= 1e-6 and score > safest[0]
            ):
                safest = candidate
            if clearance >= safety_distance and (best is None or score > best[0]):
                best = candidate

    selected = best if best is not None else safest
    assert selected is not None
    return AvoidanceCommand(
        linear_speed=selected[1],
        angular_speed=selected[2],
        intervention=True,
        predicted_clearance=selected[4],
        agent_id=selected[3],
    )

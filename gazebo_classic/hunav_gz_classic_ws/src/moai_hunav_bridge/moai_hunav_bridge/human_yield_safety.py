from __future__ import annotations

from math import cos, hypot, sin


def filtered_velocity_from_positions(
    previous_position: tuple[float, float],
    current_position: tuple[float, float],
    elapsed: float,
    previous_velocity: tuple[float, float],
    smoothing_alpha: float,
    maximum_speed: float,
) -> tuple[float, float]:
    if elapsed <= 1e-6:
        return tuple(map(float, previous_velocity))

    measured_x = (float(current_position[0]) - float(previous_position[0])) / elapsed
    measured_y = (float(current_position[1]) - float(previous_position[1])) / elapsed
    measured_speed = hypot(measured_x, measured_y)
    maximum_speed = max(float(maximum_speed), 0.0)
    if maximum_speed > 0.0 and measured_speed > maximum_speed:
        scale = maximum_speed / measured_speed
        measured_x *= scale
        measured_y *= scale

    alpha = min(max(float(smoothing_alpha), 0.0), 1.0)
    return (
        alpha * measured_x + (1.0 - alpha) * float(previous_velocity[0]),
        alpha * measured_y + (1.0 - alpha) * float(previous_velocity[1]),
    )


def is_separating_from_stationary_robot(
    robot_position: tuple[float, float],
    human_position: tuple[float, float],
    human_velocity: tuple[float, float],
) -> bool:
    relative_x = float(human_position[0]) - float(robot_position[0])
    relative_y = float(human_position[1]) - float(robot_position[1])
    distance_rate_numerator = (
        relative_x * float(human_velocity[0])
        + relative_y * float(human_velocity[1])
    )
    return distance_rate_numerator > 0.0


def minimum_predicted_clearance(
    robot_position: tuple[float, float],
    robot_yaw: float,
    linear_speed: float,
    angular_speed: float,
    human_position: tuple[float, float],
    human_velocity: tuple[float, float],
    horizon: float,
    step: float,
) -> tuple[float, float]:
    """Predict closest center distance for a constant command and human velocity."""
    horizon = max(float(horizon), 0.0)
    step = max(float(step), 0.02)
    linear_speed = float(linear_speed)
    angular_speed = float(angular_speed)
    start_x, start_y = map(float, robot_position)
    human_x, human_y = map(float, human_position)
    human_vx, human_vy = map(float, human_velocity)

    minimum_distance = float("inf")
    minimum_time = 0.0
    sample_count = max(int(horizon / step), 0)
    for index in range(sample_count + 1):
        time_offset = min(index * step, horizon)
        if abs(angular_speed) < 1e-6:
            robot_x = start_x + linear_speed * cos(robot_yaw) * time_offset
            robot_y = start_y + linear_speed * sin(robot_yaw) * time_offset
        else:
            next_yaw = robot_yaw + angular_speed * time_offset
            turn_radius = linear_speed / angular_speed
            robot_x = start_x + turn_radius * (sin(next_yaw) - sin(robot_yaw))
            robot_y = start_y - turn_radius * (cos(next_yaw) - cos(robot_yaw))

        predicted_human_x = human_x + human_vx * time_offset
        predicted_human_y = human_y + human_vy * time_offset
        distance = hypot(
            predicted_human_x - robot_x,
            predicted_human_y - robot_y,
        )
        if distance < minimum_distance:
            minimum_distance = distance
            minimum_time = time_offset

    return minimum_distance, minimum_time


def minimum_path_predicted_clearance(
    path_points: list[tuple[float, float]],
    robot_position: tuple[float, float],
    path_speed: float,
    human_position: tuple[float, float],
    human_velocity: tuple[float, float],
    horizon: float,
    step: float,
) -> tuple[float, float]:
    """Predict clearance while the robot advances along its current global path."""
    if not path_points:
        return float("inf"), 0.0

    start_x, start_y = map(float, robot_position)
    nearest_index = min(
        range(len(path_points)),
        key=lambda index: hypot(
            float(path_points[index][0]) - start_x,
            float(path_points[index][1]) - start_y,
        ),
    )
    route = [(start_x, start_y)] + [
        (float(x), float(y)) for x, y in path_points[nearest_index + 1 :]
    ]
    cumulative_distances = [0.0]
    for first, second in zip(route, route[1:]):
        cumulative_distances.append(
            cumulative_distances[-1]
            + hypot(second[0] - first[0], second[1] - first[1])
        )

    horizon = max(float(horizon), 0.0)
    step = max(float(step), 0.02)
    path_speed = max(float(path_speed), 0.0)
    human_x, human_y = map(float, human_position)
    human_vx, human_vy = map(float, human_velocity)

    minimum_distance = float("inf")
    minimum_time = 0.0
    segment_index = 0
    sample_count = max(int(horizon / step), 0)
    for index in range(sample_count + 1):
        time_offset = min(index * step, horizon)
        target_distance = path_speed * time_offset
        while (
            segment_index + 1 < len(cumulative_distances)
            and cumulative_distances[segment_index + 1] < target_distance
        ):
            segment_index += 1

        if segment_index + 1 >= len(route):
            robot_x, robot_y = route[-1]
        else:
            segment_start = cumulative_distances[segment_index]
            segment_end = cumulative_distances[segment_index + 1]
            segment_length = max(segment_end - segment_start, 1e-9)
            fraction = min(max((target_distance - segment_start) / segment_length, 0.0), 1.0)
            first = route[segment_index]
            second = route[segment_index + 1]
            robot_x = first[0] + fraction * (second[0] - first[0])
            robot_y = first[1] + fraction * (second[1] - first[1])

        predicted_human_x = human_x + human_vx * time_offset
        predicted_human_y = human_y + human_vy * time_offset
        distance = hypot(
            predicted_human_x - robot_x,
            predicted_human_y - robot_y,
        )
        if distance < minimum_distance:
            minimum_distance = distance
            minimum_time = time_offset

    return minimum_distance, minimum_time

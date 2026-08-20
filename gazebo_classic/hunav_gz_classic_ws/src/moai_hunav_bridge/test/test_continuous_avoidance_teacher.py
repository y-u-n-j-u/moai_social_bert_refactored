import math

from moai_hunav_bridge.continuous_avoidance_teacher import (
    MovingHuman,
    choose_avoidance_side,
    lane_tracking_command,
    select_continuous_avoidance_command,
)


def test_crossing_conflict_keeps_forward_motion_and_turns_behind_human():
    command = select_continuous_avoidance_command(
        robot_position=(-2.6, 0.0),
        robot_yaw=0.0,
        nominal_linear_speed=0.7,
        nominal_angular_speed=0.0,
        humans=[MovingHuman(1, 0.0, -1.3, 0.0, 0.8)],
        path_direction=(1.0, 0.0),
    )

    assert command.intervention
    assert command.linear_speed >= 0.35
    assert command.angular_speed < 0.0
    assert command.predicted_clearance >= 1.2


def test_non_conflicting_human_keeps_nominal_command():
    command = select_continuous_avoidance_command(
        robot_position=(0.0, 0.0),
        robot_yaw=0.0,
        nominal_linear_speed=0.65,
        nominal_angular_speed=0.05,
        humans=[MovingHuman(1, 2.0, 5.0, 0.0, 0.8)],
        path_direction=(1.0, 0.0),
    )

    assert not command.intervention
    assert math.isclose(command.linear_speed, 0.65)
    assert math.isclose(command.angular_speed, 0.05)


def test_predicted_conflict_outside_activation_distance_keeps_nominal_command():
    command = select_continuous_avoidance_command(
        robot_position=(-5.0, 0.0),
        robot_yaw=0.0,
        nominal_linear_speed=0.8,
        nominal_angular_speed=0.0,
        humans=[MovingHuman(1, 0.0, -5.0, 0.0, 0.8)],
        path_direction=(1.0, 0.0),
        horizon=8.0,
        activation_distance=5.0,
    )

    assert not command.intervention
    assert math.isclose(command.linear_speed, 0.8)
    assert math.isclose(command.angular_speed, 0.0)


def test_conflict_never_selects_zero_speed():
    command = select_continuous_avoidance_command(
        robot_position=(-1.8, 0.0),
        robot_yaw=0.0,
        nominal_linear_speed=0.0,
        nominal_angular_speed=0.0,
        humans=[MovingHuman(1, 0.0, -0.8, 0.0, 0.8)],
        path_direction=(1.0, 0.0),
    )

    assert command.intervention
    assert command.linear_speed >= 0.35


def test_crossing_side_is_behind_human_motion():
    side = choose_avoidance_side(
        path_direction=(1.0, 0.0),
        robot_yaw=0.0,
        human_position=(0.0, -2.0),
        human_velocity=(0.0, 0.8),
        path_reference=(-2.0, 0.0),
    )

    assert side == -1


def test_lane_tracking_turns_to_negative_offset_without_stopping():
    linear_speed, angular_speed = lane_tracking_command(
        robot_yaw=0.0,
        path_direction=(1.0, 0.0),
        current_lateral_offset=0.0,
        avoidance_side=-1,
    )

    assert linear_speed >= 0.35
    assert angular_speed < 0.0

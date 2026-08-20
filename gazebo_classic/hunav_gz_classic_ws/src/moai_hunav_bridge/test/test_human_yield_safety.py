from __future__ import annotations

import importlib.util
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SAFETY_PATH = PACKAGE_ROOT / "moai_hunav_bridge" / "human_yield_safety.py"
SPEC = importlib.util.spec_from_file_location("human_yield_safety", SAFETY_PATH)
SAFETY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SAFETY)


def test_predicts_perpendicular_crossing_conflict() -> None:
    clearance, time_to_closest = SAFETY.minimum_predicted_clearance(
        robot_position=(-3.0, 0.0),
        robot_yaw=0.0,
        linear_speed=1.0,
        angular_speed=0.0,
        human_position=(0.0, -3.0),
        human_velocity=(0.0, 1.0),
        horizon=4.0,
        step=0.05,
    )

    assert clearance < 1e-6
    assert abs(time_to_closest - 3.0) < 0.05


def test_stopping_before_crossing_preserves_clearance() -> None:
    clearance, _ = SAFETY.minimum_predicted_clearance(
        robot_position=(-2.0, 0.0),
        robot_yaw=0.0,
        linear_speed=0.0,
        angular_speed=0.0,
        human_position=(0.0, -3.0),
        human_velocity=(0.0, 1.0),
        horizon=6.0,
        step=0.05,
    )

    assert abs(clearance - 2.0) < 1e-6


def test_release_requires_pedestrian_to_be_moving_away() -> None:
    robot_position = (-2.0, 0.0)
    assert not SAFETY.is_separating_from_stationary_robot(
        robot_position,
        human_position=(0.0, 1.0),
        human_velocity=(0.0, -1.0),
    )
    assert SAFETY.is_separating_from_stationary_robot(
        robot_position,
        human_position=(0.0, -1.0),
        human_velocity=(0.0, -1.0),
    )


def test_global_path_prediction_catches_conflict_after_a_turn() -> None:
    clearance, time_to_closest = SAFETY.minimum_path_predicted_clearance(
        path_points=[(-3.0, -3.0), (0.0, -3.0), (0.0, 0.0), (3.0, 0.0)],
        robot_position=(-3.0, -3.0),
        path_speed=1.0,
        human_position=(0.0, 3.0),
        human_velocity=(0.0, -0.5),
        horizon=7.0,
        step=0.05,
    )

    assert clearance < 1e-6
    assert abs(time_to_closest - 6.0) < 0.05


def test_long_horizon_reserves_a_future_crossing_line() -> None:
    short_clearance, _ = SAFETY.minimum_predicted_clearance(
        robot_position=(-2.0, 0.0),
        robot_yaw=0.0,
        linear_speed=0.0,
        angular_speed=0.0,
        human_position=(0.0, -10.0),
        human_velocity=(0.0, 1.0),
        horizon=6.0,
        step=0.05,
    )
    reserved_clearance, time_to_closest = SAFETY.minimum_predicted_clearance(
        robot_position=(-2.0, 0.0),
        robot_yaw=0.0,
        linear_speed=0.0,
        angular_speed=0.0,
        human_position=(0.0, -10.0),
        human_velocity=(0.0, 1.0),
        horizon=15.0,
        step=0.05,
    )

    assert short_clearance > 4.0
    assert abs(reserved_clearance - 2.0) < 1e-6
    assert abs(time_to_closest - 10.0) < 0.05


def test_world_velocity_is_estimated_from_position_updates() -> None:
    velocity = SAFETY.filtered_velocity_from_positions(
        previous_position=(0.0, -2.0),
        current_position=(-0.02, -1.84),
        elapsed=0.40,
        previous_velocity=(0.0, 0.40),
        smoothing_alpha=0.50,
        maximum_speed=2.0,
    )

    assert abs(velocity[0] + 0.025) < 1e-6
    assert abs(velocity[1] - 0.40) < 1e-6

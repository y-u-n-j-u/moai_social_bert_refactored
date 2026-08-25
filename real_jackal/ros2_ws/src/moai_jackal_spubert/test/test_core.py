import math

from moai_jackal_spubert.navigation_core import (
    estimate_velocity,
    lookahead_point,
    same_frame_id,
    transform_xy,
    validate_candidate_path,
)


class FreeMap:
    resolution = 0.1

    def path_collision_cost(self, path, radius, weight):
        del radius, weight
        return sum(float(x) > 1.5 for x, _ in path)


def test_same_frame_id_tolerates_legacy_leading_slash():
    assert same_frame_id("/base_link", "base_link")
    assert not same_frame_id("laser", "base_link")


def test_transform_xy_applies_rotation_and_translation():
    x, y = transform_xy((1.0, 0.0), (2.0, 3.0), math.pi / 2.0)
    assert math.isclose(x, 2.0, abs_tol=1e-6)
    assert math.isclose(y, 4.0, abs_tol=1e-6)


def test_estimate_velocity_uses_recent_samples():
    vx, vy = estimate_velocity([(0.0, 0.0), (0.4, 0.0), (0.8, 0.0)], 0.4)
    assert math.isclose(vx, 1.0, rel_tol=1e-6)
    assert math.isclose(vy, 0.0, abs_tol=1e-6)


def test_lookahead_point_interpolates_path():
    assert lookahead_point([(0.0, 0.0), (2.0, 0.0)], (0.0, 0.0), 0.7) == (0.7, 0.0)


def test_candidate_validation_rejects_swept_collision():
    check = validate_candidate_path(
        current=(0.0, 0.0),
        path=[(1.0, 0.0), (2.0, 0.0)],
        final_goal=(4.0, 0.0),
        map_provider=FreeMap(),
        footprint_radius=0.5,
        prediction_dt=0.4,
        maximum_model_speed=5.0,
        maximum_step_ratio=1.5,
        minimum_goal_progress=0.1,
        human_histories={},
        human_sample_dt=0.4,
        minimum_human_center_distance=1.2,
        human_radius=0.35,
        human_safety_margin=0.2,
    )
    assert not check.valid
    assert check.reason == "robot_footprint_collision"


def test_candidate_validation_rejects_predicted_human_conflict():
    class AlwaysFree:
        resolution = 0.1

        @staticmethod
        def path_collision_cost(path, radius, weight):
            del path, radius, weight
            return 0.0

    check = validate_candidate_path(
        current=(0.0, 0.0),
        path=[(0.4, 0.0), (0.8, 0.0), (1.2, 0.0)],
        final_goal=(5.0, 0.0),
        map_provider=AlwaysFree(),
        footprint_radius=0.5,
        prediction_dt=0.4,
        maximum_model_speed=2.0,
        maximum_step_ratio=1.5,
        minimum_goal_progress=0.1,
        human_histories={7: [(1.2, 1.2), (1.2, 0.8), (1.2, 0.4)]},
        human_sample_dt=0.4,
        minimum_human_center_distance=1.2,
        human_radius=0.35,
        human_safety_margin=0.2,
    )
    assert not check.valid
    assert check.reason == "predicted_human_clearance"

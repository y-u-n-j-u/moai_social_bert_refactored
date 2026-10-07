import math

import pytest

from moai_jackal_spubert.navigation_core import (
    estimate_velocity,
    lookahead_point,
    PathProgressTracker,
    projected_lookahead,
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


def test_lookahead_does_not_flip_at_nearest_vertex_boundary():
    path = [(0.0, 0.0), (0.8, 0.0), (1.6, 0.0), (2.4, 0.0)]
    headings = []
    for x in (0.39, 0.41):
        target = lookahead_point(path, (x, 0.05), 0.7)
        assert target == pytest.approx((x + 0.7, 0.0))
        headings.append(math.atan2(target[1] - 0.05, target[0] - x))
    assert abs(headings[1] - headings[0]) < 1e-6


def test_lookahead_counts_distance_from_projection_not_lateral_offset():
    assert lookahead_point([(0, 0), (3, 0)], (1, 2), 0.7) == pytest.approx((1.7, 0))


def test_lookahead_follows_corner_instead_of_robot_to_vertex_chord():
    path = [(0, 0), (1, 0), (1, 2)]
    assert lookahead_point(path, (0.8, -0.1), 0.7) == pytest.approx((1, 0.5))


def test_lookahead_ignores_duplicate_points_and_clamps_at_endpoint():
    path = [(0, 0), (0, 0), (1, 0), (1, 0), (2, 0)]
    assert lookahead_point(path, (0.4, 0.1), 0.7) == pytest.approx((1.1, 0))
    assert lookahead_point(path, (1.8, 0.1), 0.7) == (2, 0)


@pytest.mark.parametrize("current, expected", [((-1, 0), (0.7, 0)), ((3, 0), (2, 0))])
def test_lookahead_clamps_projection_to_path(current, expected):
    assert lookahead_point([(0, 0), (2, 0)], current, 0.7) == pytest.approx(expected)


@pytest.mark.parametrize("path", [[], [(0, 0)], [(1, 1), (1, 1)], [(0, 0), (math.nan, 1)]])
def test_unusable_paths_have_no_target(path):
    assert lookahead_point(path, (0, 0), 0.7) is None


@pytest.mark.parametrize("current, distance", [((math.nan, 0), 0.7), ((0, 0), math.inf)])
def test_nonfinite_tracking_input_has_no_target(current, distance):
    assert lookahead_point([(0, 0), (2, 0)], current, distance) is None


def test_projection_details_and_zero_lookahead():
    result = projected_lookahead([(0, 0), (1, 0), (1, 2)], (0.8, -0.1), 0.7)
    assert result.segment_index == 0
    assert result.target_segment_index == 1
    assert result.projection == pytest.approx((0.8, 0))
    assert result.progress_m == pytest.approx(0.8)
    assert result.target_progress_m == pytest.approx(1.5)
    assert result.cross_track_error_m == pytest.approx(0.1)
    assert lookahead_point([(0, 0), (2, 0)], (0.4, 0.2), 0) == pytest.approx((0.4, 0))


def test_same_geometry_cannot_regress_and_new_geometry_or_frame_resets():
    tracker = PathProgressTracker()
    assert tracker.set_path([(0, 0), (3, 0)], "odom")
    first = tracker.update((1, 0.05), 0.7)
    assert first.progress_m == pytest.approx(1)
    assert not tracker.set_path([(0, 0), (3, 0)], "odom")
    second = tracker.update((0.9, 0.05), 0.7)
    assert second.progress_m == pytest.approx(1)
    assert tracker.version == 1
    assert tracker.set_path([(0, 0), (4, 0)], "odom")
    assert tracker.progress is None
    assert tracker.update((0.3, 0), 0.7).progress_m == pytest.approx(0.3)
    assert tracker.set_path([(0, 0), (4, 0)], "different_odom")
    assert tracker.progress is None
    assert tracker.version == 3


def test_projection_does_not_jump_to_later_crossing_branch():
    path = [(-2, 0), (2, 0), (2, 2), (0, 2), (0, -2)]
    tracker = PathProgressTracker()
    tracker.set_path(path, "odom")
    tracker.update((-0.1, 0.02), 0.7)
    # The later vertical branch is now closer, but is far ahead in arc length.
    result = tracker.update((0, 0.02), 0.7)
    assert result.segment_index == 0
    assert result.target == pytest.approx((0.7, 0))


def test_empty_path_clears_progress_and_reacquires_same_geometry():
    tracker = PathProgressTracker()
    tracker.set_path([(0, 0), (3, 0)], "odom")
    tracker.update((2, 0), 0.7)
    tracker.set_path([], "odom")
    assert tracker.update((0, 0), 0.7) is None
    tracker.set_path([(0, 0), (3, 0)], "odom")
    assert tracker.update((0, 0), 0.7).progress_m == 0


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


def test_corridor_obstacle_distance_can_follow_the_travel_direction():
    from moai_jackal_spubert.navigation_core import corridor_obstacle_distance as cod
    inc = 2 * math.pi / 720
    ranges = [math.inf] * 721
    ranges[round(math.pi / inc)] = 1.0  # one return straight ahead (angle 0) at 1.0 m
    assert cod(ranges, -math.pi, inc, 0.34) == pytest.approx(1.0)                 # default axis: seen
    assert cod(ranges, -math.pi, inc, 0.34, axis_angle=math.radians(50)) == math.inf   # path turned away: not in the way
    ranges = [math.inf] * 721
    ranges[round((math.radians(50) + math.pi) / inc)] = 1.0   # a return 50 deg to the left
    assert cod(ranges, -math.pi, inc, 0.34) == math.inf
    assert cod(ranges, -math.pi, inc, 0.34, axis_angle=math.radians(50)) == pytest.approx(1.0, abs=0.02)

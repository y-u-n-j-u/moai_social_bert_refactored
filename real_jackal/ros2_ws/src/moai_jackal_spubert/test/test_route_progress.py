"""Route detours must make progress without crossing to a distant route arm."""
import math

import pytest

from moai_jackal_spubert.navigation_core import validate_candidate_path
from moai_jackal_spubert.route_progress import RouteProgressReference


class EmptyMap:
    resolution = 0.1

    def path_collision_cost(self, points, radius, weight):
        return 0.0


def line(start, end, steps=12):
    return [(start[0] + (end[0] - start[0]) * i / steps,
             start[1] + (end[1] - start[1]) * i / steps)
            for i in range(1, steps + 1)]


def check(path, route=None, *, current=(0.0, 0.0), goal=(4.0, 0.0),
          map_provider=None, humans=None):
    return validate_candidate_path(
        current=current, path=path, final_goal=goal,
        map_provider=map_provider or EmptyMap(), footprint_radius=0.5,
        prediction_dt=0.4, maximum_model_speed=1.5, maximum_step_ratio=1.5,
        minimum_goal_progress=0.1, human_histories=humans or {},
        human_sample_dt=0.4, minimum_human_center_distance=1.2,
        human_radius=0.35, human_safety_margin=0.2,
        route_reference=RouteProgressReference(route, current) if route is not None else None,
    )


@pytest.mark.parametrize('route,end', [
    ([(0, 0), (0, 2), (4, 2), (4, 0)], (0, 1.2)),
    ([(0, 0), (-2, 0), (-2, 2), (4, 2), (4, 0)], (-1.2, 0)),
])
def test_route_detour_accepts_safe_sideways_or_initial_away_from_goal_motion(route, end):
    path = line((0, 0), end)
    old = check(path)
    new = check(path, route)
    assert old.reason == 'insufficient_goal_progress'
    assert new.valid
    assert new.goal_progress_m <= 0
    assert new.route_progress.effective_progress_m == pytest.approx(1.2)


@pytest.mark.parametrize('end,expected', [((1.2, 0), 'valid'), ((0, 0), 'insufficient_route_progress'),
                                        ((-1.2, 0), 'insufficient_route_progress')])
def test_forward_stationary_backward_on_straight_route(end, expected):
    result = check(line((0, 0), end), [(-2, 0), (4, 0)])
    assert result.reason == expected


def test_hairpin_does_not_count_other_arm_as_sixteen_meters_of_progress():
    route = [(0, 0), (10, 0), (10, 0.2), (0, 0.2)]
    result = check(line((2, 0), (1.9, 0.2)), route, current=(2, 0), goal=(0, 0.2))
    assert not result.valid
    assert result.reason == 'insufficient_route_progress'
    assert result.route_progress.arc_progress_m == pytest.approx(-0.1)


@pytest.mark.parametrize('point', [(0, 0), (0.001, 0), (0, -0.001), (-0.001, 0.001)])
def test_crossing_anchor_rejects_noise_sensitive_branch_choice(point):
    crossing = [(-2, -2), (2, 2), (2, -2), (-2, 2)]
    with pytest.raises(ValueError, match='ambiguous_progress_route'):
        RouteProgressReference(crossing, point)


def test_normal_right_angle_and_duplicate_vertices_are_not_ambiguous():
    route = [(0, 0), (0, 0), (1, 0), (1, 0), (1, 2), (4, 2)]
    reference = RouteProgressReference(route, (0.9, 0.1))
    result = reference.measure([(1, 0.2), (1, 0.8)])
    assert result.error == ''
    assert result.effective_progress_m > 0.5


@pytest.mark.parametrize('route,start,end', [
    ([(0, 0), (1, 0), (1, 2), (4, 2)], (0.8, 0.1), (1, 0.8)),
    ([(0, 0), (1, 0), (1, 2), (4, 2)], (0.9, 0.09), (1, 0.8)),
    ([(0, 0), (5, 0), (5, 5)], (4, 0.5), (5, 1.5)),
])
def test_off_route_approach_can_switch_between_adjacent_corner_legs(route, start, end):
    result = RouteProgressReference(route, start).measure(line(start, end))
    assert result.error == ''
    assert result.effective_progress_m > 0.5


@pytest.mark.parametrize('wobble', [False, True])
def test_corner_projection_switch_does_not_credit_nearly_stationary_path(wobble):
    start, end = (4.51, 0.48), (4.52, 0.49)
    path = line(start, end)
    if wobble:
        path = [start, (4.41, 0.48), start, (4.41, 0.48)] + path
    result = check(path, [(0, 0), (5, 0), (5, 5)], current=start, goal=(5, 5))
    assert result.reason == 'insufficient_route_progress'
    assert result.route_progress.effective_progress_m <= math.dist(start, end) + 1e-9


@pytest.mark.parametrize('start,end', [
    ((12, 0), (10.8, 0)),  # right to left along the main corridor
    ((0.4, 0), (0, 0)),   # approach the left turn before moving downward
    ((0, -1), (0, -2.2)), # after the left turn, toward the lower endpoint
    ((8, 0), (6.8, 0.4)), # a modest lateral detour on the straight section
])
def test_user_l_shaped_scenario_right_to_left_then_down(start, end):
    route = [(12, 0), (0, 0), (0, -3)]
    result = check(line(start, end), route, current=start, goal=(0, -3))
    assert result.valid
    if start == (0.4, 0):
        old = check(line(start, end), current=start, goal=(0, -3))
        assert old.reason == 'insufficient_goal_progress'


def test_user_l_shaped_scenario_traverses_left_turn():
    route = [(12, 0), (0, 0), (0, -3)]
    start = (0.4, 0)
    path = line(start, (0, 0), 4) + line((0, 0), (0, -0.8), 8)
    result = check(path, route, current=start, goal=(0, -3))
    assert result.valid
    assert result.route_progress.arc_progress_m == pytest.approx(1.2)


def test_route_corner_traversal_uses_arc_not_final_goal_dot():
    route = [(0, 0), (0, 1), (2, 1), (2, 0)]
    path = line((0, 0), (0, 1), 6) + line((0, 1), (0.6, 1), 6)
    result = check(path, route, goal=(2, 0))
    assert result.valid
    assert result.route_progress.arc_progress_m == pytest.approx(1.6)


def test_far_off_route_path_cannot_trade_offset_for_large_arc_gain():
    result = check(line((0, 0), (8, 6)), [(0, 0), (10, 0)], goal=(10, 0))
    assert result.maximum_step_m < 0.9
    assert result.reason == 'route_deviation_exceeded'


def test_lateral_excursion_counts_distance_needed_to_return_to_route():
    result = check(line((0, 0), (1.2, 1.15)), [(0, 0), (4, 0)])
    assert result.reason == 'insufficient_route_progress'
    assert result.route_progress.arc_progress_m == pytest.approx(1.2)
    assert result.route_progress.effective_progress_m == pytest.approx(0.05)


def test_route_projection_never_clamps_an_outside_projection_to_window_edge():
    reference = RouteProgressReference([(0, 0), (10, 0)], (0, 0))
    assert reference._projections((3, 0), 0, 1) == []


@pytest.mark.parametrize('route', [[], [(0, 0)], [(0, 0), (0, 0)], [(0, 0), (math.nan, 0)]])
def test_invalid_reference_is_explicitly_rejected(route):
    with pytest.raises(ValueError):
        RouteProgressReference(route, (0, 0))


def test_current_position_far_from_route_does_not_create_fake_progress():
    with pytest.raises(ValueError, match='progress_route_too_far'):
        RouteProgressReference([(0, 0), (4, 0)], (0, 2))


def test_collision_guard_still_rejects_a_route_progressing_detour():
    class BlockedMap(EmptyMap):
        def path_collision_cost(self, points, radius, weight):
            return float(any(y > 0.5 for x, y in points))
    result = check(line((0, 0), (0, 1.2)), [(0, 0), (0, 2), (4, 2), (4, 0)],
                   map_provider=BlockedMap())
    assert result.reason == 'robot_footprint_collision'


def test_human_guard_still_rejects_a_route_progressing_detour():
    result = check(line((0, 0), (0, 1.2)), [(0, 0), (0, 2), (4, 2), (4, 0)],
                   humans={1: [(0, 0.6)]})
    assert result.reason == 'predicted_human_clearance'


def test_kinematic_jump_is_not_excused_by_route_progress():
    result = check([(0, 1.2)] * 12, [(0, 0), (0, 2), (4, 2), (4, 0)])
    assert result.reason == 'kinematic_jump'


def test_rotating_and_translating_scene_preserves_route_progress():
    route = [(0, 0), (0, 2), (4, 2), (4, 0)]
    path = line((0, 0), (0, 1.2))
    def transform(p):
        return 3 - p[1], 7 + p[0]
    original = check(path, route)
    changed = check([transform(p) for p in path], [transform(p) for p in route],
                    current=transform((0, 0)), goal=transform((4, 0)))
    assert original.valid and changed.valid
    assert original.route_progress.effective_progress_m == pytest.approx(changed.route_progress.effective_progress_m)

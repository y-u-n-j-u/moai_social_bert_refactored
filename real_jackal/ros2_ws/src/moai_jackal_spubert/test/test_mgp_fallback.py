import math

from moai_jackal_spubert.mgp_fallback import curve_to_goal, pick_goal_samples


def test_curve_ends_at_the_goal_when_it_is_within_reach():
    path = curve_to_goal((0.0, 0.0), 0.0, (2.0, 0.5), count=12, spacing=0.25)
    assert path[-1] == (2.0, 0.5)
    assert len(path) <= 12


def test_curve_leaves_along_the_heading():
    path = curve_to_goal((0.0, 0.0), 0.0, (2.0, 1.0), count=12, spacing=0.25)
    first = path[0]
    assert abs(math.atan2(first[1], first[0])) < 0.35        # starts roughly straight ahead, then bends to the goal


def test_curve_steps_are_the_requested_spacing_except_the_last():
    path = curve_to_goal((0.0, 0.0), 0.0, (6.0, 0.0), count=12, spacing=0.25)
    assert len(path) == 12
    steps = [math.dist(a, b) for a, b in zip([(0.0, 0.0)] + path, path)]
    assert all(abs(s - 0.25) < 0.02 for s in steps)


def test_goal_behind_or_beside_the_robot_gives_no_curve():
    assert curve_to_goal((0.0, 0.0), 0.0, (-2.0, 0.0), 12, 0.25) is None
    assert curve_to_goal((0.0, 0.0), 0.0, (0.0, 2.0), 12, 0.25) is None


def test_goal_too_close_gives_no_curve():
    assert curve_to_goal((0.0, 0.0), 0.0, (0.1, 0.0), 12, 0.25) is None


def test_pick_goal_samples_safe_only_nearest_first_and_separated():
    goals = [(3.0, 0.0), (3.1, 0.0), (3.0, 1.0), (2.0, 0.0), (5.0, 0.0)]
    safe = [True, True, True, False, True]
    picked = pick_goal_samples(goals, safe, guidance=(3.0, 0.0), maximum=3)
    assert [i for i, _ in picked] == [0, 2, 4]               # 1 is closer than 0.4 m to 0, 3 is not safe


def test_without_spacing_the_points_span_the_whole_curve_and_end_at_the_goal():
    path = curve_to_goal((0.0, 0.0), 0.0, (2.5, 0.6), count=12)
    assert len(path) == 12
    assert math.dist(path[-1], (2.5, 0.6)) < 0.02


def test_a_goal_beyond_the_maximum_length_is_approached_not_reached():
    path = curve_to_goal((0.0, 0.0), 0.0, (10.0, 0.0), count=12, max_length_m=3.0)
    assert len(path) == 12 and 2.9 < path[-1][0] < 3.1


def test_lateral_offset_moves_the_curve_to_that_side_and_still_ends_at_the_goal():
    from moai_jackal_spubert.mgp_fallback import curve_shapes, curve_to_goal
    left = curve_to_goal((0.0, 0.0), 0.0, (3.0, 0.0), 12, None, 1.0, lateral_m=0.6)
    right = curve_to_goal((0.0, 0.0), 0.0, (3.0, 0.0), 12, None, 1.0, lateral_m=-0.6)
    plain = curve_to_goal((0.0, 0.0), 0.0, (3.0, 0.0), 12, None, 1.0)
    assert max(p[1] for p in left) > 0.1 and min(p[1] for p in right) < -0.1 and max(abs(p[1]) for p in plain) < 1e-9
    for path in (left, right):
        assert math.hypot(path[-1][0] - 3.0, path[-1][1]) < 1e-6
    shapes = curve_shapes(1.0, 0.6)
    assert shapes[0] == (1.0, 0.0) and len(shapes) == len(set(shapes)) and any(l > 0 for _, l in shapes) and any(l < 0 for _, l in shapes)

import math

from moai_jackal_spubert.goal_lifecycle import GoalLifecycle
from moai_jackal_spubert.planning_stability import PathOption, SelectionConfig, rank_valid_paths


def test_unsafe_candidate_never_wins_even_with_best_guidance_distance():
    options = [PathOption(0, [(0.4, 0), (1, 0)], 0, False), PathOption(1, [(0.4, 0.2), (1, 0.3)], 10, True)]
    ranked = rank_valid_paths(options, current=(0, 0), robot_yaw=0)
    assert [item.index for item in ranked] == [1]


def test_no_valid_path_is_not_replaced_with_previous_path():
    assert rank_valid_paths([PathOption(0, [(0.5, 0), (1, 0)], 0, False)], current=(0, 0), robot_yaw=0, previous_path=[(0, 0), (2, 0)]) == []


def test_continuity_can_outweigh_small_guidance_distance_advantage():
    right = [(0.4, -0.15), (0.8, -0.3), (1.2, -0.4)]
    left = [(x, -y) for x, y in right]
    options = [PathOption(0, left, 0.9, True), PathOption(1, right, 1.0, True)]
    ranked = rank_valid_paths(options, current=(0, 0), robot_yaw=0, previous_path=[(0, 0), *right])
    assert ranked[0].index == 1
    assert ranked[0].continuity_distance_m < 1e-9


def test_first_valid_mode_preserves_baseline_order():
    options = [PathOption(0, [(0.4, 0.4), (1, 1)], 4, True), PathOption(1, [(0.4, 0), (1, 0)], 0, True)]
    assert rank_valid_paths(options, current=(0, 0), robot_yaw=0, config=SelectionConfig(mode="first_valid"))[0].index == 0
    assert rank_valid_paths(options, current=(0, 0), robot_yaw=0)[0].index == 1


def test_rejects_nonfinite_candidates_and_wraps_heading_at_pi():
    options = [PathOption(0, [(math.nan, 0), (1, 0)], 0, True), PathOption(1, [(-0.5, -0.001), (-1, -0.002)], 1, True)]
    ranked = rank_valid_paths(options, current=(0, 0), robot_yaw=math.pi - 0.001)
    assert len(ranked) == 1
    assert abs(ranked[0].heading_error_rad) < 0.01


def test_goal_completion_is_latched_and_old_goal_cannot_complete_new_goal():
    state = GoalLifecycle("test")
    first = state.start()
    assert state.complete(first)
    assert state.completed
    second = state.start()
    assert first != second and not state.completed
    assert not state.complete(first)
    assert not state.completed
    assert state.complete(second)


def test_completion_from_old_bridge_session_is_ignored():
    old = GoalLifecycle("old")
    new = GoalLifecycle("new")
    old_id = old.start()
    new.start()
    assert not new.complete(old_id)

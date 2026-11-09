from moai_jackal_spubert.candidate_display_hold import CandidateDisplayHold, display_mgp


def shown(hold, now=0.0, generation=1, mgp=(3.0, 0.0)):
    hold.note_shown(now, generation, mgp)


def test_off_always_refreshes():
    hold = CandidateDisplayHold(enabled=False)
    shown(hold)
    assert hold.should_refresh(0.1, (0.0, 0.0), 1)


def test_first_set_is_shown():
    assert CandidateDisplayHold(enabled=True).should_refresh(0.0, (0.0, 0.0), 1)


def test_holds_until_the_robot_reaches_the_goal_sample():
    hold = CandidateDisplayHold(enabled=True, reach_radius_m=0.5)
    shown(hold, mgp=(3.0, 0.0))
    assert not hold.should_refresh(1.0, (0.2, 0.0), 1)
    assert not hold.should_refresh(5.0, (2.0, 0.0), 1)
    assert hold.should_refresh(9.0, (2.6, 0.0), 1)          # within 0.5 m of the goal sample


def test_new_goal_refreshes_at_once():
    hold = CandidateDisplayHold(enabled=True)
    shown(hold, generation=1)
    assert hold.should_refresh(0.5, (0.0, 0.0), 2)


def test_hold_runs_out_when_the_robot_does_not_get_there():
    hold = CandidateDisplayHold(enabled=True, max_hold_s=20.0)
    shown(hold)
    assert not hold.should_refresh(19.0, (0.0, 0.0), 1)
    assert hold.should_refresh(21.0, (0.0, 0.0), 1)


def test_a_set_without_a_selected_candidate_is_replaced_every_cycle():
    hold = CandidateDisplayHold(enabled=True)
    shown(hold, mgp=None)
    assert hold.should_refresh(0.5, (0.0, 0.0), 1)


def test_display_mgp_selected_candidate_wins():
    rows = [{"candidate_index": 1, "valid": False, "goal": [1.0, 0.0]}, {"candidate_index": 2, "valid": True, "goal": [4.0, 1.0]}]
    assert display_mgp(rows, 2) == (4.0, 1.0)


def test_display_mgp_falls_back_to_the_first_candidate_goal_when_nothing_was_selected():
    rows = [{"candidate_index": 1, "valid": False, "goal": [1.0, 0.0]}, {"candidate_index": 2, "valid": False, "goal": [4.0, 1.0]}]
    assert display_mgp(rows, None) == (1.0, 0.0)


def test_display_mgp_none_without_any_goal():
    assert display_mgp([{"candidate_index": 1, "valid": False, "goal": None}], None) is None
    assert display_mgp([], None) is None

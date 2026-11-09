import importlib.util
import pathlib

import pytest

PATH = pathlib.Path(__file__).resolve().parents[4] / "field_scripts" / "selection_view.py"


@pytest.fixture(scope="module")
def view():
    pytest.importorskip("tkinter")
    pytest.importorskip("rclpy")
    spec = importlib.util.spec_from_file_location("selection_view_under_test", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report(valid_flags):
    return {"selected_candidate_index": 0, "executed_candidate_index": 0, "executed_source": "model", "selection_reason": "only_valid_option",
            "candidates": [{"candidate_index": i, "rank": i + 1, "valid": ok, "reason": None if ok else "robot_footprint_collision",
                            "score": {"eligible": True, "total": 1.0} if ok else None} for i, ok in enumerate(valid_flags)]}


def test_at_most_two_rejected_rows(view):
    rows = list(view.candidate_rows(report([True, False, False, False, False]), 2))
    assert sum(1 for tag, _ in rows if tag == "rejected") == 2
    assert len(rows) == 3


def test_all_rows_without_a_limit(view):
    assert len(list(view.candidate_rows(report([True, False, False, False, False]), None))) == 5


def test_hidden_count(view):
    assert view.hidden_rejected(report([True, False, False, False, False]), 2) == 2
    assert view.hidden_rejected(report([True, False]), 2) == 0
    assert view.hidden_rejected(report([False] * 5), None) == 0


def hold_report(pose, mgp, generation=1, chosen=1):
    return {"robot_pose": list(pose) + [0.0], "goal_generation": generation, "selected_candidate_index": chosen,
            "candidates": [{"candidate_index": chosen, "valid": True, "goal": list(mgp)}]}


def test_table_stays_until_the_goal_sample_is_reached(view, monkeypatch):
    monkeypatch.setattr(view, "HOLD_ENABLED", True)
    hold = view.TableHold()
    first = hold_report((0.0, 0.0), (3.0, 0.0))
    assert hold.should_refresh(first, 0.0)
    hold.note_shown(first, 0.0)
    assert not hold.should_refresh(hold_report((1.0, 0.0), (9.0, 9.0)), 1.0)       # still on its way to (3, 0)
    assert hold.should_refresh(hold_report((2.7, 0.0), (9.0, 9.0)), 6.0)           # within 0.5 m of (3, 0)


def test_table_hold_ends_on_new_goal_or_timeout(view, monkeypatch):
    monkeypatch.setattr(view, "HOLD_ENABLED", True)
    hold = view.TableHold()
    first = hold_report((0.0, 0.0), (3.0, 0.0))
    hold.note_shown(first, 0.0)
    assert hold.should_refresh(hold_report((0.5, 0.0), (3.0, 0.0), generation=2), 1.0)
    assert hold.should_refresh(hold_report((0.5, 0.0), (3.0, 0.0)), view.HOLD_MAX_S + 1.0)


def test_only_three_results_are_shown(view):
    base = {"selected_candidate_index": 1, "executed_candidate_index": None, "executed_source": "retained", "selection_reason": "minimum_weighted_cost",
            "retained_option": {"path": [[0, 0], [1, 0]], "score": {"total": 0.1}},
            "candidates": [
                {"candidate_index": 1, "rank": 1, "valid": True, "reason": "valid", "score": {"eligible": True, "total": 0.5}},     # chosen, not published
                {"candidate_index": 2, "rank": 2, "valid": True, "reason": "valid", "score": {"eligible": False, "total": 0.9}},    # excluded by heading
                {"candidate_index": 3, "rank": 3, "valid": True, "reason": "valid", "score": {"eligible": True, "total": 0.7}},
                {"candidate_index": 4, "rank": 4, "valid": False, "reason": "robot_footprint_collision", "score": None}]}
    states = {values[2] for _, values in view.candidate_rows(base, None)}
    assert states <= {"검사 통과", "검사 탈락", "선택·발행"}
    assert len(list(view.candidate_rows(base, None))) == 4                       # no extra row for the retained previous path
    published = dict(base, executed_source="model", executed_candidate_index=1)
    assert "선택·발행" in {values[2] for _, values in view.candidate_rows(published, None)}

"""Continuity-aware choice among valid candidates: pure functions."""
import importlib.util
import math
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "candidate_select_under_test", Path(__file__).parents[1] / "moai_jackal_spubert/candidate_select.py")
cs = importlib.util.module_from_spec(spec)
import sys
sys.modules["candidate_select_under_test"] = cs  # dataclasses need the module registered
spec.loader.exec_module(cs)

STRAIGHT = [(0.2 * (i + 1), 0.0) for i in range(12)]
LEFT = [(0.1 * (i + 1) * 0.1, 0.1 * (i + 1)) for i in range(12)]        # leaves the heading by ~85 deg
RIGHT_SLIGHT = [(0.2 * (i + 1), -0.02 * (i + 1)) for i in range(12)]
CFG = cs.SelectionConfig(rank_weight=0.15, continuity_weight=2.0, heading_weight=0.3, heading_limit_rad=0.9)


def test_inactive_config_keeps_first_valid_candidate():
    assert cs.choose_candidate([LEFT, STRAIGHT], (0, 0), 0.0, STRAIGHT, 0.2, cs.SelectionConfig()) == 0


def test_heading_limit_drops_a_sideways_swerve_when_a_straight_candidate_exists():
    assert cs.choose_candidate([LEFT, STRAIGHT], (0, 0), 0.0, None, 99.0, CFG) == 1


def test_never_drops_every_candidate():
    assert cs.choose_candidate([LEFT], (0, 0), 0.0, None, 0.0, CFG) == 0
    assert cs.choose_candidate([LEFT, LEFT], (0, 0), 0.0, None, 0.0, CFG) in (0, 1)


def test_continuity_prefers_the_candidate_close_to_the_previous_path():
    previous = [(0.0, 0.0)] + RIGHT_SLIGHT
    mid_left = [(0.2 * (i + 1), 0.05 * (i + 1)) for i in range(12)]   # also fine heading-wise
    assert cs.choose_candidate([mid_left, RIGHT_SLIGHT], (0, 0), 0.0, previous, 0.3, CFG) == 1


def test_stale_previous_path_is_ignored():
    previous = [(0.0, 0.0)] + RIGHT_SLIGHT
    mid_left = [(0.2 * (i + 1), 0.05 * (i + 1)) for i in range(12)]
    # old path (3 s) is ignored, so the better-ranked first candidate wins on rank
    assert cs.choose_candidate([mid_left, RIGHT_SLIGHT], (0, 0), 0.0, previous, 3.0, CFG) == 0


def test_geometry_helpers():
    assert math.isclose(cs.point_to_polyline_distance((1, 1), [(0, 0), (2, 0)]), 1.0)
    assert cs.point_to_polyline_distance((5, 0), [(0, 0), (2, 0)]) == 3.0
    assert cs.continuity_cost(STRAIGHT, None, 6) == 0.0
    assert math.isclose(cs.heading_deviation((0, 0), 0.0, LEFT), math.atan2(0.4, 0.04), abs_tol=1e-6)
    assert cs.heading_deviation((0, 0), math.pi / 2, [(1, 0)] * 5) == math.pi / 2

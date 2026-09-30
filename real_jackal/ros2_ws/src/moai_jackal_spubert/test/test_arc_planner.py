"""Arc geometry for the model-free local planner: pure functions."""
import importlib.util
import math
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "arc_planner_under_test", Path(__file__).parents[1] / "moai_jackal_spubert/arc_planner.py"
)
arc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arc)


def test_straight_arc_follows_heading_at_even_spacing():
    pts = arc.arc_points((1.0, 2.0), math.pi / 2, 0.0, 1.2, 4)
    assert [round(p[0], 6) for p in pts] == [1.0] * 4
    assert [round(p[1], 6) for p in pts] == [2.3, 2.6, 2.9, 3.2]


def test_positive_curvature_turns_left_and_length_is_preserved():
    pts = arc.arc_points((0.0, 0.0), 0.0, 1.0, 1.0, 50)
    assert pts[-1][1] > 0.0 and pts[-1][0] < 1.0
    # end of a unit-radius arc after 1 rad of turning
    assert math.isclose(pts[-1][0], math.sin(1.0), abs_tol=1e-6)
    assert math.isclose(pts[-1][1], 1 - math.cos(1.0), abs_tol=1e-6)
    right = arc.arc_points((0.0, 0.0), 0.0, -1.0, 1.0, 50)
    assert math.isclose(right[-1][1], -pts[-1][1], abs_tol=1e-9)


def test_fan_is_straight_first_then_alternating():
    fan = arc.curvature_fan(0.3, 0.1)
    assert len(fan) == 7 and fan[0] == 0.0
    assert fan[1:] == [pytest.approx(v) for v in (0.1, -0.1, 0.2, -0.2, 0.3, -0.3)]
    assert arc.curvature_fan(0.0, 0.1) == [0.0]
    assert arc.curvature_fan(1.0, 0.0) == [0.0]


def test_bad_input_fails_closed():
    assert arc.arc_points((0, 0), float("nan"), 0.1, 1.0, 5) is None
    assert arc.arc_points((0, 0), 0.0, 0.1, 0.0, 5) is None
    assert arc.arc_points((0, 0), 0.0, 0.1, 1.0, 0) is None
    assert arc.arc_points((0, 0), 0.0, float("inf"), 1.0, 5) is None


def test_cost_prefers_near_target_then_gentle_then_long():
    near = arc.arc_cost((1.4, 0.0), (1.5, 0.0), 0.0, 1.5, 1.5)
    far = arc.arc_cost((1.4, 0.8), (1.5, 0.0), 0.0, 1.5, 1.5)
    assert near < far
    assert arc.arc_cost((1.5, 0.0), (1.5, 0.0), 0.2, 1.5, 1.5) < arc.arc_cost(
        (1.5, 0.0), (1.5, 0.0), 1.2, 1.5, 1.5)
    assert arc.arc_cost((1.5, 0.0), (1.5, 0.0), 0.0, 1.5, 1.5) < arc.arc_cost(
        (1.5, 0.0), (1.5, 0.0), 0.0, 1.0, 1.5)

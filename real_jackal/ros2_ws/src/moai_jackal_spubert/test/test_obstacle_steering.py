"""Avoidance steering: pure functions, no ROS."""
import importlib.util
import math
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "obstacle_steering_under_test", Path(__file__).parents[1] / "moai_jackal_spubert/obstacle_steering.py")
os_ = importlib.util.module_from_spec(spec)
sys.modules["obstacle_steering_under_test"] = os_
spec.loader.exec_module(os_)

CFG = os_.SteeringConfig()
N = 720
A0, DA = -math.pi, 2 * math.pi / N


def scan(points, corridor_half=1.1, far=8.0):
    """Corridor walls at +-corridor_half plus point obstacles (x, y) in the scan frame."""
    ranges = []
    for i in range(N):
        a = A0 + i * DA
        r = far
        s = math.sin(a)
        if abs(s) > 1e-3:
            r = min(r, corridor_half / abs(s))
        ranges.append(r)
    for x, y in points:
        a = math.atan2(y, x)
        i = int(round((a - A0) / DA)) % N
        ranges[i] = min(ranges[i], math.hypot(x, y))
    return ranges


def steer(points, **kw):
    return os_.avoidance_steering(scan(points), A0, DA, CFG, **kw)


def test_empty_corridor_walls_alone_do_not_steer():
    s = steer([])
    assert s.rate == 0.0 and s.side == 0 and math.isinf(s.nearest_x_m)


def test_obstacle_on_the_left_steers_right_and_vice_versa():
    left = steer([(2.0, 0.40)])
    right = steer([(2.0, -0.40)])
    assert left.rate < 0 and left.side == -1
    assert right.rate > 0 and right.side == 1


def test_closer_obstacle_turns_harder_and_far_ones_are_ignored():
    assert abs(steer([(1.0, 0.3)]).rate) > abs(steer([(2.5, 0.3)]).rate) > 0.0
    assert steer([(3.5, 0.3)]).rate == 0.0


def test_obstacle_beside_the_strip_is_ignored():
    assert steer([(1.5, 0.95)]).rate == 0.0


def test_dead_ahead_uses_the_preferred_side_and_is_capped():
    assert steer([(0.8, 0.0)], prefer_side=1).rate > 0
    assert steer([(0.8, 0.0)], prefer_side=-1).rate < 0
    assert abs(steer([(0.5, 0.0)], prefer_side=1).rate) <= CFG.max_rate + 1e-9


def test_path_already_turning_away_is_not_doubled():
    st = steer([(1.5, 0.3)])                        # obstacle left -> steer right (negative)
    assert st.rate < 0
    assert os_.combine_with_path_turn(0.0, st) == st.rate
    assert os_.combine_with_path_turn(-1.0, st) == -1.0          # path already turns right harder
    assert os_.combine_with_path_turn(0.3, st) < 0.3 + st.rate + 1e-9  # path toward obstacle: add fully
    assert math.isclose(os_.combine_with_path_turn(0.3, st), 0.3 + st.rate)


def test_disabled_config_does_nothing():
    off = os_.SteeringConfig(gain=0.0)
    assert os_.avoidance_steering(scan([(1.0, 0.0)]), A0, DA, off).rate == 0.0


GAP = os_.GapConfig()


def gap(points, **kw):
    return os_.gap_heading(scan(points, corridor_half=1.1), A0, DA, GAP, **kw)


def box_face(x, y0, y1, step=0.04):
    n = int(round((y1 - y0) / step))
    return [(x, y0 + i * step) for i in range(n + 1)]


def test_clear_straight_line_needs_no_gap_search():
    g = gap([])
    assert not g.blocked and g.heading is None


def test_box_dead_ahead_in_a_wide_corridor_heads_to_the_nearer_opening():
    # corridor 2.2 m wide (walls at +-1.1), box face 0.8 m wide slightly right of centre
    g = gap(box_face(3.0, -0.60, 0.20))
    assert g.blocked and g.heading is not None
    assert g.side == 1 and g.heading > 0.1          # opening on the left (0.2 .. 1.1 = 0.9 m)


def test_the_blocked_side_is_never_chosen_even_if_nearer():
    # the real run: right wall 0.25 m beyond the box, a 1.28 m opening on the left
    # robot axis y=0; box covers rel y -0.8..-0.1 (partly on the axis), right wall at -1.28 (the
    # 0.48 m sliver between box and wall is no way through), left wall at +0.97 (1.07 m opening)
    rel = [(2.0, -0.8 + 0.04 * i) for i in range(18)]
    rel += [(x / 10.0, -1.28) for x in range(5, 35)]
    rel += [(x / 10.0, 0.97) for x in range(5, 35)]
    g = os_.gap_heading(scan(rel, corridor_half=9.0), A0, DA, GAP)
    assert g.blocked and g.side == 1 and 0.1 < g.heading < 0.5
    assert g.gap_width_m > 1.0


def test_no_opening_wide_enough_reports_blocked_without_heading():
    g = gap(box_face(2.0, -0.9, 0.9))                # fills the 2.2 m corridor except 0.2 m slivers
    assert g.blocked and g.heading is None


def test_remembered_side_is_kept_when_both_sides_are_passable():
    wide = os_.GapConfig(view_half_m=3.0)
    pts = box_face(2.5, -0.40, 0.40)
    s = scan(pts, corridor_half=2.0)
    left = os_.gap_heading(s, A0, DA, wide, prefer_side=1)
    right = os_.gap_heading(s, A0, DA, wide, prefer_side=-1)
    assert left.side == 1 and right.side == -1

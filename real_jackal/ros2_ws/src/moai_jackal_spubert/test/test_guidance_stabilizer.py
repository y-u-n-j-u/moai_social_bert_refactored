"""Guidance point: comfortable-clearance preference and cycle-to-cycle stabilisation (pure functions)."""
import math

from moai_jackal_spubert.guided_spubert_runtime import adaptive_guidance_point_along_path, route_offset_guidance_point
from moai_jackal_spubert.guidance_stabilizer import StabilizerConfig, point_at_s, route_s, stabilize

ROUTE = [(0.0, 0.0), (10.0, 0.0)]
CFG = StabilizerConfig()


def test_route_projection_and_point_at_s():
    s, d = route_s(ROUTE, (3.0, 0.5))
    assert math.isclose(s, 3.0) and math.isclose(d, 0.5)
    assert point_at_s(ROUTE, 4.5) == (4.5, 0.0)
    assert point_at_s(ROUTE, 99.0) == (10.0, 0.0)


def test_gp_does_not_fall_back_more_than_the_allowed_backtrack():
    prev = (4.0, 0.0)
    got = stabilize(prev, (1.0, 0.0), ROUTE, lambda p: True, CFG)    # the safety search wants a much nearer GP
    assert math.isclose(got[0], 3.7)                                  # 0.3 m back at most while that stays safe


def test_gp_steps_forward_gradually():
    got = stabilize((2.0, 0.0), (9.0, 0.0), ROUTE, lambda p: True, CFG)
    assert math.isclose(got[0], 2.8)


def test_safety_wins_over_calm():
    prev = (4.0, 0.0)
    # the point 0.3 m back is blocked: the (safe) new GP is used as it is
    assert stabilize(prev, (1.0, 0.0), ROUTE, lambda p: p[0] < 2.0, CFG) == (1.0, 0.0)


def test_route_that_moved_away_resets_and_a_sideways_jump_is_spread_out():
    far = (4.0, 3.0)                                                  # previous GP is 3 m from the new route (> 2.5 m)
    assert stabilize(far, (5.0, 0.0), ROUTE, lambda p: True, CFG) == (5.0, 0.0)
    prev = (4.0, -0.6)
    got = stabilize(prev, (4.0, 0.6), [(0.0, 0.6), (10.0, 0.6)], lambda p: True, CFG)
    assert math.hypot(got[0] - prev[0], got[1] - prev[1]) <= 0.8 + 1e-9


def test_disabled_or_first_cycle_returns_the_new_point():
    assert stabilize(None, (3.0, 0.0), ROUTE, lambda p: True, CFG) == (3.0, 0.0)
    assert stabilize((1.0, 0.0), (3.0, 0.0), ROUTE, lambda p: True, StabilizerConfig(max_step_m=0.0)) == (3.0, 0.0)


def _safe_to(limit):
    return lambda pts: all(abs(p[1]) < 1e-9 and p[0] <= limit for p in pts)


def test_comfortable_point_wins_over_a_farther_one_that_barely_clears():
    # chords are 'safe' up to 3.0 m but 'comfortable' only up to 2.0 m
    plain = adaptive_guidance_point_along_path(ROUTE, (0.0, 0.0), (10.0, 0.0), 5.0, _safe_to(3.0))
    comfy = adaptive_guidance_point_along_path(ROUTE, (0.0, 0.0), (10.0, 0.0), 5.0, _safe_to(3.0),
                                               is_comfortable_path=_safe_to(2.0))
    assert math.isclose(plain[0], 3.0) and math.isclose(comfy[0], 2.0)


def test_no_comfortable_point_far_enough_keeps_the_old_choice():
    comfy = adaptive_guidance_point_along_path(ROUTE, (0.0, 0.0), (10.0, 0.0), 5.0, _safe_to(3.0),
                                               is_comfortable_path=_safe_to(1.0))   # only 1 m is comfortable (< 1.5 m)
    assert math.isclose(comfy[0], 3.0)


def _blocked_box(chord):
    """A box on the route at x 2.5..3.5 (|y| < 0.5): every straight line that ends on or beyond it is unsafe."""
    for p in chord:
        if 2.0 <= p[0] <= 4.0 and abs(p[1]) < 0.9:
            return False
    return True


def test_route_offset_finds_a_point_beside_the_route_when_the_route_itself_is_blocked():
    # no point on the route at 1.5..3.0 m is reachable (the box is on it), but the space beside it is free
    point = route_offset_guidance_point(ROUTE, (0.0, 0.0), (10.0, 0.0), _blocked_box)
    assert point is not None and abs(point[1]) >= 0.9 - 1e-9 and point[0] >= 1.4


def test_route_offset_prefers_a_comfortable_point_then_the_smaller_offset():
    safe = lambda chord: all(p[1] > -0.3 for p in chord)                       # only the left side is safe
    comfy = lambda chord: all(p[1] > 0.0 for p in chord)
    p1 = route_offset_guidance_point(ROUTE, (0.0, 0.1), (10.0, 0.0), safe)
    p2 = route_offset_guidance_point(ROUTE, (0.0, 0.1), (10.0, 0.0), safe, is_comfortable_path=comfy)
    assert p1 is not None and p2 is not None and p2[1] > 0.0


def test_route_offset_returns_none_when_nothing_is_safe():
    assert route_offset_guidance_point(ROUTE, (0.0, 0.0), (10.0, 0.0), lambda chord: False) is None


def test_blocked_calm_point_falls_back_to_a_safe_point_between_not_to_a_far_jump():
    prev = (1.0, 0.0)
    far_new = (8.0, 0.0)                                   # the safety search now wants 7 m ahead
    # the point 0.8 m ahead (1.8) is blocked, but 1.2 is fine
    got = stabilize(prev, far_new, ROUTE, lambda p: p[0] <= 1.5 or p[0] >= 7.5, CFG)
    assert 1.0 <= got[0] <= 1.5


def test_nothing_between_is_safe_then_the_new_point_is_taken():
    assert stabilize((1.0, 0.0), (8.0, 0.0), ROUTE, lambda p: p[0] >= 7.5, CFG) == (8.0, 0.0)

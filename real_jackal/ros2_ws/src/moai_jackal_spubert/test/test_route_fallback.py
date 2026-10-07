"""Geometry of the global-path fallback: pure functions, no ROS or model."""
import importlib.util
import math
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "route_fallback_under_test",
    Path(__file__).parents[1] / "moai_jackal_spubert/route_fallback.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
route_prefix_points = module.route_prefix_points

STRAIGHT = [(0.0, 0.0), (10.0, 0.0)]


def test_walks_ahead_at_fixed_spacing_from_the_projection():
    points = route_prefix_points(STRAIGHT, (2.0, 0.3), count=4, spacing=0.5)
    assert points == [(2.5, 0.0), (3.0, 0.0), (3.5, 0.0), (4.0, 0.0)]


def test_follows_a_corner_by_arc_length():
    route = [(0.0, 0.0), (1.0, 0.0), (1.0, 5.0)]
    points = route_prefix_points(route, (0.5, 0.0), count=3, spacing=0.75)
    assert points[0] == (1.0, 0.25)
    assert math.isclose(points[1][1], 1.0) and points[1][0] == 1.0
    assert math.isclose(points[2][1], 1.75)


def test_truncates_at_route_end_instead_of_extrapolating():
    points = route_prefix_points(STRAIGHT, (9.0, 0.0), count=12, spacing=0.5)
    assert points == [(9.5, 0.0), (10.0, 0.0)]


def test_robot_far_from_route_gets_no_prefix():
    assert route_prefix_points(STRAIGHT, (3.0, 1.5), count=4, spacing=0.5) is None
    assert route_prefix_points(STRAIGHT, (3.0, 1.5), count=4, spacing=0.5,
                               max_route_offset=2.0) is not None


def test_route_end_reached_returns_none():
    assert route_prefix_points(STRAIGHT, (10.0, 0.0), count=4, spacing=0.5) is None


def test_bad_input_fails_closed():
    assert route_prefix_points([], (0.0, 0.0), 4, 0.5) is None
    assert route_prefix_points([(0.0, 0.0)], (0.0, 0.0), 4, 0.5) is None
    assert route_prefix_points(STRAIGHT, (0.0, 0.0), 0, 0.5) is None
    assert route_prefix_points(STRAIGHT, (0.0, 0.0), 4, 0.0) is None
    assert route_prefix_points(STRAIGHT, (float("nan"), 0.0), 4, 0.5) is None
    assert route_prefix_points([(0.0, 0.0), (float("inf"), 0.0)], (0.0, 0.0), 4, 0.5) is None
    assert route_prefix_points([(1.0, 1.0), (1.0, 1.0)], (1.0, 1.0), 4, 0.5) is None


shift_path_laterally = module.shift_path_laterally
lateral_offsets = module.lateral_offsets


def test_shift_moves_left_of_travel_and_ramps_in_from_the_robot():
    route = [(0.5 * (i + 1), 0.0) for i in range(6)]
    shifted = shift_path_laterally(route, (0.0, 0.0), 0.2, taper_m=1.0)
    assert [round(p[0], 6) for p in shifted] == [p[0] for p in route]  # x unchanged on a straight route
    assert [round(p[1], 6) for p in shifted] == [0.1, 0.2, 0.2, 0.2, 0.2, 0.2]
    right = shift_path_laterally(route, (0.0, 0.0), -0.2, taper_m=1.0)
    assert right[-1][1] == -0.2


def test_shift_follows_direction_change_and_handles_zero_and_bad_input():
    corner = [(1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)]
    shifted = shift_path_laterally(corner, (0.0, 0.0), 0.3, taper_m=0.0)
    assert shifted[0] == (1.0, 0.3)  # travelling +x: left is +y
    assert shifted[-1] == (1.7, 2.0)  # travelling +y: left is -x
    assert shift_path_laterally(corner, (0.0, 0.0), 0.0) == corner
    assert shift_path_laterally([], (0.0, 0.0), 0.1) is None
    assert shift_path_laterally(corner, (0.0, 0.0), float("nan")) is None
    assert shift_path_laterally([(1.0, 0.0), (1.0, 0.0)], (1.0, 0.0), 0.1) is None


def test_offsets_are_ordered_nearest_to_centre_first():
    assert lateral_offsets(0.09, 0.03) == [0.0, 0.03, -0.03, 0.06, -0.06, 0.09, -0.09]
    assert lateral_offsets(0.0, 0.03) == [0.0]
    assert lateral_offsets(-1.0, 0.03) == [0.0]
    assert lateral_offsets(0.3, 0.0) == [0.0]


def test_longer_ramp_spreads_the_sideways_move_and_halves_the_heading_change():
    route = [(0.25 * (i + 1), 0.0) for i in range(12)]   # 3 m straight prefix

    def max_heading_change(taper):
        pts = shift_path_laterally(route, (0.0, 0.0), 0.8, taper_m=taper)
        chain = [(0.0, 0.0), *pts]
        return max(abs(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))) for a, b in zip(chain, chain[1:]))

    sharp, gentle = max_heading_change(0.4), max_heading_change(1.5)
    assert sharp > 55 and gentle < 32 and gentle < 0.55 * sharp

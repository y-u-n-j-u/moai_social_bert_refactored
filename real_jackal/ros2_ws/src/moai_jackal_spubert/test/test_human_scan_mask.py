import math

from moai_jackal_spubert.human_scan_mask import human_slow_scale, mask_people_in_ranges, nearest_person_ahead


def scan_with(point_xy, beams=181, span=math.pi):
    """One beam hits ``point_xy`` (range along its bearing); all other beams hit a wall 5 m away."""
    angle_min, inc = -span / 2, span / (beams - 1)
    ranges = [5.0] * beams
    bearing = math.atan2(point_xy[1], point_xy[0])
    ranges[round((bearing - angle_min) / inc)] = math.hypot(*point_xy)
    return ranges, angle_min, inc


def test_person_points_removed_wall_kept():
    ranges, a0, inc = scan_with((2.0, 0.0))
    out = mask_people_in_ranges(ranges, a0, inc, [(2.0, 0.0)], 0.5)
    assert math.isinf(out[90])
    assert out[0] == 5.0 and out[-1] == 5.0 and sum(math.isinf(v) for v in out) <= 3


def test_wall_beyond_the_radius_survives():
    ranges, a0, inc = scan_with((2.0, 0.0))
    out = mask_people_in_ranges(ranges, a0, inc, [(2.0, 1.0)], 0.5)
    assert out[90] == 2.0


def test_off_when_radius_zero_or_no_people():
    ranges, a0, inc = scan_with((2.0, 0.0))
    assert mask_people_in_ranges(ranges, a0, inc, [(2.0, 0.0)], 0.0) == ranges
    assert mask_people_in_ranges(ranges, a0, inc, [], 0.5) == ranges


def test_nearest_person_ahead_ignores_behind_and_side():
    assert nearest_person_ahead([(-1.0, 0.0), (0.2, 3.0), (2.0, 0.1)], 0.7) == math.hypot(2.0, 0.1)
    assert math.isinf(nearest_person_ahead([(-1.0, 0.0)], 0.7))


def test_human_slow_scale():
    assert human_slow_scale(1.0, 0.3, 0.7) == 1.0
    assert abs(human_slow_scale(0.5, 0.3, 0.7) - 0.5) < 1e-9
    assert human_slow_scale(0.3, 0.3, 0.7) == 0.0
    assert human_slow_scale(0.4, 0.3, 0.0) == 1.0

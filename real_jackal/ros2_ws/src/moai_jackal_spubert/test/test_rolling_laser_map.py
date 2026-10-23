import math

import numpy as np

from moai_jackal_spubert.rolling_laser_map import RollingLaserMapProvider


def build_map(ranges):
    provider = RollingLaserMapProvider(
        size_m=8.0,
        resolution=0.10,
        free_gap_fill_m=0.0,
        unknown_is_occupied=True,
    )
    provider.update_from_scan(
        robot_x=0.0,
        robot_y=0.0,
        robot_yaw=0.0,
        ranges=ranges,
        angle_min=0.0,
        angle_increment=0.1,
        range_min=0.1,
        range_max=3.5,
        free_ray_limit_m=3.0,
    )
    return provider


def value_at(provider, x, y):
    return int(provider.occupancy_at_world(np.asarray([x]), np.asarray([y]))[0])


def test_finite_range_marks_free_ray_and_obstacle_endpoint():
    provider = build_map([2.0])
    assert value_at(provider, 1.0, 0.0) == 1
    assert value_at(provider, 2.0, 0.0) == 2


def test_positive_infinity_marks_visible_free_space_without_fake_obstacle():
    provider = build_map([math.inf])
    assert value_at(provider, 2.5, 0.0) == 1
    assert value_at(provider, 3.0, 0.0) != 2


def test_unknown_space_is_blocked_for_execution():
    provider = build_map([2.0])
    assert provider.point_occupied_with_radius(0.0, 1.0, 0.1)


def test_scene_patch_shape_matches_model_contract():
    provider = build_map([math.inf])
    patches, attention = provider.scene_patches(
        trans=(0.0, 0.0),
        theta=0.0,
        env_range=10.0,
        env_resol=0.625,
        patch_size=16,
        side_patches=2,
    )
    assert patches.shape == (4, 256)
    assert attention.shape == (4,)


def _person_scan(provider, person_xy, wall_y=None):
    """Scan from the origin with a person (a few beams at ``person_xy``) and optionally a wall line at y = wall_y."""
    import math as _m
    beams, a0 = 721, -_m.pi
    ranges = [_m.inf] * beams
    inc = 2 * _m.pi / (beams - 1)
    for i in range(beams):
        a = a0 + i * inc
        r = _m.inf
        dx, dy = _m.hypot(*person_xy), _m.atan2(person_xy[1], person_xy[0])
        if abs(((a - dy + _m.pi) % (2 * _m.pi)) - _m.pi) < 0.07:
            r = dx
        elif wall_y is not None and _m.sin(a) != 0 and wall_y / _m.sin(a) > 0:
            r = min(r, wall_y / _m.sin(a))
        ranges[i] = r
    provider.update_from_scan(
        robot_x=0.0, robot_y=0.0, robot_yaw=0.0, ranges=ranges, angle_min=a0, angle_increment=inc,
        range_min=0.1, range_max=30.0,
    )


def _occupied_near(provider, x, y, reach=0.4):
    xs, ys = np.meshgrid(np.arange(x - reach, x + reach, 0.05), np.arange(y - reach, y + reach, 0.05))
    return int((provider.occupancy_at_world(xs.ravel(), ys.ravel()) == 2).sum())


def test_clear_people_frees_an_isolated_person_blob():
    provider = RollingLaserMapProvider()
    _person_scan(provider, (3.0, 0.0))
    assert _occupied_near(provider, 3.0, 0.0) > 0
    assert provider.clear_people([(3.0, 0.0)], 0.5) > 0
    assert _occupied_near(provider, 3.0, 0.0) == 0


def test_clear_people_keeps_a_person_merged_with_a_wall():
    provider = RollingLaserMapProvider()
    _person_scan(provider, (3.0, 1.0), wall_y=1.2)
    before = _occupied_near(provider, 3.0, 1.0)
    assert before > 0
    provider.clear_people([(3.0, 1.0)], 0.5, max_cells=20)
    assert _occupied_near(provider, 3.0, 1.0) == before


def test_clear_people_off_and_no_people_do_nothing():
    provider = RollingLaserMapProvider()
    _person_scan(provider, (3.0, 0.0))
    assert provider.clear_people([(3.0, 0.0)], 0.0) == 0
    assert provider.clear_people([], 0.5) == 0

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

"""Collision evidence must explain, never weaken, the existing map decision."""
import math

import numpy as np
import pytest

from moai_jackal_spubert.rolling_laser_map import RollingLaserMapProvider


def legacy_check(provider, x, y, radius):
    # Frozen baseline from c15c6f8; protects map-edge and quantization semantics.
    center = provider._world_to_cell(x, y)
    if center is None:
        return True
    reach = int(math.ceil(max(radius, 0.0) / provider.resolution))
    cx, cy = center
    for row in range(cy - reach, cy + reach + 1):
        for col in range(cx - reach, cx + reach + 1):
            if not (0 <= col < provider.width and 0 <= row < provider.height):
                return True
            if math.hypot((col-cx)*provider.resolution, (row-cy)*provider.resolution) > radius + 0.5*provider.resolution:
                continue
            value = provider.grid[row, col]
            if value == 2 or (value == 0 and provider.unknown_is_occupied):
                return True
    return False


@pytest.mark.parametrize('unknown_is_occupied', [True, False])
def test_diagnostics_preserve_existing_decisions_and_input(unknown_is_occupied):
    rng = np.random.default_rng(1024)
    p = RollingLaserMapProvider(size_m=4, unknown_is_occupied=unknown_is_occupied)
    p.grid[:] = rng.choice([0, 1, 2], size=p.grid.shape, p=[0.03, 0.95, 0.02])
    original = p.grid.copy()
    for radius in [0, 0.1, 0.34, 0.45, 0.5, 0.55]:
        for x, y in rng.uniform(-2.1, 2.1, size=(100, 2)):
            expected = legacy_check(p, x, y, radius)
            assert p.point_occupied_with_radius(x, y, radius) == expected
            assert (p.first_path_collision([(x, y)], radius) is not None) == expected
    np.testing.assert_array_equal(p.grid, original)


@pytest.mark.parametrize('value,kind', [(2, 'occupied'), (0, 'unknown')])
def test_distinguishes_obstacle_from_unknown(value, kind):
    p = RollingLaserMapProvider(size_m=4)
    p.grid.fill(1)
    col, row = p._world_to_cell(0.5, 0)
    p.grid[row, col] = value
    detail = p.first_path_collision([(-1, 0), (0.5, 0)], 0)
    assert detail['sample_index'] == 1
    assert detail['kind'] == kind
    assert detail['cell'] == [col, row]


def test_reports_outside_map_and_invalid_path():
    p = RollingLaserMapProvider(size_m=4)
    p.grid.fill(1)
    assert p.first_path_collision([(3, 0)], 0.5)['kind'] == 'outside_map'
    assert p.first_path_collision([(math.nan, 0)], 0.5)['kind'] == 'nonfinite_path'
    assert p.first_path_collision([(0, 0)], 0.5) is None


def test_start_collision_distinguished_from_a_later_blocker():
    p = RollingLaserMapProvider(size_m=4)
    p.grid.fill(1)
    col, row = p._world_to_cell(0.2, 0)
    p.grid[row, col] = 2
    detail = p.first_path_collision([(0, 0), (0.1, 0), (0.2, 0)], 0.5)
    assert detail['sample_index'] == 0
    assert detail['footprint_radius_m'] == 0.5

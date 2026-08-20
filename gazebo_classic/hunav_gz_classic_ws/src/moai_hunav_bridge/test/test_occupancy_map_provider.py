import unittest

import numpy as np

from moai_hunav_bridge.social_bert_compute_agents_node import OccupancyMapProvider


class OccupancyMapProviderFootprintTest(unittest.TestCase):
    @staticmethod
    def _provider() -> OccupancyMapProvider:
        provider = OccupancyMapProvider.__new__(OccupancyMapProvider)
        provider.resolution = 1.0
        provider.origin_x = 0.0
        provider.origin_y = 0.0
        provider.height = 4
        provider.width = 4
        provider.occupied = np.zeros((4, 4), dtype=bool)
        provider.occupied[2, 2] = True
        provider._inflated_occupancy_cache = {0: provider.occupied}
        return provider

    def test_circle_detects_diagonal_obstacle_corner(self):
        provider = self._provider()

        self.assertTrue(provider.point_occupied_with_radius(1.7, 2.3, 0.5))
        self.assertEqual(
            provider.path_collision_cost([(1.7, 2.3)], radius=0.5, weight=1.0),
            1.0,
        )

    def test_circle_remains_free_away_from_obstacle(self):
        provider = self._provider()

        self.assertFalse(provider.point_occupied_with_radius(0.5, 0.5, 0.25))


if __name__ == "__main__":
    unittest.main()

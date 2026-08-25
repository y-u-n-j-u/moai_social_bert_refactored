import math
import sys
import types
import unittest

import torch

from moai_hunav_bridge.guided_spubert_runtime import (
    GuidedSpubertRuntime,
    adaptive_guidance_point_along_path,
    guidance_point,
    guidance_point_along_path,
    heading_from_history,
    local_to_world,
    pad_history,
    sample_polyline,
    seed_runtime_rng,
    world_to_local,
)


class GuidedSpubertGeometryTest(unittest.TestCase):
    def test_runtime_seed_reproduces_mgp_random_stream(self):
        seed_runtime_rng(21, torch)
        first = torch.randn(4)
        seed_runtime_rng(21, torch)
        second = torch.randn(4)
        self.assertTrue(torch.equal(first, second))

    def test_guidance_point_uses_circle_intersection(self):
        point = guidance_point((1.0, 2.0), (11.0, 2.0), 8.0)
        self.assertAlmostEqual(point[0], 9.0)
        self.assertAlmostEqual(point[1], 2.0)

    def test_guidance_point_uses_nearby_final_goal(self):
        point = guidance_point((1.0, 2.0), (3.0, 2.0), 8.0)
        self.assertEqual(point, (3.0, 2.0))

    def test_route_guidance_follows_turn_in_global_path(self):
        point = guidance_point_along_path(
            [(0.0, 0.0), (0.0, 5.0), (10.0, 5.0)],
            current=(0.0, 0.0),
            final_goal=(10.0, 5.0),
            radius=8.0,
        )
        self.assertAlmostEqual(point[0], 3.0)
        self.assertAlmostEqual(point[1], 5.0)

    def test_route_guidance_starts_at_projection_nearest_robot(self):
        point = guidance_point_along_path(
            [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
            current=(4.0, 1.0),
            final_goal=(10.0, 10.0),
            radius=8.0,
        )
        self.assertAlmostEqual(point[0], 10.0)
        self.assertAlmostEqual(point[1], 2.0)

    def test_route_guidance_uses_final_goal_when_path_is_short(self):
        point = guidance_point_along_path(
            [(0.0, 0.0), (2.0, 0.0)],
            current=(0.0, 0.0),
            final_goal=(2.0, 0.0),
            radius=8.0,
        )
        self.assertEqual(point, (2.0, 0.0))

    def test_route_guidance_rejects_empty_path(self):
        with self.assertRaises(ValueError):
            guidance_point_along_path([], (0.0, 0.0), (1.0, 0.0), 8.0)

    def test_adaptive_route_guidance_keeps_eight_metres_in_free_space(self):
        point = adaptive_guidance_point_along_path(
            [(0.0, 0.0), (10.0, 0.0)],
            current=(0.0, 0.0),
            final_goal=(10.0, 0.0),
            max_radius=8.0,
            is_direct_path_safe=lambda _: True,
        )
        self.assertEqual(point, (8.0, 0.0))

    def test_adaptive_route_guidance_shortens_before_occluded_turn(self):
        def avoids_corner(points):
            endpoint = points[-1]
            return not (endpoint[0] > 0.0 and endpoint[1] >= 4.0)

        point = adaptive_guidance_point_along_path(
            [(0.0, 0.0), (0.0, 5.0), (10.0, 5.0)],
            current=(0.0, 0.0),
            final_goal=(10.0, 5.0),
            max_radius=8.0,
            min_radius=2.0,
            probe_step=0.5,
            is_direct_path_safe=avoids_corner,
        )
        self.assertEqual(point, (0.0, 5.0))

    def test_adaptive_route_guidance_uses_short_safe_recovery(self):
        def only_nearby_is_safe(points):
            endpoint = points[-1]
            return math.hypot(endpoint[0], endpoint[1]) <= 1.0

        point = adaptive_guidance_point_along_path(
            [(0.0, 0.0), (10.0, 0.0)],
            current=(0.0, 0.0),
            final_goal=(10.0, 0.0),
            max_radius=8.0,
            min_radius=2.0,
            probe_step=0.5,
            is_direct_path_safe=only_nearby_is_safe,
        )
        self.assertEqual(point, (1.0, 0.0))

    def test_adaptive_route_guidance_rejects_when_no_probe_is_safe(self):
        with self.assertRaises(ValueError):
            adaptive_guidance_point_along_path(
                [(0.0, 0.0), (10.0, 0.0)],
                current=(0.0, 0.0),
                final_goal=(10.0, 0.0),
                max_radius=8.0,
                is_direct_path_safe=lambda _: False,
            )

    def test_polyline_sampling_checks_between_sparse_tgp_points(self):
        points = sample_polyline([(0.0, 0.0), (1.0, 0.0)], max_spacing=0.3)
        self.assertEqual(len(points), 5)
        self.assertEqual(points[0], (0.0, 0.0))
        self.assertEqual(points[-1], (1.0, 0.0))
        self.assertTrue(any(0.4 < x < 0.6 for x, _ in points))

    def test_polyline_sampling_rejects_invalid_spacing(self):
        with self.assertRaises(ValueError):
            sample_polyline([(0.0, 0.0), (1.0, 0.0)], max_spacing=0.0)

    def test_candidate_mask_uses_world_footprint_collision(self):
        class FakeMapProvider:
            @staticmethod
            def path_collision_cost(path, radius, weight):
                self.assertAlmostEqual(radius, 0.375)
                return weight if path[0][0] >= 2.0 else 0.0

        runtime = GuidedSpubertRuntime.__new__(GuidedSpubertRuntime)
        runtime.map_provider = FakeMapProvider()
        runtime.footprint_radius = 0.375
        runtime.torch = torch
        candidates = torch.tensor([[[0.5, 0.0], [1.5, 0.0]]])

        mask = runtime._candidate_footprint_safe_mask(
            candidates,
            origin=(1.0, 0.0),
            theta=0.0,
        )

        self.assertEqual(mask.tolist(), [[True, False]])

    def test_target_frame_round_trip(self):
        world = (4.5, -1.25)
        origin = (1.2, 2.4)
        theta = 0.73
        local = world_to_local(world, origin, theta)
        restored = local_to_world(local, origin, theta)
        self.assertAlmostEqual(restored[0], world[0], places=6)
        self.assertAlmostEqual(restored[1], world[1], places=6)

    def test_short_history_is_left_padded(self):
        self.assertEqual(
            pad_history([(1.0, 2.0), (2.0, 3.0)], 4),
            [(1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (2.0, 3.0)],
        )

    def test_stationary_history_uses_robot_yaw(self):
        yaw = heading_from_history([(1.0, 1.0), (1.0, 1.0)], 1.25)
        self.assertAlmostEqual(yaw, 1.25)

    def test_moving_history_uses_motion_heading(self):
        yaw = heading_from_history([(0.0, 0.0), (0.0, 2.0)], -1.0)
        self.assertAlmostEqual(yaw, math.pi / 2.0)

    def test_incompatible_spubert_package_is_discarded(self):
        package = types.ModuleType("spubert")
        package.__file__ = "/old/workspace/spubert/__init__.py"
        child = types.ModuleType("spubert.model")
        child.__file__ = "/old/workspace/spubert/model.py"
        sys.modules["spubert"] = package
        sys.modules["spubert.model"] = child
        try:
            GuidedSpubertRuntime._discard_incompatible_module(
                "spubert", "/new/model/SPU-BERT"
            )
            self.assertNotIn("spubert", sys.modules)
            self.assertNotIn("spubert.model", sys.modules)
        finally:
            sys.modules.pop("spubert", None)
            sys.modules.pop("spubert.model", None)


if __name__ == "__main__":
    unittest.main()

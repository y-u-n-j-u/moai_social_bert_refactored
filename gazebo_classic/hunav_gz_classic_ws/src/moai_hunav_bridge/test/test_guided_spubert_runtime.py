import math
import sys
import types
import unittest

from moai_hunav_bridge.guided_spubert_runtime import (
    GuidedSpubertRuntime,
    guidance_point,
    heading_from_history,
    local_to_world,
    pad_history,
    world_to_local,
)


class GuidedSpubertGeometryTest(unittest.TestCase):
    def test_guidance_point_uses_circle_intersection(self):
        point = guidance_point((1.0, 2.0), (11.0, 2.0), 8.0)
        self.assertAlmostEqual(point[0], 9.0)
        self.assertAlmostEqual(point[1], 2.0)

    def test_guidance_point_uses_nearby_final_goal(self):
        point = guidance_point((1.0, 2.0), (3.0, 2.0), 8.0)
        self.assertEqual(point, (3.0, 2.0))

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

#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "postprocess_pedestrian_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("postprocess_pedestrian_dataset", SCRIPT)
POSTPROCESS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(POSTPROCESS)


def map_data(occupied: np.ndarray, resolution: float = 1.0):
    return {
        "occupied": np.asarray(occupied, dtype=np.float32),
        "resolution": float(resolution),
        "origin": (0.0, 0.0, 0.0),
    }


class RouteGuidanceTest(unittest.TestCase):
    @staticmethod
    def avoidance_args() -> SimpleNamespace:
        return SimpleNamespace(
            avoidance_velocity_window=3,
            avoidance_max_ttc=4.0,
            avoidance_min_ttc=0.2,
            avoidance_conflict_distance=1.2,
            avoidance_min_closing_speed=0.2,
            avoidance_response_window=3,
            avoidance_response_extra_time=1.0,
            avoidance_min_reference_speed=0.25,
            avoidance_min_slowdown_ratio=0.20,
            avoidance_min_route_deviation=0.35,
            min_social_distance=1.2,
        )

    def test_swept_footprint_detects_obstacle_between_recorded_poses(self):
        occupied = np.zeros((15, 15), dtype=np.float32)
        occupied[7, 6] = 1.0
        planner = POSTPROCESS.OccupancyGridRoutePlanner(
            map_data(occupied),
            clearance_m=0.0,
        )

        metrics = POSTPROCESS.trajectory_footprint_metrics(
            np.asarray([[2.0, 7.0], [10.0, 7.0]], dtype=np.float32),
            planner,
        )

        self.assertEqual(metrics["target_pose_collision_count"], 0.0)
        self.assertGreater(metrics["target_swept_collision_count"], 0.0)
        self.assertEqual(metrics["target_map_safe"], 0.0)

    def test_quality_reports_acceleration_and_path_yaw_rate(self):
        trajs = np.zeros((2, 20, 2), dtype=np.float32)
        trajs[0, :10, 0] = np.arange(10, dtype=np.float32) * 0.2
        trajs[0, 10:, 0] = trajs[0, 9, 0]
        trajs[0, 10:, 1] = np.arange(1, 11, dtype=np.float32) * 0.2
        trajs[1] = trajs[0] + np.asarray([0.0, 2.0], dtype=np.float32)

        quality = POSTPROCESS.sample_quality(trajs, 8, 12, 0.4)

        self.assertGreater(quality["max_acceleration"], 0.0)
        self.assertGreater(quality["max_yaw_rate"], 1.0)

    def test_reactive_avoidance_requires_predicted_conflict_and_response(self):
        dt = 0.4
        trajs = np.zeros((2, 20, 2), dtype=np.float32)
        trajs[0, :8, 0] = np.arange(-7, 1, dtype=np.float32) * dt
        trajs[1, :8, 0] = np.arange(14, 6, -1, dtype=np.float32) * dt
        trajs[0, 8:, 0] = np.arange(1, 13, dtype=np.float32) * 0.08
        trajs[1, 8:, 0] = 2.8 - np.arange(1, 13, dtype=np.float32) * dt
        args = self.avoidance_args()

        metrics = POSTPROCESS.avoidance_conflict_metrics(trajs, 8, 12, dt, args)
        quality = {
            **metrics,
            "same_time_min_distance_all": 1.3,
            "route_future_max_deviation_m": 0.1,
        }
        classification = POSTPROCESS.avoidance_classification(quality, args)

        self.assertEqual(metrics["avoidance_conflict"], 1.0)
        self.assertGreater(metrics["robot_slowdown_ratio"], 0.5)
        self.assertEqual(classification["avoidance_slowdown"], 1.0)
        self.assertEqual(classification["avoidance_reactive"], 1.0)

        quality["avoidance_conflict"] = 0.0
        classification = POSTPROCESS.avoidance_classification(quality, args)
        self.assertEqual(classification["avoidance_reactive"], 0.0)

    def test_route_response_measures_lateral_departure(self):
        target = np.zeros((20, 2), dtype=np.float32)
        target[:, 0] = np.linspace(0.0, 9.5, 20)
        target[8:, 1] = 0.5
        route = np.stack(
            (np.linspace(3.5, 10.0, 20), np.zeros(20)),
            axis=1,
        ).astype(np.float32)

        metrics = POSTPROCESS.route_response_metrics(target, 8, 12, route)

        self.assertAlmostEqual(metrics["route_future_max_deviation_m"], 0.5, places=5)

    def test_recorded_continuous_teacher_intervention_verifies_early_avoidance(self):
        args = self.avoidance_args()
        quality = {
            "avoidance_response": 1.0,
            "avoidance_safe_outcome": 1.0,
            "avoidance_reactive": 0.0,
            "avoidance_score": 0.0,
        }

        classification = POSTPROCESS.teacher_avoidance_classification(
            {
                "human_avoidance_mode": "continuous",
                "human_avoidance_active_any": True,
            },
            quality,
            args,
        )

        self.assertEqual(classification["avoidance_teacher_verified"], 1.0)
        self.assertEqual(classification["avoidance_reactive"], 1.0)

        classification = POSTPROCESS.teacher_avoidance_classification(
            {
                "human_avoidance_mode": "continuous",
                "human_avoidance_active_any": False,
            },
            quality,
            args,
        )
        self.assertEqual(classification["avoidance_teacher_verified"], 0.0)
        self.assertEqual(classification["avoidance_reactive"], 0.0)

    def test_timing_metrics_use_all_recorded_intervals(self):
        metrics = POSTPROCESS.sample_timing_metrics(
            {"frame_intervals_s": [0.4, 0.41, 0.62]},
            seq_len=4,
            expected_dt=0.4,
        )

        self.assertEqual(metrics["timing_available"], 1.0)
        self.assertAlmostEqual(metrics["max_frame_interval_error_s"], 0.22)

    def test_route_override_replaces_stale_straight_metadata(self):
        data = map_data(np.zeros((30, 30), dtype=np.float32))
        trajs = np.zeros((2, 20, 2), dtype=np.float32)
        trajs[0, :, 0] = np.linspace(2.0, 12.0, 20)
        trajs[0, :, 1] = 5.0
        trajs[1, :, 0] = np.linspace(2.0, 12.0, 20)
        trajs[1, :, 1] = 7.0
        route_guidance = np.asarray([8.0, 9.0], dtype=np.float32)
        args = SimpleNamespace(
            map_size_m=8.0,
            map_grid_size=16,
            guidance_radius=8.0,
        )

        sample = POSTPROCESS.make_model_sample(
            trajs,
            {
                "final_goal": [20.0, 5.0],
                "guidance_point": [10.0, 5.0],
                "guidance_policy": "circle_line_intersection_to_final_goal",
            },
            {},
            8,
            12,
            data,
            args,
            0,
            guidance_point_override=route_guidance,
            guidance_metadata={
                "guidance_policy": "inflated_occupancy_grid_route_lookahead"
            },
        )

        np.testing.assert_allclose(sample["guidance_point"], route_guidance)
        self.assertEqual(
            sample["meta"]["guidance_policy"],
            "inflated_occupancy_grid_route_lookahead",
        )

    def test_dual_route_map_produces_safe_route_guidance(self):
        map_yaml = (
            Path(__file__).resolve().parents[2]
            / "hunav_gazebo_wrapper"
            / "maps"
            / "training_dual_route.yaml"
        )
        data = POSTPROCESS.load_map(map_yaml)
        planner = POSTPROCESS.OccupancyGridRoutePlanner(
            data,
            clearance_m=0.375,
        )
        route, metrics = planner.route(
            np.asarray([-9.0, 0.0], dtype=np.float32),
            np.asarray([9.0, 0.0], dtype=np.float32),
            0.75,
            0.0,
        )

        guidance = POSTPROCESS.guidance_point_along_route(route, 8.0)

        self.assertGreater(metrics["route_path_length_m"], 18.0)
        self.assertTrue(planner.is_safe_world(guidance))

    def test_route_guidance_follows_free_space_around_barrier(self):
        occupied = np.zeros((25, 25), dtype=np.float32)
        occupied[1:23, 10] = 1.0
        data = map_data(occupied)
        planner = POSTPROCESS.OccupancyGridRoutePlanner(
            data,
            clearance_m=0.0,
        )
        start = np.asarray([5.0, 12.0], dtype=np.float32)
        goal = np.asarray([20.0, 12.0], dtype=np.float32)

        route, metrics = planner.route(start, goal, 0.0, 0.0)
        guidance = POSTPROCESS.guidance_point_along_route(route, 8.0)
        straight = POSTPROCESS.guidance_point_from_goal(start, goal, 8.0)

        self.assertGreater(metrics["route_path_length_m"], 15.0)
        self.assertGreater(float(np.linalg.norm(guidance - straight)), 1.0)
        row, col = POSTPROCESS.world_to_pixel(
            float(guidance[0]),
            float(guidance[1]),
            data,
        )
        self.assertFalse(planner.blocked[row, col])
        for point in route:
            row, col = POSTPROCESS.world_to_pixel(
                float(point[0]),
                float(point[1]),
                data,
            )
            self.assertFalse(planner.blocked[row, col])

    def test_short_route_uses_final_goal(self):
        data = map_data(np.zeros((12, 12), dtype=np.float32))
        planner = POSTPROCESS.OccupancyGridRoutePlanner(data, clearance_m=0.0)
        start = np.asarray([2.0, 4.0], dtype=np.float32)
        goal = np.asarray([5.0, 4.0], dtype=np.float32)
        route, _ = planner.route(start, goal, 0.0, 0.0)

        guidance = POSTPROCESS.guidance_point_along_route(route, 8.0)

        np.testing.assert_allclose(guidance, goal, atol=1e-6)

    def test_inflation_blocks_robot_center_near_obstacle(self):
        occupied = np.zeros((21, 21), dtype=np.float32)
        occupied[10, 10] = 1.0

        blocked = POSTPROCESS.inflate_occupancy_grid(
            occupied,
            resolution=0.1,
            clearance_m=0.25,
        )

        self.assertTrue(blocked[10, 10])
        self.assertTrue(blocked[8, 8])
        self.assertFalse(blocked[6, 10])

    def test_unsafe_final_goal_is_rejected_without_straight_fallback(self):
        occupied = np.zeros((15, 15), dtype=np.float32)
        occupied[7, 10] = 1.0
        data = map_data(occupied)
        planner = POSTPROCESS.OccupancyGridRoutePlanner(data, clearance_m=0.0)

        with self.assertRaisesRegex(ValueError, "final goal"):
            planner.route(
                np.asarray([2.0, 7.0], dtype=np.float32),
                np.asarray([10.0, 7.0], dtype=np.float32),
                0.0,
                0.0,
            )


if __name__ == "__main__":
    unittest.main()

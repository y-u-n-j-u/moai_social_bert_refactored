"""Heading regressions without ROS, a checkpoint, or GPU execution."""

import math
import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

from moai_jackal_spubert.guided_spubert_runtime import (
    GuidedSpubertRuntime,
    heading_from_history,
    world_to_local,
)
from moai_jackal_spubert.heading_stability import (
    HeadingConfig,
    HeadingSelector,
    angle_difference_rad,
)


class HeadingSelectorTest(unittest.TestCase):
    @staticmethod
    def moving_history(speed, heading=math.radians(10.0)):
        times = [0.0, 0.4, 0.8, 1.2]
        points = [(speed * t * math.cos(heading), speed * t * math.sin(heading)) for t in times]
        return points, times

    def test_one_millimetre_threshold_is_reproduced_but_not_used_by_default(self):
        selector = HeadingSelector()
        for dy, legacy_expected in [(0.0009, 0.0), (0.0011, math.pi / 2.0)]:
            history = [(0.0, 0.0), (0.0, dy)]
            self.assertAlmostEqual(heading_from_history(history, 0.0), legacy_expected)
            result = selector.select(history_xy=history, sample_times_s=[0.0, 0.4], odom_yaw_rad=0.0)
            self.assertEqual(result.theta_rad, 0.0)
            self.assertEqual(result.source, "odom_yaw_low_recent_motion")
            self.assertAlmostEqual(result.diagnostics["legacy_heading_rad"], legacy_expected)

    def test_default_follows_current_yaw_during_turn_in_place_and_reverse(self):
        selector = HeadingSelector()
        for yaw in (0.0, math.pi / 2.0, -math.pi + 0.01):
            result = selector.select(history_xy=[(0.0, 0.0)] * 3, sample_times_s=[0.0, 0.4, 0.8], odom_yaw_rad=yaw)
            self.assertEqual(result.theta_rad, yaw)
        result = selector.select(history_xy=[(0.0, 0.0), (-0.2, 0.0)], sample_times_s=[1.0, 1.4], odom_yaw_rad=0.0)
        self.assertEqual(result.theta_rad, 0.0)
        self.assertAlmostEqual(result.diagnostics["legacy_heading_rad"], math.pi)

    def test_adaptive_rejects_millimetre_noise(self):
        selector = HeadingSelector(HeadingConfig(mode="adaptive"))
        result = selector.select(history_xy=[(0.0, 0.0), (0.0, 0.0011)], sample_times_s=[0.0, 0.4], odom_yaw_rad=0.0)
        self.assertEqual(result.theta_rad, 0.0)
        self.assertEqual(result.source, "odom_yaw_low_recent_motion")

    def test_default_preserves_training_last_segment_during_reliable_motion(self):
        # A curved window has a different mean direction. The default must use
        # the latest segment, exactly as transform_to_target does in training.
        points = [(0, 0), (0.08, 0), (0.16, 0.01), (0.24, 0.03)]
        result = HeadingSelector().select(history_xy=points,
                                         sample_times_s=[0, 0.4, 0.8, 1.2],
                                         odom_yaw_rad=0.0)
        self.assertEqual(result.source, "guarded_recent_motion")
        self.assertAlmostEqual(result.theta_rad, math.atan2(0.02, 0.08))
        self.assertAlmostEqual(result.theta_rad, heading_from_history(points, 0))
        self.assertNotAlmostEqual(result.theta_rad, result.diagnostics["window_heading_rad"])

    def test_guarded_recent_lateral_noise_cannot_hide_in_forward_window(self):
        points = [(0, 0), (0.1, 0), (0.2, 0), (0.2, 0.05)]
        result = HeadingSelector().select(history_xy=points,
                                         sample_times_s=[0, 0.4, 0.8, 1.2],
                                         odom_yaw_rad=0.0)
        self.assertEqual(result.theta_rad, 0)
        self.assertEqual(result.source, "odom_yaw_disagreement")

    def test_explicit_odom_mode_remains_available(self):
        points, times = self.moving_history(0.2)
        result = HeadingSelector(HeadingConfig(mode="odom_yaw")).select(
            history_xy=points, sample_times_s=times, odom_yaw_rad=0.0)
        self.assertEqual(result.theta_rad, 0)
        self.assertEqual(result.source, "odom_yaw")

    def test_adaptive_uses_actual_timestamps(self):
        points, times = self.moving_history(0.12)
        selector = HeadingSelector(HeadingConfig(mode="adaptive"))
        result = selector.select(history_xy=points, sample_times_s=times, odom_yaw_rad=0.0)
        self.assertEqual(result.source, "adaptive_motion_window")
        self.assertAlmostEqual(result.diagnostics["window_speed_mps"], 0.12)
        selector.reset()
        slower_times = [0.0, 0.6, 1.2, 1.8]
        result = selector.select(history_xy=points, sample_times_s=slower_times, odom_yaw_rad=0.0)
        self.assertTrue(result.source.startswith("odom_yaw"))
        self.assertAlmostEqual(result.diagnostics["recent_speed_mps"], 0.08)

    def test_adaptive_hysteresis_and_reset(self):
        selector = HeadingSelector(HeadingConfig(mode="adaptive"))
        points, times = self.moving_history(0.12)
        selector.select(history_xy=points, sample_times_s=times, odom_yaw_rad=0.0)
        points, times = self.moving_history(0.07)
        result = selector.select(history_xy=points, sample_times_s=times, odom_yaw_rad=0.0)
        self.assertEqual(result.source, "adaptive_motion_window")
        selector.reset()
        result = selector.select(history_xy=points, sample_times_s=times, odom_yaw_rad=0.0)
        self.assertEqual(result.source, "odom_yaw_below_motion_threshold")

    def test_adaptive_turn_in_place_does_not_hold_old_motion_heading(self):
        selector = HeadingSelector(HeadingConfig(mode="adaptive"))
        points, times = self.moving_history(0.2, heading=0.0)
        selector.select(history_xy=points, sample_times_s=times, odom_yaw_rad=0.0)
        # The window still contains forward movement, but its latest segment
        # has stopped. A small current yaw change must be followed immediately.
        result = selector.select(history_xy=points + [points[-1]], sample_times_s=times + [1.6], odom_yaw_rad=0.2)
        self.assertEqual(result.theta_rad, 0.2)
        self.assertEqual(result.source, "odom_yaw_low_recent_motion")
        self.assertFalse(result.diagnostics["motion_active_after"])

    def test_adaptive_rejects_reverse_heading_disagreement(self):
        points, times = self.moving_history(0.2, heading=math.pi)
        result = HeadingSelector(HeadingConfig(mode="adaptive")).select(history_xy=points, sample_times_s=times, odom_yaw_rad=0.0)
        self.assertEqual(result.theta_rad, 0.0)
        self.assertEqual(result.source, "odom_yaw_disagreement")

    def test_wraparound_does_not_look_like_full_revolution(self):
        self.assertAlmostEqual(angle_difference_rad(math.radians(-179), math.radians(179)), math.radians(2))
        points, times = self.moving_history(0.2, heading=math.radians(-179))
        result = HeadingSelector(HeadingConfig(mode="adaptive")).select(history_xy=points, sample_times_s=times, odom_yaw_rad=math.radians(179))
        self.assertEqual(result.source, "adaptive_motion_window")
        self.assertAlmostEqual(result.diagnostics["window_yaw_disagreement_rad"], math.radians(2))

    def test_invalid_or_gapped_times_fall_back_to_current_yaw(self):
        for times in ([0.0, 0.0], [0.4, 0.0], [0.0, math.nan], [0.0, 2.0], None):
            with self.subTest(times=times):
                result = HeadingSelector(HeadingConfig(mode="adaptive")).select(history_xy=[(0.0, 0.0), (0.4, 0.0)], sample_times_s=times, odom_yaw_rad=0.1)
                self.assertEqual(result.theta_rad, 0.1)
                self.assertTrue(result.source.startswith("odom_yaw"))

    def test_nonfinite_geometry_and_mismatched_times_are_rejected(self):
        selector = HeadingSelector()
        with self.assertRaises(ValueError):
            selector.select(history_xy=[(0.0, 0.0)], sample_times_s=[0.0], odom_yaw_rad=math.nan)
        with self.assertRaises(ValueError):
            selector.select(history_xy=[(math.inf, 0.0)], sample_times_s=[0.0], odom_yaw_rad=0.0)
        with self.assertRaises(ValueError):
            selector.select(history_xy=[(0.0, 0.0)], sample_times_s=[], odom_yaw_rad=0.0)


class _ArrayTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __getitem__(self, key):
        return _ArrayTensor(self.value[key])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class RuntimeHeadingOverrideTest(unittest.TestCase):
    def setUp(self):
        # Exercise live runtime preprocessing and world decoding, replacing
        # only model/tensor execution so this test requires no checkpoint.
        runtime = GuidedSpubertRuntime.__new__(GuidedSpubertRuntime)
        runtime.obs_len = 8
        runtime.pred_len = 12
        runtime.num_nbr = 4
        runtime.seq_input_len = 58
        runtime.social_range = 2.0
        runtime.view_range = 20.0
        runtime.view_angle = math.pi / 3.0
        runtime.guidance_radius = 8.0
        runtime.d_sample = 20
        runtime.tgp_top_k = 1
        runtime.args = SimpleNamespace(reject_unknown_goals=True)
        runtime.torch = SimpleNamespace(no_grad=nullcontext)
        runtime._SpatialTokens = lambda _: None
        self.captured = {}

        def build_streams(**kwargs):
            self.captured["streams"] = kwargs
            return {}

        def build_scene(**kwargs):
            self.captured["scene"] = kwargs
            return {}

        runtime._build_streams = build_streams
        runtime._build_scene = build_scene
        keys = (
            "mgp_spatial_ids", "mgp_temporal_ids", "mgp_segment_ids", "mgp_attn_mask",
            "tgp_temporal_ids", "tgp_segment_ids", "tgp_attn_mask", "guidance_lbl",
            "env_spatial_ids", "env_temporal_ids", "env_segment_ids", "env_attn_mask",
            "envs", "envs_params",
        )
        runtime._tensor_batch = lambda *args: dict.fromkeys(keys)
        output = {
            "guided_pred_trajs": [[[(0.1 * step, 0.0) for step in range(1, 13)]]],
            "candidate_goals": [[(1.2, 0.0)]],
            "guided_candidate_goals": [[(1.2, 0.0)]],
            "guided_candidate_indices": [[0]],
            "guided_candidate_goal_valid": [[True]],
            "guided_candidate_guidance_distances": [[1.0]],
            "guided_trajectory_map_safe": [[True]],
            "guided_execution_valid": [[True]],
            "candidate_safe_mask": [[True]],
        }
        runtime.model = SimpleNamespace(inference_guided_candidates=lambda **kwargs: {key: _ArrayTensor(value) for key, value in output.items()})
        self.runtime = runtime
        self.inputs = {
            "robot_history": [(1.0, 2.0), (1.0, 2.0011)],
            "robot_yaw": 0.0,
            "human_histories": {7: [(1.5, 2.0), (1.5, 2.1)]},
            "final_goal": (5.0, 2.0),
            "guidance_point_world": (3.0, 2.0),
        }

    def test_default_legacy_and_override_apply_to_all_model_transforms(self):
        legacy = self.runtime.predict_candidates(**self.inputs)[0]
        self.assertAlmostEqual(self.runtime.last_input_diagnostics["theta_rad"], math.pi / 2.0)
        self.assertEqual(self.runtime.last_input_diagnostics["heading_source"], "legacy_history")
        self.assertAlmostEqual(legacy.path_world[0][1], 2.1011)

        stable = self.runtime.predict_candidates(**self.inputs, heading_override_rad=0.0, heading_source="odom_yaw")[0]
        diag = self.runtime.last_input_diagnostics
        self.assertEqual(diag["status"], "model_returned")
        self.assertEqual(diag["theta_rad"], 0.0)
        self.assertAlmostEqual(diag["legacy_heading_rad"], math.pi / 2.0)
        self.assertEqual(self.captured["scene"]["theta"], 0.0)
        origin = tuple(diag["origin_world"])
        np.testing.assert_allclose(self.captured["streams"]["trajs"][0, -13], world_to_local(self.inputs["robot_history"][-1], origin, 0.0), atol=1e-6)
        np.testing.assert_allclose(self.captured["streams"]["guidance_lbl"], world_to_local((3.0, 2.0), origin, 0.0), atol=1e-6)
        np.testing.assert_allclose(self.captured["streams"]["trajs"][1, 7], world_to_local((1.5, 2.1), origin, 0.0), atol=1e-6)
        self.assertAlmostEqual(stable.path_world[0][0], 1.1)
        self.assertAlmostEqual(stable.path_world[0][1], 2.0011)
        self.assertAlmostEqual(stable.selected_goal_world[0], 2.2)

    def test_predict_forwards_override(self):
        result = self.runtime.predict(**self.inputs, heading_override_rad=0.0, heading_source="odom_yaw")
        self.assertAlmostEqual(result.path_world[0][0], 1.1)
        self.assertEqual(self.runtime.last_input_diagnostics["heading_source"], "odom_yaw")

    def test_nonfinite_override_or_input_fails_before_model(self):
        cases = [
            {"heading_override_rad": math.nan},
            {"heading_override_rad": math.inf},
            {"robot_yaw": math.nan},
            {"robot_history": [(0.0, math.inf)]},
            {"human_histories": {1: [(0.0, math.nan)]}},
            {"final_goal": (math.inf, 0.0)},
            {"guidance_point_world": (0.0, math.nan)},
            {"robot_history": []},
        ]
        for changes in cases:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self.runtime.predict_candidates(**{**self.inputs, **changes})
                self.assertEqual(self.runtime.last_input_diagnostics["status"], "invalid_input")
                self.assertFalse(self.runtime.last_input_diagnostics["input_valid"])


if __name__ == "__main__":
    unittest.main()

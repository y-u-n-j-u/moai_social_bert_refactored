from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loss import goal_collision_loss, pos_collision_loss
from src.model import GoalPooler
from src.utils import bootstrap_paths

bootstrap_paths()

from spubert.datasets.moai_social_nav_extended_goal import (
    _SpatialTokens,
    build_local_map_streams,
    build_mgp_tgp_streams,
)
from spubert.model import (
    SBertPlusFTModel,
    classify_map_points,
    select_guided_goal_candidates,
)


class _FixedMGP(torch.nn.Module):
    def __init__(self, goals: torch.Tensor):
        super().__init__()
        self.register_buffer("goals", goals)

    def inference(self, spatial_ids, **kwargs):
        return {
            "pred_goals": self.goals.expand(spatial_ids.size(0), -1, -1),
            "attentions": None,
        }


class _MovingTGP(torch.nn.Module):
    def inference(self, spatial_ids, **kwargs):
        return {
            "pred_trajs": torch.ones(
                spatial_ids.size(0),
                12,
                2,
                device=spatial_ids.device,
                dtype=spatial_ids.dtype,
            ),
            "attentions": None,
        }


class _GoalAwareTGP(torch.nn.Module):
    def inference(self, spatial_ids, **kwargs):
        goals = spatial_ids[:, 21, :2]
        paths = goals.unsqueeze(1).repeat(1, 12, 1)
        unsafe = goals[:, 0] > 0
        paths[unsafe, 0] = torch.tensor(
            [0.5, 0.5],
            device=spatial_ids.device,
            dtype=spatial_ids.dtype,
        )
        return {
            "pred_trajs": paths,
            "attentions": None,
        }


class GuidedTokenContractTest(unittest.TestCase):
    def test_mgp_and_tgp_token_positions(self):
        obs_len = 8
        pred_len = 12
        trajs = np.zeros((1, obs_len + pred_len, 2), dtype=np.float32)
        trajs[0, :obs_len, 0] = np.arange(obs_len, dtype=np.float32)
        goal = np.asarray([2.0, 1.0], dtype=np.float32)
        guidance = np.asarray([8.0, -0.5], dtype=np.float32)
        spatial = _SpatialTokens(view_range=20.0)

        streams = build_mgp_tgp_streams(
            trajs=trajs,
            goal_lbl=goal,
            guidance_lbl=guidance,
            obs_len=obs_len,
            pred_len=pred_len,
            num_nbr=4,
            seq_input_len=58,
            spatial=spatial,
        )
        mgp = streams["mgp_spatial_ids"]
        tgp = streams["tgp_spatial_ids"]
        self.assertEqual(mgp.shape, (58, 2))
        self.assertEqual(tgp.shape, (58, 2))
        np.testing.assert_allclose(
            mgp[9:20],
            np.broadcast_to(np.asarray(spatial.pad_id), (11, 2)),
        )
        np.testing.assert_allclose(mgp[20], np.asarray(spatial.msk_id))
        np.testing.assert_allclose(mgp[21], guidance)
        np.testing.assert_allclose(
            tgp[9:21],
            np.broadcast_to(np.asarray(spatial.msk_id), (12, 2)),
        )
        np.testing.assert_allclose(tgp[21], goal)
        np.testing.assert_array_equal(streams["mgp_segment_ids"][9:20], 0)
        np.testing.assert_array_equal(streams["mgp_attn_mask"][9:20], 0)
        self.assertEqual(int(streams["mgp_temporal_ids"][20]), 20)
        self.assertEqual(int(streams["mgp_temporal_ids"][21]), 21)

    def test_goal_pooler_uses_configured_token_offset(self):
        hidden_size = 2
        states = torch.zeros(1, 58, hidden_size)
        states[0, 20] = torch.tensor([2.0, 0.0])
        states[0, 21] = torch.tensor([0.0, 2.0])

        guided = GoalPooler(hidden_size, obs_len=8, pred_len=12, token_offset=0)
        legacy = GoalPooler(hidden_size, obs_len=8, pred_len=12, token_offset=1)
        for pooler in (guided, legacy):
            pooler.eval()
            with torch.no_grad():
                pooler.linear.weight.copy_(torch.eye(hidden_size))
                pooler.linear.bias.zero_()

        guided_out = guided(states)
        legacy_out = legacy(states)
        self.assertGreater(float(guided_out[0, 0]), float(guided_out[0, 1]))
        self.assertLess(float(legacy_out[0, 0]), float(legacy_out[0, 1]))


class GuidedMapContractTest(unittest.TestCase):
    def test_gazebo_top_row_is_flipped_to_positive_y_rows(self):
        source = np.zeros((4, 4), dtype=np.float32)
        source[1, 2] = 1.0
        source[2, 1] = 0.5
        streams = build_local_map_streams(
            sample={"local_map": source},
            patch_size=2,
            num_nbr=4,
            obs_len=8,
            local_map_size_m=4.0,
            env_resol=1.0,
            theta=0.0,
            align_to_target=False,
            source_unknown_value=0.5,
        )
        env = streams["envs"]
        self.assertEqual(float(env[2, 2]), 2.0)
        self.assertEqual(float(env[1, 1]), 0.0)
        self.assertEqual(float(env[3, 0]), 1.0)
        np.testing.assert_array_equal(
            streams["env_spatial_ids"][0],
            env[:2, :2].reshape(-1),
        )

    def test_selector_rejects_occupied_unknown_and_out_of_bounds(self):
        env = torch.ones(1, 4, 4)
        env[0, 2, 2] = 2.0
        env[0, 2, 1] = 0.0
        params = torch.tensor([[-2.0, -2.0, 4.0, 4.0, 1.0, 2.0]])
        candidates = torch.tensor(
            [[[0.5, 0.5], [-0.5, 0.5], [1.5, 0.5], [2.5, 0.5]]]
        )
        guidance = torch.tensor([[0.5, 0.5]])

        selected = select_guided_goal_candidates(
            candidates,
            guidance,
            env,
            params,
            reject_unknown=True,
        )
        self.assertEqual(int(selected["selected_indices"][0]), 2)
        torch.testing.assert_close(selected["selected_goals"][0], candidates[0, 2])
        self.assertEqual(selected["candidate_safe_mask"][0].tolist(), [False, False, True, False])
        self.assertTrue(bool(selected["selected_goal_valid"][0]))

    def test_all_invalid_returns_stop_goal_and_negative_index(self):
        env = torch.full((1, 4, 4), 2.0)
        params = torch.tensor([[-2.0, -2.0, 4.0, 4.0, 1.0, 2.0]])
        candidates = torch.tensor([[[0.5, 0.5], [1.5, 0.5]]])
        guidance = torch.tensor([[1.0, 0.5]])

        selected = select_guided_goal_candidates(
            candidates,
            guidance,
            env,
            params,
        )
        self.assertEqual(int(selected["selected_indices"][0]), -1)
        torch.testing.assert_close(selected["selected_goals"][0], torch.zeros(2))
        self.assertFalse(bool(selected["selected_goal_valid"][0]))
        self.assertTrue(bool(selected["all_candidates_invalid"][0]))

    def test_guided_inference_zeros_trajectory_when_all_candidates_are_invalid(self):
        model = SBertPlusFTModel.__new__(SBertPlusFTModel)
        torch.nn.Module.__init__(model)
        model.cfgs = SimpleNamespace(
            guidance_conditioned=True,
            obs_len=8,
            pred_len=12,
            goal_dim=2,
            view_range=20.0,
        )
        model.mgp_model = _FixedMGP(
            torch.tensor([[[0.5, 0.5], [1.5, 0.5]]])
        )
        model.tgp_model = _MovingTGP()

        trajectory_tokens = torch.zeros(1, 58, 2)
        token_ids = torch.zeros(1, 58, dtype=torch.long)
        map_tokens = torch.zeros(1, 4, 256)
        map_ids = torch.zeros(1, 4, dtype=torch.long)
        result = model.inference_guided(
            mgp_spatial_ids=trajectory_tokens,
            mgp_temporal_ids=token_ids,
            mgp_segment_ids=token_ids,
            mgp_attn_mask=token_ids,
            tgp_temporal_ids=token_ids,
            tgp_segment_ids=token_ids,
            tgp_attn_mask=token_ids,
            guidance_points=torch.tensor([[1.0, 0.5]]),
            env_spatial_ids=map_tokens,
            env_temporal_ids=map_ids,
            env_segment_ids=map_ids,
            env_attn_mask=map_ids,
            envs=torch.full((1, 4, 4), 2.0),
            envs_params=torch.tensor([[-2.0, -2.0, 4.0, 4.0, 1.0, 2.0]]),
        )
        self.assertFalse(bool(result["selected_goal_valid"][0]))
        self.assertFalse(bool(result["execution_valid"][0]))
        torch.testing.assert_close(result["pred_goals"], torch.zeros(1, 2))
        torch.testing.assert_close(result["pred_trajs"], torch.zeros(1, 12, 2))

    def test_trajectory_safety_checks_intermediate_points_not_only_endpoint(self):
        env = torch.ones(1, 4, 4)
        env[0, 2, 2] = 2.0
        params = torch.tensor([[-2.0, -2.0, 4.0, 4.0, 1.0, 2.0]])
        trajectory = torch.tensor(
            [[[-1.5, 0.5], [0.5, 0.5], [1.5, 0.5]]]
        )

        classification = classify_map_points(
            trajectory,
            env,
            params,
            reject_unknown=True,
        )
        self.assertEqual(
            classification["point_safe_mask"][0].tolist(),
            [True, False, True],
        )
        self.assertFalse(bool(classification["point_safe_mask"].all(dim=1)[0]))

    def test_guided_top_k_keeps_second_trajectory_when_first_is_unsafe(self):
        model = SBertPlusFTModel.__new__(SBertPlusFTModel)
        torch.nn.Module.__init__(model)
        model.cfgs = SimpleNamespace(
            guidance_conditioned=True,
            obs_len=8,
            pred_len=12,
            goal_dim=2,
            view_range=20.0,
        )
        model.mgp_model = _FixedMGP(
            torch.tensor([[[1.5, 0.5], [-1.5, 0.5], [-2.5, -2.5]]])
        )
        model.tgp_model = _GoalAwareTGP()

        trajectory_tokens = torch.zeros(1, 58, 2)
        token_ids = torch.zeros(1, 58, dtype=torch.long)
        map_tokens = torch.zeros(1, 4, 256)
        map_ids = torch.zeros(1, 4, dtype=torch.long)
        env = torch.ones(1, 6, 6)
        env[0, 3, 3] = 2.0
        result = model.inference_guided_candidates(
            mgp_spatial_ids=trajectory_tokens,
            mgp_temporal_ids=token_ids,
            mgp_segment_ids=token_ids,
            mgp_attn_mask=token_ids,
            tgp_temporal_ids=token_ids,
            tgp_segment_ids=token_ids,
            tgp_attn_mask=token_ids,
            guidance_points=torch.tensor([[1.4, 0.5]]),
            env_spatial_ids=map_tokens,
            env_temporal_ids=map_ids,
            env_segment_ids=map_ids,
            env_attn_mask=map_ids,
            envs=env,
            envs_params=torch.tensor([[-3.0, -3.0, 6.0, 6.0, 1.0, 2.0]]),
            top_k=2,
        )

        self.assertEqual(
            result["guided_candidate_indices"][0].tolist(),
            [0, 1],
        )
        self.assertEqual(tuple(result["guided_pred_trajs"].shape), (1, 2, 12, 2))
        self.assertEqual(
            result["guided_trajectory_map_safe"][0].tolist(),
            [False, True],
        )
        self.assertEqual(
            result["guided_execution_valid"][0].tolist(),
            [False, True],
        )

    def test_guided_inference_stops_when_tgp_path_is_unsafe(self):
        model = SBertPlusFTModel.__new__(SBertPlusFTModel)
        torch.nn.Module.__init__(model)
        model.cfgs = SimpleNamespace(
            guidance_conditioned=True,
            obs_len=8,
            pred_len=12,
            goal_dim=2,
            view_range=20.0,
        )
        model.mgp_model = _FixedMGP(torch.tensor([[[0.5, 0.5]]]))
        model.tgp_model = _MovingTGP()

        trajectory_tokens = torch.zeros(1, 58, 2)
        token_ids = torch.zeros(1, 58, dtype=torch.long)
        map_tokens = torch.zeros(1, 4, 256)
        map_ids = torch.zeros(1, 4, dtype=torch.long)
        env = torch.ones(1, 4, 4)
        env[0, 3, 3] = 2.0
        result = model.inference_guided(
            mgp_spatial_ids=trajectory_tokens,
            mgp_temporal_ids=token_ids,
            mgp_segment_ids=token_ids,
            mgp_attn_mask=token_ids,
            tgp_temporal_ids=token_ids,
            tgp_segment_ids=token_ids,
            tgp_attn_mask=token_ids,
            guidance_points=torch.tensor([[0.5, 0.5]]),
            env_spatial_ids=map_tokens,
            env_temporal_ids=map_ids,
            env_segment_ids=map_ids,
            env_attn_mask=map_ids,
            envs=env,
            envs_params=torch.tensor([[-2.0, -2.0, 4.0, 4.0, 1.0, 2.0]]),
        )
        self.assertTrue(bool(result["selected_goal_valid"][0]))
        self.assertFalse(bool(result["trajectory_map_safe"][0]))
        self.assertFalse(bool(result["execution_valid"][0]))
        torch.testing.assert_close(result["pred_trajs"], torch.zeros(1, 12, 2))

    def test_collision_loss_penalizes_unknown_occupied_and_out_of_bounds(self):
        env = torch.ones(1, 4, 4)
        env[0, 2, 2] = 2.0
        env[0, 2, 1] = 0.0
        params = torch.tensor([[-2.0, -2.0, 4.0, 4.0, 1.0, 2.0]])
        candidates = torch.tensor(
            [[[1.5, 0.5], [0.5, 0.5], [-0.5, 0.5], [2.5, 0.5]]]
        )
        collision_rate = goal_collision_loss(candidates, env, params)
        self.assertAlmostEqual(float(collision_rate), 0.8125)

    def test_collision_loss_pushes_out_of_bounds_goal_toward_map(self):
        env = torch.ones(1, 4, 4)
        params = torch.tensor([[-2.0, -2.0, 4.0, 4.0, 1.0, 2.0]])
        candidate = torch.tensor([[[2.5, 0.5]]], requires_grad=True)

        loss = goal_collision_loss(candidate, env, params)
        loss.backward()

        self.assertIsNotNone(candidate.grad)
        self.assertGreater(float(candidate.grad[0, 0, 0]), 0.0)

    def test_collision_loss_backpropagates_away_from_occupied_cell(self):
        env = torch.ones(1, 4, 4)
        env[0, 2, 2] = 2.0
        params = torch.tensor([[-2.0, -2.0, 4.0, 4.0, 1.0, 2.0]])
        candidate = torch.tensor([[[1.0, 0.5]]], requires_grad=True)

        loss = goal_collision_loss(candidate, env, params)
        loss.backward()

        self.assertGreater(float(loss), 0.0)
        self.assertIsNotNone(candidate.grad)
        self.assertLess(float(candidate.grad[0, 0, 0]), 0.0)

    def test_trajectory_collision_loss_has_prediction_gradient(self):
        env = torch.ones(1, 4, 4)
        env[0, 2, 2] = 2.0
        params = torch.tensor([[-2.0, -2.0, 4.0, 4.0, 1.0, 2.0]])
        trajectory = torch.tensor(
            [[[[1.0, 0.5], [1.5, 0.5]]]],
            requires_grad=True,
        )

        loss = pos_collision_loss(trajectory, env, params)
        loss.backward()

        self.assertGreater(float(trajectory.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()

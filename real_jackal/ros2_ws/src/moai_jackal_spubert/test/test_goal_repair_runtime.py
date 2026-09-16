"""Exercise raw-sample repair through runtime transforms, maps and bridge hold.

Neural/tensor execution is doubled; actual safety, runtime and bridge code runs.
"""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from moai_jackal_spubert.navigation_core import validate_candidate_path
from moai_jackal_spubert.rolling_laser_map import RollingLaserMapProvider
from test_bridge_stability import plan_tick, ready_bridge
import test_heading_stability as heading_fixture
from test_goal_candidate_repair import FakeTensor


class Decoder:
    def __init__(self, points):
        self.points = np.asarray(points, dtype=np.float32)
        self.hooks = {}

    def register_forward_hook(self, callback):
        key = object()
        self.hooks[key] = callback
        return NS(remove=lambda: self.hooks.pop(key))

    def __call__(self, hidden):
        output = FakeTensor(self.points)
        for callback in self.hooks.values():
            callback(self, (hidden,), output)
        return output


class Mgp:
    def __init__(self, raw, original):
        self.sbert_decoder = Decoder(raw)
        self.original = np.asarray(original, dtype=np.float32)

    def goal_predictor(self, hidden, k_sample, d_sample=0):
        self.sbert_decoder(hidden)
        return FakeTensor(self.original[None])


def free_map():
    provider = RollingLaserMapProvider(size_m=24, resolution=0.1, unknown_is_occupied=True)
    provider.origin_x = provider.origin_y = -12.0
    provider.grid.fill(1)
    provider.ready = True
    return provider


def classifier(goals, envs, params, *, reject_unknown=True):
    # Neutral model-cell test; the real world-footprint map remains authoritative.
    finite = np.isfinite(goals.numpy()[..., :2]).all(axis=-1)
    return {"point_safe_mask": FakeTensor(finite), "point_in_bounds_mask": FakeTensor(finite),
            "point_cell_values": FakeTensor(np.where(finite, 1.0, -1.0))}


def runtime_with_samples(raw, original, provider=None):
    fixture = heading_fixture.RuntimeHeadingOverrideTest()
    fixture.setUp()
    runtime = fixture.runtime
    runtime.goal_candidate_policy = "preserve_safe_samples"
    runtime.tgp_top_k = 5
    runtime.d_sample = len(raw)
    runtime.footprint_radius = 0.5
    runtime.map_provider = provider or free_map()
    runtime._classify_map_points = classifier
    runtime.torch.as_tensor = lambda value, dtype=None, device=None: FakeTensor(np.asarray(value, dtype=dtype))
    runtime.torch.bool = np.bool_
    mgp = Mgp(raw, original)

    def inference(**kwargs):
        points = mgp.goal_predictor(FakeTensor([[0.0]]), k_sample=len(original), d_sample=len(raw))
        safe = kwargs["candidate_safety_fn"](points).numpy()[0]
        safe &= runtime._classify_map_points(points, None, None)["point_safe_mask"].numpy()[0]
        points_np = points.numpy()[0]
        distance = np.linalg.norm(points_np[:, :2] - [8, 0], axis=-1)
        order = np.argsort(np.where(safe, distance, np.inf), kind="stable")[:min(5, len(original))]
        valid = safe[order]
        indices = np.where(valid, order, -1)
        goals = np.where(valid[:, None], points_np[order], 0)
        paths = goals[:, None, :2] * np.arange(1, 13)[None, :, None] / 12
        values = {
            "candidate_goals": points_np[None], "candidate_safe_mask": safe[None],
            "guided_pred_trajs": paths[None], "guided_candidate_goals": goals[None],
            "guided_candidate_indices": indices[None], "guided_candidate_goal_valid": valid[None],
            "guided_candidate_guidance_distances": np.where(valid, distance[order], np.inf)[None],
            "guided_trajectory_map_safe": np.ones((1, len(order)), dtype=bool),
            "guided_execution_valid": valid[None],
        }
        return {key: FakeTensor(value) for key, value in values.items()}

    runtime.model = NS(mgp_model=mgp, inference_guided_candidates=inference)
    inputs = dict(robot_history=[(0, 0)] * 8, robot_yaw=0, human_histories={},
                  final_goal=(8, 0), guidance_point_world=(8, 0), heading_override_rad=0)
    return runtime, inputs


def test_unsafe_means_recover_safe_raw_endpoints_but_cross_wall_paths_still_fail():
    provider = free_map()
    wall_col, _ = provider._world_to_cell(2, 0)
    provider.grid[:, wall_col] = 2
    runtime, inputs = runtime_with_samples([(1, -.2), (3, -.2), (1, .2), (3, .2)],
                                           [(2, -.2), (2, .2)], provider)
    results = runtime.predict_candidates(**inputs)
    diag = runtime.last_input_diagnostics['goal_sampling']
    assert diag['original_safe_count'] == 0
    assert diag['raw_safe_count'] == 4
    assert diag['output_safe_count'] == 2
    assert len(results) == 2 and all(r.selected_goal_valid for r in results)
    assert len({r.candidate_index for r in results}) == 2
    assert diag['original']['rejection_counts'] == {'footprint_occupied': 2}
    # A repaired endpoint beyond a wall must not be treated as an executable path.
    across = next(r for r in results if r.selected_goal_world[0] == 3)
    check = validate_candidate_path(
        current=(0, 0), path=across.path_world, final_goal=(8, 0),
        map_provider=provider, footprint_radius=.5, prediction_dt=.4,
        maximum_model_speed=1.5, maximum_step_ratio=1.5, minimum_goal_progress=.1,
        human_histories={}, human_sample_dt=.4, minimum_human_center_distance=1.2,
        human_radius=.35, human_safety_margin=.2,
    )
    assert check.reason == 'robot_footprint_collision'
    assert runtime.model.mgp_model.sbert_decoder.hooks == {}
    assert 'goal_predictor' not in runtime.model.mgp_model.__dict__


def test_all_unsafe_samples_keep_no_map_safe_goal_and_bridge_hold():
    provider = free_map()
    col, _ = provider._world_to_cell(2, 0)
    provider.grid[:, col] = 2
    runtime, inputs = runtime_with_samples([(2, -.2), (2, .2)], [(2, 0)], provider)
    results = runtime.predict_candidates(**inputs)
    assert runtime.last_input_diagnostics['goal_sampling']['repaired_count'] == 0
    assert not results[0].selected_goal_valid
    node = ready_bridge()
    node._runtime.candidates = results
    plan_tick(node)
    assert json.loads(node.plan_context_pub.messages[-1].data)['state'] == 'hold'
    assert node.path_pub.messages[-1].poses == []
    assert json.loads(node._diagnostics_file.getvalue().splitlines()[-1])['attempts'][0]['reason'] == 'no_map_safe_goal'


def test_safe_mean_between_unsafe_raw_samples_is_preserved():
    provider = free_map()
    for y in (-1, 1):
        _, row = provider._world_to_cell(2, y)
        provider.grid[row, :] = 2
    runtime, inputs = runtime_with_samples([(2, -1), (2, 1)], [(2, 0)], provider)
    result = runtime.predict_candidates(**inputs)[0]
    diag = runtime.last_input_diagnostics['goal_sampling']
    assert diag['raw_safe_count'] == 0
    assert diag['original_safe_count'] == diag['output_safe_count'] == 1
    assert diag['repaired_count'] == 0
    assert result.selected_goal_world == (2, 0)


def test_safety_diagnostics_separate_model_crop_unknown_and_world_footprint():
    runtime, _ = runtime_with_samples([(0, 0)], [(0, 0)])
    col, row = runtime.map_provider._world_to_cell(4.2, 0)
    runtime.map_provider.grid[row, col] = 0
    runtime._classify_map_points = lambda *a, **kw: {
        'point_safe_mask': FakeTensor([[False, False, False, True, False]]),
        'point_in_bounds_mask': FakeTensor([[True, True, False, True, False]]),
        'point_cell_values': FakeTensor([[0, 2, -1, 1, -1]]),
    }
    safe, details = runtime._goal_safety_snapshot(
        FakeTensor([[(1, 0), (2, 0), (3, 0), (4, 0), (np.nan, 0)]]),
        batch={'envs': None, 'envs_params': None}, origin=(0, 0), theta=0,
    )
    assert not safe.any()
    assert details['rejection_reasons'] == [
        ['unknown_model_map'], ['occupied_model_map'], ['outside_model_map'],
        ['footprint_unknown'], ['nonfinite'],
    ]


def test_legacy_mode_keeps_original_unsafe_representatives_for_comparison():
    provider = free_map()
    col, _ = provider._world_to_cell(2, 0)
    provider.grid[:, col] = 2
    runtime, inputs = runtime_with_samples([(1, 0), (3, 0)], [(2, 0)], provider)
    runtime.goal_candidate_policy = 'legacy'
    assert not runtime.predict_candidates(**inputs)[0].selected_goal_valid
    assert runtime.last_input_diagnostics['goal_sampling']['policy'] == 'legacy'


def test_model_recheck_disagreement_is_rejected_not_overridden():
    runtime, inputs = runtime_with_samples([(1, 0), (2, 0)], [(1.5, 0)])
    original = runtime.model.inference_guided_candidates
    def disagree(**kwargs):
        output = original(**kwargs)
        output['candidate_safe_mask'] = FakeTensor([[False]])
        return output
    runtime.model.inference_guided_candidates = disagree
    with pytest.raises(RuntimeError, match='model safety recheck'):
        runtime.predict_candidates(**inputs)
    assert 'goal_predictor' not in runtime.model.mgp_model.__dict__


def test_marker_indices_keep_original_slots_when_nonfinite_points_are_hidden():
    runtime, inputs = runtime_with_samples([(1, 0)], [(np.nan, 0), (1, 0)])
    runtime.goal_candidate_policy = 'legacy'
    result = runtime.predict_candidates(**inputs)[0]
    assert result.candidate_index == 1
    assert result.candidate_goal_indices == (1,)
    assert result.candidate_goals_world == [(1, 0)]

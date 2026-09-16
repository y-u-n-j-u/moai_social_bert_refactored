"""Safe-centroid preservation and scoped decoder capture, without torch/ROS."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from moai_jackal_spubert.goal_candidate_repair import (
    capture_and_repair_goal_samples,
    repair_goal_candidates,
)


class FakeTensor:
    """Small tensor protocol used by the adapter and runtime integration tests."""
    def __init__(self, values, *, dtype=None, device="cpu"):
        self.array = np.asarray(values, dtype=dtype)
        self.device = device

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def dtype(self):
        return self.array.dtype

    def detach(self):
        return self

    def clone(self):
        return FakeTensor(self.array.copy(), device=self.device)

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return FakeTensor(self.array.reshape(*shape), device=self.device)

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]

    def __len__(self):
        return len(self.array)

    def __getitem__(self, key):
        return FakeTensor(self.array[key], device=self.device)


class FakeHookDecoder:
    def __init__(self, raw_points):
        raw = np.asarray(raw_points, dtype=np.float32)
        self.output = FakeTensor(raw.reshape(-1, raw.shape[-1]))
        self.hooks = []
        self.calls = 0

    def register_forward_hook(self, callback):
        self.hooks.append(callback)
        return NS(remove=lambda: self.hooks.remove(callback))

    def __call__(self, value):
        self.calls += 1
        output = self.output
        for callback in tuple(self.hooks):
            replacement = callback(self, (value,), output)
            if replacement is not None:
                output = replacement
        return output


class FakeMgp:
    def __init__(self, raw_points, original_points):
        self.sbert_decoder = FakeHookDecoder(raw_points)
        original = np.asarray(original_points, dtype=np.float32)
        self.original = FakeTensor(original[None] if original.ndim == 2 else original)
        self.cfgs = NS(goal_dim=original.shape[-1])
        self.calls = []
        self.fail_after_decoder = False
        self.rng = np.random.default_rng(20260916)

    def goal_predictor(self, pred_goal_h, k_sample, d_sample=0):
        self.calls.append((k_sample, d_sample))
        self.rng.normal(size=max(k_sample, d_sample))
        self.sbert_decoder(pred_goal_h)
        if self.fail_after_decoder:
            raise RuntimeError("original predictor failed")
        return self.original.clone()


def prediction_inputs():
    return FakeTensor(np.zeros((1, 8), dtype=np.float32))


def test_only_unsafe_slots_change_and_replacements_are_actual_safe_samples():
    raw = np.array([[-0.8, 0], [0.8, 0], [2, 0], [9, 9]], dtype=np.float32)
    raw_safe = np.array([True, True, True, False])
    original = np.array([[0, 0], [2, 0], [5, 0]], dtype=np.float32)
    original_safe = np.array([False, True, False])
    before = tuple(value.copy() for value in (raw, raw_safe, original, original_safe))
    repaired, metadata = repair_goal_candidates(raw, raw_safe, original, original_safe, [3, 0])
    np.testing.assert_array_equal(repaired[1], original[1])
    assert repaired.dtype == original.dtype
    assert repaired.shape == original.shape
    assert {item["slot"] for item in metadata["replacement_sources"]} == {0, 2}
    for item in metadata["replacement_sources"]:
        point = repaired[item["slot"]]
        assert any(np.array_equal(point, sample) for sample in raw[raw_safe])
        assert not np.array_equal(point, original[1])
    for actual, expected in zip((raw, raw_safe, original, original_safe), before):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("safe_index", [0, 1, 2, 3])
def test_one_safe_raw_sample_is_retained_once_not_duplicated_to_fill_slots(safe_index):
    raw = np.array([[0, 1], [1, 1], [2, 1], [3, 1]], dtype=np.float32)
    raw_safe = np.arange(4) == safe_index
    original = np.array([[8, 8], [9, 9], [10, 10]], dtype=np.float32)
    repaired, metadata = repair_goal_candidates(raw, raw_safe, original, [False] * 3, [0, 0])
    assert len(metadata["replacement_sources"]) == 1
    assert sum(np.array_equal(point, raw[safe_index]) for point in repaired) == 1
    assert sum(np.array_equal(a, b) for a, b in zip(repaired, original)) == 2


def test_safe_original_centroid_is_kept_even_when_every_raw_sample_is_unsafe():
    original = np.array([[0, 0], [1, 0]], dtype=np.float32)
    repaired, metadata = repair_goal_candidates(
        [[-1, 0], [1, 0]], [False, False], original, [True, False], [4, 0],
    )
    np.testing.assert_array_equal(repaired, original)
    assert metadata["replacement_sources"] == []


def test_all_unsafe_remains_unchanged_without_a_fabricated_origin_or_safe_candidate():
    original = np.array([[8, 8], [9, 9]], dtype=np.float32)
    repaired, metadata = repair_goal_candidates(
        [[6, 6], [7, 7]], [False, False], original, [False, False], [1, 0],
    )
    np.testing.assert_array_equal(repaired, original)
    assert metadata["replacement_sources"] == []


def test_nonfinite_raw_coordinates_are_never_used_even_if_mask_claims_safe():
    raw = np.array([[np.nan, 0], [0, np.inf], [-np.inf, 0], [1, 1]], dtype=np.float32)
    original = np.array([[8, 8], [9, 9]], dtype=np.float32)
    repaired, metadata = repair_goal_candidates(raw, [True] * 4, original, [False] * 2, [2, 2])
    assert np.isfinite(repaired).all()
    assert len(metadata["replacement_sources"]) == 1
    assert any(np.array_equal(point, [1, 1]) for point in repaired)


def test_near_duplicate_raw_points_do_not_create_false_diversity():
    raw = np.array([[1, 0], [1, 0], [1 + 5e-7, 0], [2, 0], [3, 0]], dtype=np.float64)
    original = np.array([[1, 0], [8, 8], [9, 9]], dtype=np.float64)
    repaired, metadata = repair_goal_candidates(raw, [True] * 5, original, [True, False, False], [0, 0])
    assert {tuple(point) for point in repaired} == {(1, 0), (2, 0), (3, 0)}
    assert len(metadata["replacement_sources"]) == 2


def test_repair_is_deterministic_and_does_not_consume_numpy_rng():
    raw = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.float32)
    original = np.array([[8, 8], [9, 9]], dtype=np.float32)
    saved_rng = np.random.get_state()
    try:
        np.random.seed(17)
        expected = np.random.random()
        np.random.seed(17)
        first = repair_goal_candidates(raw, [True] * 4, original, [False] * 2, [0, 0])
        actual = np.random.random()
        second = repair_goal_candidates(raw, [True] * 4, original, [False] * 2, [0, 0])
    finally:
        np.random.set_state(saved_rng)
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1] == second[1]
    assert actual == expected


@pytest.mark.parametrize("instance_override", [False, True])
def test_scoped_capture_calls_original_once_preserves_rng_and_restores_instance(instance_override):
    raw = [[-1, 0], [1, 0], [2, 0], [3, 0]]
    original = [[0, 0], [2, 0]]
    model, baseline = FakeMgp(raw, original), FakeMgp(raw, original)
    if instance_override:
        model.goal_predictor = model.goal_predictor
    before_predictor, before_attributes = model.goal_predictor, set(model.__dict__)
    captured = []

    def repair(raw_tensor, original_tensor):
        assert model.sbert_decoder.hooks == []
        captured.append((raw_tensor.clone(), original_tensor.clone()))
        result = original_tensor.clone()
        result.array[0, 0] = raw_tensor.array[0, 0]
        return result

    with capture_and_repair_goal_samples(model, repair):
        output = model.goal_predictor(prediction_inputs(), k_sample=2, d_sample=4)
    baseline.goal_predictor(prediction_inputs(), k_sample=2, d_sample=4)
    assert model.calls == [(2, 4)]
    assert model.sbert_decoder.calls == 1
    assert model.rng.normal() == baseline.rng.normal()
    assert len(captured) == 1
    assert captured[0][0].shape == (1, 4, 2)
    np.testing.assert_array_equal(captured[0][0].numpy()[0], raw)
    np.testing.assert_array_equal(captured[0][1].numpy()[0], original)
    np.testing.assert_array_equal(output.numpy()[0], [[-1, 0], [2, 0]])
    np.testing.assert_array_equal(model.original.numpy()[0], original)
    assert model.goal_predictor == before_predictor
    assert set(model.__dict__) == before_attributes
    assert model.sbert_decoder.hooks == []


def test_second_call_is_rejected_before_resampling_and_adapter_is_instance_local():
    model = FakeMgp([[-1, 0], [1, 0]], [[0, 0]])
    other = FakeMgp([[-2, 0], [2, 0]], [[0, 0]])
    before, other_before = model.goal_predictor, other.goal_predictor
    callbacks = []

    def repair(raw, original):
        callbacks.append(raw)
        return original

    with pytest.raises(RuntimeError, match="exactly one predictor call"):
        with capture_and_repair_goal_samples(model, repair):
            other.goal_predictor(prediction_inputs(), 1, 2)
            assert callbacks == []
            model.goal_predictor(prediction_inputs(), 1, 2)
            model.goal_predictor(prediction_inputs(), 1, 2)
    assert model.calls == [(1, 2)]
    assert other.calls == [(1, 2)]
    assert len(callbacks) == 1
    assert model.goal_predictor == before
    assert other.goal_predictor == other_before
    assert model.sbert_decoder.hooks == []


def test_wrong_raw_decoder_shape_is_rejected_before_callback_and_hook_is_removed():
    model = FakeMgp([[-1, 0], [1, 0], [2, 0]], [[0, 0]])
    before, callbacks = model.goal_predictor, []
    with pytest.raises(RuntimeError, match="raw shape"):
        with capture_and_repair_goal_samples(model, lambda raw, original: callbacks.append(raw)):
            model.goal_predictor(prediction_inputs(), 1, 4)
    assert callbacks == []
    assert model.goal_predictor == before
    assert model.sbert_decoder.hooks == []


@pytest.mark.parametrize("failure", ["body", "predictor", "repair", "no_call"])
def test_scoped_capture_removes_hook_and_restores_predictor_after_failures(failure):
    model = FakeMgp([[-1, 0], [1, 0]], [[0, 0]])
    before_predictor, before_attributes = model.goal_predictor, set(model.__dict__)

    def repair(raw, original):
        if failure == "repair":
            raise RuntimeError("repair callback failed")
        return original

    with pytest.raises(RuntimeError):
        with capture_and_repair_goal_samples(model, repair):
            if failure == "body":
                raise RuntimeError("context body failed")
            if failure == "predictor":
                model.fail_after_decoder = True
            if failure != "no_call":
                model.goal_predictor(prediction_inputs(), 1, 2)
    assert model.goal_predictor == before_predictor
    assert set(model.__dict__) == before_attributes
    assert model.sbert_decoder.hooks == []


@pytest.mark.parametrize("bad_output", ["shape", "dtype", "device"])
def test_adapter_rejects_repair_that_changes_tensor_contract(bad_output):
    model = FakeMgp([[-1, 0], [1, 0]], [[0, 0]])
    before = model.goal_predictor

    def repair(raw, original):
        if bad_output == "shape":
            return raw
        if bad_output == "dtype":
            return FakeTensor(original.array, dtype=np.float64)
        return FakeTensor(original.array, device="wrong_device")

    with pytest.raises((ValueError, RuntimeError)):
        with capture_and_repair_goal_samples(model, repair):
            model.goal_predictor(prediction_inputs(), 1, 2)
    assert model.goal_predictor == before
    assert model.sbert_decoder.hooks == []

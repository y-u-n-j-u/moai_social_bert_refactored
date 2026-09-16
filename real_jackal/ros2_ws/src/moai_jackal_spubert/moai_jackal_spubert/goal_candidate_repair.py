"""Recover safe decoded goals without changing the learned sampler or K-means.

The selector preserves every originally safe centroid and fills only unsafe
slots with distinct, already classified safe raw samples.  It does not decide
whether a trajectory may execute: the normal goal and trajectory safety checks
must still run on its output.

The scoped adapter observes one decoder call inside the original predictor.
It neither samples again nor replaces the original clustering operation, so
the base model's computation and random-number consumption remain unchanged.
Use the adapter only from the runtime's single model worker.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import inspect
import operator
from typing import Any, Callable, Dict, Iterator, Tuple

import numpy as np


def _points(value: Any, name: str) -> np.ndarray:
    points = np.asarray(value)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"{name} must have shape (count, dimensions >= 2)")
    if not np.issubdtype(points.dtype, np.number) or np.iscomplexobj(points):
        raise ValueError(f"{name} must contain real numeric coordinates")
    return points


def _safe_mask(value: Any, count: int, name: str) -> np.ndarray:
    mask = np.asarray(value)
    if mask.shape != (count,) or mask.dtype.kind != "b":
        raise ValueError(f"{name} must be a Boolean array of shape ({count},)")
    return mask


def repair_goal_candidates(
    raw_points: Any,
    raw_safe: Any,
    original_points: Any,
    original_safe: Any,
    guidance_xy: Any,
    *,
    duplicate_tolerance_m: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Keep safe centroids and replace unsafe slots with safe raw points.

    Inputs use one common metric coordinate frame.  Safety masks must be
    Boolean arrays supplied by the existing map/footprint checks.  Nonfinite
    points are treated as unsafe even when their mask is True.  The result is
    a copy; no input array or safety flag is modified.

    The first replacement is nearest to the guidance point.  Later choices
    maximize their minimum XY distance to all retained/repaired safe points,
    breaking ties by guidance distance and then raw index.  Raw points within
    ``duplicate_tolerance_m`` (inclusive) of an existing safe point are skipped.
    Existing safe duplicates are preserved.  When distinct safe raw samples
    run out, unfilled unsafe slots retain their original values.
    """
    raw = _points(raw_points, "raw_points")
    original = _points(original_points, "original_points")
    if raw.shape[1] != original.shape[1]:
        raise ValueError("raw and original goal dimensions must match")
    raw_mask = _safe_mask(raw_safe, len(raw), "raw_safe")
    original_mask = _safe_mask(original_safe, len(original), "original_safe")
    guidance = np.asarray(guidance_xy, dtype=np.float64)
    if guidance.shape != (2,) or not np.isfinite(guidance).all():
        raise ValueError("guidance_xy must contain two finite coordinates")
    tolerance = float(duplicate_tolerance_m)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("duplicate_tolerance_m must be finite and nonnegative")

    raw_mask = raw_mask & np.isfinite(raw).all(axis=1)
    original_mask = original_mask & np.isfinite(original).all(axis=1)
    output = original.astype(np.result_type(raw.dtype, original.dtype, np.float32), copy=True)
    raw_xy = raw[:, :2].astype(np.float64, copy=False)
    retained_xy = [point.copy() for point in output[original_mask, :2].astype(np.float64)]
    remaining = [int(index) for index in np.flatnonzero(raw_mask)]
    replacements = []

    def guidance_distance(index: int) -> float:
        delta = raw_xy[index] - guidance
        return float(np.hypot(delta[0], delta[1]))

    def separation(index: int) -> float:
        if not retained_xy:
            return float("inf")
        differences = np.asarray(retained_xy) - raw_xy[index]
        return float(np.min(np.hypot(differences[:, 0], differences[:, 1])))

    for slot in np.flatnonzero(~original_mask):
        # Recheck after every insertion so near-identical raw samples never
        # consume multiple repaired slots or duplicate a retained centroid.
        remaining = [index for index in remaining if separation(index) > tolerance]
        if not remaining:
            break
        if not replacements:
            selected = min(remaining, key=lambda index: (guidance_distance(index), index))
        else:
            selected = min(
                remaining,
                key=lambda index: (-separation(index), guidance_distance(index), index),
            )
        output[int(slot)] = raw[selected]
        retained_xy.append(raw_xy[selected].copy())
        remaining.remove(selected)
        replacements.append({"slot": int(slot), "raw_index": int(selected)})

    original_safe_count = int(np.count_nonzero(original_mask))
    metadata = {
        "raw_count": int(len(raw)),
        "original_count": int(len(original)),
        "repaired_count": int(len(replacements)),
        "original_safe_count": original_safe_count,
        "output_safe_count": original_safe_count + int(len(replacements)),
        "raw_safe_count": int(np.count_nonzero(raw_mask)),
        "replacement_sources": replacements,
    }
    return output, metadata


def _sample_count(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer sample count")
    try:
        count = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer sample count") from exc
    if count < (0 if allow_zero else 1):
        raise ValueError(f"{name} has an invalid sample count")
    return count


def _tensor_shape(value: Any, name: str) -> tuple:
    if not all(hasattr(value, attribute) for attribute in ("shape", "dtype", "device", "detach", "clone", "reshape")):
        raise RuntimeError(f"{name} must be a tensor with shape, dtype and device")
    return tuple(value.shape)


@contextmanager
def capture_and_repair_goal_samples(
    mgp_model: Any,
    repair_callback: Callable[[Any, Any], Any],
) -> Iterator[None]:
    """Adapt exactly one original goal prediction, restoring all hooks/state.

    ``repair_callback(raw, original)`` receives tensors shaped ``(B,D,dim)``
    and ``(B,K,dim)`` and must return a tensor matching the original shape,
    dtype and device.  It runs even when no replacement is needed, allowing
    diagnostics to describe every successful prediction.  A missing decoder
    capture, changed model contract or bypassed predictor raises an error.

    This per-instance adapter is intentionally non-nested and is for inference
    on a single model worker only.  It does not patch model classes, weights,
    configuration, clustering, decoder outputs, or random-number generators.
    """
    original_predictor = getattr(mgp_model, "goal_predictor", None)
    decoder = getattr(mgp_model, "sbert_decoder", None)
    if not callable(original_predictor) or not callable(repair_callback):
        raise RuntimeError("goal predictor and repair callback must be callable")
    if getattr(original_predictor, "_captures_goal_samples", False):
        raise RuntimeError("goal sample repair contexts cannot be nested")
    if not callable(getattr(decoder, "register_forward_hook", None)):
        raise RuntimeError("MGP decoder does not support forward hooks")
    try:
        signature = inspect.signature(original_predictor)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("cannot inspect the MGP goal predictor signature") from exc
    if not {"k_sample", "d_sample"}.issubset(signature.parameters):
        raise RuntimeError("MGP goal predictor must expose k_sample and d_sample")

    had_instance_attribute = "goal_predictor" in vars(mgp_model)
    instance_attribute = vars(mgp_model).get("goal_predictor")
    call_count = 0
    completed_count = 0

    @wraps(original_predictor)
    def wrapped_predictor(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count, completed_count
        call_count += 1
        if call_count != 1:
            raise RuntimeError("goal sample repair expects exactly one predictor call")
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        k_sample = _sample_count(bound.arguments["k_sample"], "k_sample")
        d_sample = max(k_sample, _sample_count(bound.arguments["d_sample"], "d_sample", allow_zero=True))
        captures = []

        def capture_decoder_output(module: Any, inputs: Any, output: Any) -> None:
            _tensor_shape(output, "MGP decoder output")
            captures.append(output.detach().clone())
            return None

        handle = decoder.register_forward_hook(capture_decoder_output)
        try:
            # Preserve the base sampler and clustering, including RNG state.
            original_goals = original_predictor(*args, **kwargs)
        finally:
            handle.remove()
        if len(captures) != 1:
            raise RuntimeError("expected exactly one MGP decoder output capture")
        original_shape = _tensor_shape(original_goals, "original goals")
        if len(original_shape) != 3 or original_shape[0] < 1 or original_shape[1] != k_sample or original_shape[2] < 2:
            raise RuntimeError("MGP goal predictor returned an incompatible goal shape")
        batch_size, _, dimensions = original_shape
        raw = captures[0]
        if _tensor_shape(raw, "raw goals") != (batch_size * d_sample, dimensions):
            raise RuntimeError("captured MGP decoder output has an incompatible raw shape")
        if raw.dtype != original_goals.dtype or raw.device != original_goals.device:
            raise RuntimeError("raw and original goals must share dtype and device")
        raw = raw.reshape(batch_size, d_sample, dimensions)
        repaired = repair_callback(raw, original_goals)
        if _tensor_shape(repaired, "repaired goals") != original_shape:
            raise RuntimeError("repaired goals must preserve the original goal shape")
        if repaired.dtype != original_goals.dtype or repaired.device != original_goals.device:
            raise RuntimeError("repaired goals must preserve the original dtype and device")
        completed_count += 1
        return repaired

    wrapped_predictor._captures_goal_samples = True
    setattr(mgp_model, "goal_predictor", wrapped_predictor)
    try:
        yield
    finally:
        if had_instance_attribute:
            setattr(mgp_model, "goal_predictor", instance_attribute)
        else:
            delattr(mgp_model, "goal_predictor")
    if call_count != 1 or completed_count != 1:
        raise RuntimeError("goal sample repair requires one completed predictor/callback call")

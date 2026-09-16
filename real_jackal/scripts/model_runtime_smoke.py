#!/usr/bin/env python3
"""Load the deployed checkpoint and execute one complete guided inference."""

import hashlib
import logging
import math
import os

from moai_jackal_spubert.guided_spubert_runtime import GuidedSpubertRuntime
from moai_jackal_spubert.heading_stability import HeadingSelector
from moai_jackal_spubert.rolling_laser_map import RollingLaserMapProvider


MODEL_ROOT = os.environ.get("MOAI_MODEL_ROOT", "/root/moai_social_bert_refactored")


def model_state_sha256(model, torch_module):
    """Hash parameter/buffer contents without changing or retaining tensors."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch_module.uint8)
        digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def sampling_state(runtime):
    mgp = runtime.model.mgp_model
    method = mgp.goal_predictor
    return {
        "k_values": (int(runtime.args.k_sample), int(runtime.model.cfgs.k_sample),
                     int(mgp.cfgs.k_sample)),
        "d_sample": int(runtime.d_sample),
        "top_k": int(runtime.tgp_top_k),
        "method_function": getattr(method, "__func__", method),
        "method_owner": getattr(method, "__self__", None),
        "instance_override": "goal_predictor" in vars(mgp),
        "instance_value": vars(mgp).get("goal_predictor"),
        "decoder_hooks": tuple((key, id(hook)) for key, hook in mgp.sbert_decoder._forward_hooks.items()),
    }


def check_sampling_restored(before, runtime):
    after = sampling_state(runtime)
    for name in ("k_values", "d_sample", "top_k", "instance_override", "decoder_hooks"):
        if before[name] != after[name]:
            raise RuntimeError(f"goal candidate wrapper changed persistent sampling state: {name}")
    for name in ("method_function", "method_owner", "instance_value"):
        if before[name] is not after[name]:
            raise RuntimeError(f"goal candidate wrapper did not restore {name}")


def check_goal_sampling(diagnostics):
    sampling = diagnostics.get("goal_sampling")
    if not isinstance(sampling, dict) or sampling.get("policy") != "preserve_safe_samples":
        raise RuntimeError("checkpoint smoke did not exercise preserve_safe_samples goal sampling")
    names = ("raw_count", "original_count", "repaired_count", "original_safe_count",
             "output_safe_count", "raw_safe_count")
    for name in names:
        if type(sampling.get(name)) is not int or sampling[name] < 0:
            raise RuntimeError(f"goal sampling diagnostic {name} is missing or invalid")
    raw, original = sampling["raw_count"], sampling["original_count"]
    repaired = sampling["repaired_count"]
    original_safe, output_safe = sampling["original_safe_count"], sampling["output_safe_count"]
    raw_safe = sampling["raw_safe_count"]
    if (raw, original) != (40, 20):
        raise RuntimeError(f"social005 smoke requires 40 raw samples and 20 representatives; got {raw}/{original}")
    if original_safe > original or output_safe > original or raw_safe > raw:
        raise RuntimeError("goal sampling safe counts exceed their candidate counts")
    if output_safe < original_safe or output_safe != original_safe + repaired:
        raise RuntimeError("goal repair lost safe representatives or reported inconsistent replacements")
    if raw_safe > 0 and original_safe == 0 and output_safe == 0:
        raise RuntimeError("goal repair discarded all safe raw samples when every centroid was unsafe")
    if repaired > 0 and raw_safe == 0:
        raise RuntimeError("goal repair reported a replacement without a safe raw sample")
    sources = sampling.get("replacement_sources")
    if not isinstance(sources, list) or len(sources) != repaired:
        raise RuntimeError("goal sampling replacement_sources does not match repaired_count")
    slots = set()
    for source in sources:
        if not isinstance(source, dict):
            raise RuntimeError("goal sampling replacement source must identify a slot and raw_index")
        slot, raw_index = source.get("slot"), source.get("raw_index")
        if (type(slot) is not int or type(raw_index) is not int
                or not 0 <= slot < original or not 0 <= raw_index < raw or slot in slots):
            raise RuntimeError("goal sampling replacement indices are invalid or duplicate a slot")
        slots.add(slot)
    return sampling


def check_default_heading_selector():
    """Exercise the deployed default at rest and at reliable moving speed."""
    selector = HeadingSelector()
    if selector.config.mode != "motion_guarded":
        raise RuntimeError("runtime smoke expects the motion_guarded deployment default")
    stationary = selector.select(
        history_xy=[(0.0, 0.0), (0.0, 0.0009), (0.0, 0.0020)],
        sample_times_s=[0.0, 0.4, 0.8],
        odom_yaw_rad=0.3,
    )
    if not stationary.source.startswith("odom_yaw") or not math.isclose(stationary.theta_rad, 0.3):
        raise RuntimeError("millimetre noise did not fall back to odometry yaw")
    selector.reset()
    angle = math.radians(10.0)
    history = [((index - 7) * 0.08 * math.cos(angle),
                (index - 7) * 0.08 * math.sin(angle)) for index in range(8)]
    moving = selector.select(
        history_xy=history,
        sample_times_s=[0.4 * index for index in range(8)],
        odom_yaw_rad=0.0,
    )
    if moving.source != "guarded_recent_motion" or not math.isclose(moving.theta_rad, angle):
        raise RuntimeError("reliable 0.2 m/s motion did not preserve the last-segment heading")
    return history, moving


def main() -> None:
    logger = logging.getLogger("model_runtime_smoke")
    logging.basicConfig(level=logging.INFO)
    robot_history, heading = check_default_heading_selector()

    map_provider = RollingLaserMapProvider(
        size_m=24.0,
        resolution=0.10,
        free_gap_fill_m=0.20,
        unknown_is_occupied=True,
    )
    ray_count = 720
    map_provider.update_from_scan(
        robot_x=0.0,
        robot_y=0.0,
        robot_yaw=0.0,
        ranges=[math.inf] * ray_count,
        angle_min=-math.pi,
        angle_increment=2.0 * math.pi / ray_count,
        range_min=0.35,
        range_max=30.0,
        free_ray_limit_m=11.5,
    )

    runtime = GuidedSpubertRuntime(
        repo_path=MODEL_ROOT,
        config_path=os.path.join(
            MODEL_ROOT,
            "configs/spubert/moai_social_nav_route_gp_continue_b21_col01_social005.yaml",
        ),
        checkpoint_path=os.path.join(
            MODEL_ROOT,
            "output/spubert_route_gp_continue_b21_col01_social005/model_best.pth",
        ),
        map_provider=map_provider,
        use_cuda=False,
        d_sample=40,
        runtime_seed=21,
        guidance_radius=8.0,
        tgp_top_k=5,
        footprint_radius=0.50,
        goal_candidate_policy="preserve_safe_samples",
        logger=logger,
    )
    before_sampling = sampling_state(runtime)
    if before_sampling["k_values"] != (20, 20, 20):
        raise RuntimeError(f"social005 checkpoint smoke expects unchanged k_sample=20: {before_sampling['k_values']}")
    if (before_sampling["d_sample"], before_sampling["top_k"]) != (40, 5):
        raise RuntimeError("social005 checkpoint smoke expects d_sample=40 and tgp_top_k=5")
    before_state_sha = model_state_sha256(runtime.model, runtime.torch)
    candidates = runtime.predict_candidates(
        robot_history=robot_history,
        robot_yaw=0.0,
        human_histories={},
        final_goal=(8.0 * math.cos(heading.theta_rad), 8.0 * math.sin(heading.theta_rad)),
        guidance_point_world=(8.0 * math.cos(heading.theta_rad), 8.0 * math.sin(heading.theta_rad)),
        heading_override_rad=heading.theta_rad,
        heading_source=heading.source,
    )
    check_sampling_restored(before_sampling, runtime)
    if model_state_sha256(runtime.model, runtime.torch) != before_state_sha:
        raise RuntimeError("checkpoint parameters or buffers changed during scoped goal repair")
    diagnostics = runtime.last_input_diagnostics
    goal_sampling = check_goal_sampling(diagnostics)
    if not diagnostics.get("heading_override_used") or diagnostics.get("heading_source") != heading.source:
        raise RuntimeError("runtime smoke did not exercise the motion_guarded heading override")
    valid_count = sum(
        candidate.selected_goal_valid
        and candidate.trajectory_map_safe
        and candidate.execution_valid
        for candidate in candidates
    )
    if not candidates or valid_count == 0:
        raise RuntimeError(
            f"checkpoint inference returned no valid path: candidates={len(candidates)}"
        )
    print(
        "REAL JACKAL MODEL RUNTIME SMOKE: PASS "
        f"candidates={len(candidates)} valid={valid_count} "
        f"goal_policy={goal_sampling['policy']} raw={goal_sampling['raw_count']} "
        f"representatives={goal_sampling['original_count']} repaired={goal_sampling['repaired_count']} "
        f"safe={goal_sampling['original_safe_count']}->{goal_sampling['output_safe_count']} "
        f"model_state_sha256={before_state_sha}"
    )


if __name__ == "__main__":
    main()

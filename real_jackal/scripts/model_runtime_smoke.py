#!/usr/bin/env python3
"""Load the deployed checkpoint and execute one complete guided inference."""

import logging
import math
import os

from moai_jackal_spubert.guided_spubert_runtime import GuidedSpubertRuntime
from moai_jackal_spubert.heading_stability import HeadingSelector
from moai_jackal_spubert.rolling_laser_map import RollingLaserMapProvider


MODEL_ROOT = os.environ.get("MOAI_MODEL_ROOT", "/root/moai_social_bert_refactored")


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
        logger=logger,
    )
    candidates = runtime.predict_candidates(
        robot_history=robot_history,
        robot_yaw=0.0,
        human_histories={},
        final_goal=(8.0 * math.cos(heading.theta_rad), 8.0 * math.sin(heading.theta_rad)),
        guidance_point_world=(8.0 * math.cos(heading.theta_rad), 8.0 * math.sin(heading.theta_rad)),
        heading_override_rad=heading.theta_rad,
        heading_source=heading.source,
    )
    diagnostics = runtime.last_input_diagnostics
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
        f"candidates={len(candidates)} valid={valid_count}"
    )


if __name__ == "__main__":
    main()

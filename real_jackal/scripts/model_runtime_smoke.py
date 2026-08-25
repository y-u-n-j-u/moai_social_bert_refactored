#!/usr/bin/env python3
"""Load the deployed checkpoint and execute one complete guided inference."""

import logging
import math
import os

from moai_jackal_spubert.guided_spubert_runtime import GuidedSpubertRuntime
from moai_jackal_spubert.rolling_laser_map import RollingLaserMapProvider


MODEL_ROOT = os.environ.get("MOAI_MODEL_ROOT", "/root/moai_social_bert_refactored")


def main() -> None:
    logger = logging.getLogger("model_runtime_smoke")
    logging.basicConfig(level=logging.INFO)

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
        robot_history=[(-1.4 + 0.2 * index, 0.0) for index in range(8)],
        robot_yaw=0.0,
        human_histories={},
        final_goal=(8.0, 0.0),
        guidance_point_world=(8.0, 0.0),
    )
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

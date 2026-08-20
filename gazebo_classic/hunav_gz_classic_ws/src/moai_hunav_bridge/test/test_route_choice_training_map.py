from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
POSTPROCESS_PATH = PACKAGE_ROOT / "scripts" / "postprocess_pedestrian_dataset.py"
SPEC = importlib.util.spec_from_file_location("postprocess_pedestrian_dataset", POSTPROCESS_PATH)
POSTPROCESS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(POSTPROCESS)

MAP_PATH = (
    Path(__file__).resolve().parents[2]
    / "hunav_gazebo_wrapper"
    / "maps"
    / "training_route_choice.yaml"
)


def direct_segment_is_blocked(planner, start: np.ndarray, goal: np.ndarray) -> bool:
    return any(
        not planner.is_safe_world(start + fraction * (goal - start))
        for fraction in np.linspace(0.0, 1.0, 361)
    )


def test_upper_and_lower_profiles_require_distinct_safe_bypasses() -> None:
    map_data = POSTPROCESS.load_map(MAP_PATH)
    planner = POSTPROCESS.OccupancyGridRoutePlanner(
        map_data,
        clearance_m=0.375,
        planning_resolution_m=0.10,
    )

    lower_start = np.asarray([-8.6, -1.6], dtype=np.float32)
    lower_goal = np.asarray([9.0, -1.6], dtype=np.float32)
    upper_start = np.asarray([-8.4, 1.6], dtype=np.float32)
    upper_goal = np.asarray([9.0, 1.6], dtype=np.float32)

    lower_route, _ = planner.route(lower_start, lower_goal, 0.2, 0.2)
    upper_route, _ = planner.route(upper_start, upper_goal, 0.2, 0.2)
    lower_gp = POSTPROCESS.guidance_point_along_route(lower_route, 8.0)
    upper_gp = POSTPROCESS.guidance_point_along_route(upper_route, 8.0)

    assert direct_segment_is_blocked(planner, lower_start, lower_goal)
    assert direct_segment_is_blocked(planner, upper_start, upper_goal)
    assert planner.is_safe_world(lower_gp)
    assert planner.is_safe_world(upper_gp)
    assert float(lower_gp[1]) <= -2.3
    assert float(upper_gp[1]) >= 3.8

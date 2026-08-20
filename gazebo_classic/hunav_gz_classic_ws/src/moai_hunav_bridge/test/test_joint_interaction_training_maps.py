from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
POSTPROCESS_PATH = PACKAGE_ROOT / "scripts" / "postprocess_pedestrian_dataset.py"
SPEC = importlib.util.spec_from_file_location("postprocess_pedestrian_dataset", POSTPROCESS_PATH)
POSTPROCESS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(POSTPROCESS)

WRAPPER_ROOT = Path(__file__).resolve().parents[2] / "hunav_gazebo_wrapper"
CATALOG_PATH = WRAPPER_ROOT / "scenarios" / "joint_interaction_catalog.json"
TEACHER_NAV_PARAMS_PATH = (
    WRAPPER_ROOT / "launch" / "pmb2_params" / "pmb2_nav_route_choice_teacher.yaml"
)
PROFESSOR_DEMO_SCENARIO_PATH = (
    WRAPPER_ROOT / "scenarios" / "agents_professor_balanced_demo.yaml"
)
PROFESSOR_DEMO_RUNNER_PATH = PACKAGE_ROOT.parents[2] / "run-professor-model-demo.bash"


def direct_segment_is_blocked(planner, start: np.ndarray, goal: np.ndarray) -> bool:
    return any(
        not planner.is_safe_world(start + fraction * (goal - start))
        for fraction in np.linspace(0.0, 1.0, 401)
    )


def straight_guidance(start: np.ndarray, goal: np.ndarray, lookahead: float) -> np.ndarray:
    delta = goal - start
    distance = float(np.linalg.norm(delta))
    return start + delta / distance * min(lookahead, distance)


def test_joint_catalog_routes_require_safe_map_aware_guidance() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    assert set(catalog) == {
        "agents_training_route_choice_joint_lower_yield.yaml",
        "agents_training_route_choice_joint_lower_crossing.yaml",
        "agents_training_route_choice_joint_upper_crossing.yaml",
        "agents_training_route_choice_joint_lower_oncoming.yaml",
        "agents_training_route_choice_joint_upper_oncoming.yaml",
        "agents_training_bottleneck_merge_joint_gate.yaml",
        "agents_training_outdoor_chicane_joint_crossing.yaml",
    }

    for scenario_name, profile in catalog.items():
        scenario_path = WRAPPER_ROOT / "scenarios" / scenario_name
        assert scenario_path.exists()
        scenario_text = scenario_path.read_text(encoding="utf-8")
        assert f"    map: {profile['map']}" in scenario_text
        assert "cyclic_goals: true" not in scenario_text
        assert scenario_text.count("cyclic_goals: false") == profile["agent_count"]

        map_path = WRAPPER_ROOT / "maps" / f"{profile['map']}.yaml"
        map_data = POSTPROCESS.load_map(map_path)
        planner = POSTPROCESS.OccupancyGridRoutePlanner(
            map_data,
            clearance_m=0.375,
            planning_resolution_m=0.10,
        )
        start = np.asarray(profile["robot_start"], dtype=np.float32)
        goal = np.asarray(profile["robot_goal"], dtype=np.float32)
        route, metrics = planner.route(start, goal, 0.75, 0.0)
        guidance = POSTPROCESS.guidance_point_along_route(route, 8.0)
        direct_guidance = straight_guidance(start, goal, 8.0)

        assert direct_segment_is_blocked(planner, start, goal), scenario_name
        assert planner.is_safe_world(guidance), scenario_name
        assert metrics["route_path_length_m"] > float(np.linalg.norm(goal - start))
        assert float(np.linalg.norm(guidance - direct_guidance)) >= 0.50


def test_new_map_assets_are_complete() -> None:
    for map_name in ("training_bottleneck_merge", "training_outdoor_chicane"):
        assert (WRAPPER_ROOT / "maps" / f"{map_name}.pgm").exists()
        assert (WRAPPER_ROOT / "maps" / f"{map_name}.yaml").exists()
        assert (WRAPPER_ROOT / "worlds" / f"{map_name}.world").exists()
        for density in ("low", "medium", "high"):
            assert (
                WRAPPER_ROOT / "scenarios" / f"agents_{map_name}_{density}.yaml"
            ).exists()


def test_route_choice_interactions_exit_crossings_and_bias_oncoming_outward() -> None:
    expected_oncoming_y = {
        "agents_training_route_choice_joint_lower_oncoming.yaml": -2.55,
        "agents_training_route_choice_joint_upper_oncoming.yaml": 4.15,
    }
    for scenario_name, expected_y in expected_oncoming_y.items():
        params = yaml.safe_load(
            (WRAPPER_ROOT / "scenarios" / scenario_name).read_text(encoding="utf-8")
        )["hunav_loader"]["ros__parameters"]
        agent = params["agent1"]
        goal = params["global_goals"][agent["goals"][0]]
        assert float(agent["init_pose"]["y"]) == expected_y
        assert float(goal["y"]) == expected_y

    upper_crossing = yaml.safe_load(
        (
            WRAPPER_ROOT
            / "scenarios"
            / "agents_training_route_choice_joint_upper_crossing.yaml"
        ).read_text(encoding="utf-8")
    )["hunav_loader"]["ros__parameters"]
    second_agent = upper_crossing["agent2"]
    second_goal = upper_crossing["global_goals"][second_agent["goals"][0]]
    assert float(second_goal["y"]) == 1.0

    lower_yield = yaml.safe_load(
        (
            WRAPPER_ROOT
            / "scenarios"
            / "agents_training_route_choice_joint_lower_yield.yaml"
        ).read_text(encoding="utf-8")
    )["hunav_loader"]["ros__parameters"]
    assert float(lower_yield["agent1"]["max_vel"]) == 0.35
    first_goal = lower_yield["global_goals"][lower_yield["agent1"]["goals"][0]]
    second_goal = lower_yield["global_goals"][lower_yield["agent2"]["goals"][0]]
    assert float(first_goal["y"]) == 6.4
    assert float(second_goal["y"]) == 4.5


def test_route_choice_teacher_uses_conservative_static_clearance() -> None:
    params = yaml.safe_load(TEACHER_NAV_PARAMS_PATH.read_text(encoding="utf-8"))

    for costmap_name in ("global_costmap", "local_costmap"):
        costmap = params[costmap_name][costmap_name]["ros__parameters"]
        assert float(costmap["robot_radius"]) == 0.50
        assert float(costmap["inflation_layer"]["inflation_radius"]) == 0.70


def test_professor_demo_uses_clear_separated_continuous_human_lanes() -> None:
    params = yaml.safe_load(
        PROFESSOR_DEMO_SCENARIO_PATH.read_text(encoding="utf-8")
    )["hunav_loader"]["ros__parameters"]
    assert params["map"] == "training_route_choice"
    assert len(params["agents"]) == 4

    map_data = POSTPROCESS.load_map(
        WRAPPER_ROOT / "maps" / "training_route_choice.yaml"
    )
    planner = POSTPROCESS.OccupancyGridRoutePlanner(
        map_data,
        clearance_m=0.375,
        planning_resolution_m=0.10,
    )
    initial_positions = []
    expected_speeds = {
        "agent1": 0.70,
        "agent2": 0.72,
        "agent3": 0.62,
        "agent4": 0.58,
    }
    for name in params["agents"]:
        agent = params[name]
        assert agent["cyclic_goals"] is True
        assert int(agent["behavior"]["configuration"]) == 1
        assert float(agent["max_vel"]) == expected_speeds[name]
        assert len(agent["goals"]) == 4

        initial = np.asarray(
            [agent["init_pose"]["x"], agent["init_pose"]["y"]],
            dtype=float,
        )
        initial_positions.append(initial)
        route = [initial]
        route.extend(
            np.asarray(
                [params["global_goals"][goal_id]["x"], params["global_goals"][goal_id]["y"]],
                dtype=float,
            )
            for goal_id in agent["goals"]
        )
        route.append(route[1])
        for start, goal in zip(route, route[1:]):
            for fraction in np.linspace(0.0, 1.0, 201):
                assert planner.is_safe_world(start + fraction * (goal - start)), name

    for index, first in enumerate(initial_positions):
        for second in initial_positions[index + 1:]:
            assert float(np.linalg.norm(first - second)) >= 2.0

    second_agent = params["agent2"]
    second_entry = params["global_goals"][second_agent["goals"][0]]
    second_exit = params["global_goals"][second_agent["goals"][1]]
    assert float(second_entry["x"]) == -1.3
    assert float(second_exit["x"]) == -1.3
    assert float(second_entry["x"] - second_agent["init_pose"]["x"]) == 4.0

    runner = PROFESSOR_DEMO_RUNNER_PATH.read_text(encoding="utf-8")
    assert "export HUNAV_USE_NAVGOAL_TO_START=True" in runner

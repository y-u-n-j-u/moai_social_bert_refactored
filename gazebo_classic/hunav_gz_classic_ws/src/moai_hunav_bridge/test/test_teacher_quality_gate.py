import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[4]
    / "scripts"
    / "teacher_quality_gate.py"
)
SPEC = importlib.util.spec_from_file_location("teacher_quality_gate", SCRIPT_PATH)
quality_gate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(quality_gate)


def _write_recording(
    tmp_path: Path,
    human_y: float,
    clean_avoidance_samples: int = 0,
) -> tuple[Path, Path]:
    robot = np.stack((np.linspace(0.0, 4.0, 20), np.zeros(20)), axis=1)
    human = np.stack((np.full(20, 2.0), np.full(20, human_y)), axis=1)
    raw = {
        "all_trajs": [np.stack((robot, human), axis=0)],
        "sample_meta": [{
            "episode_id": 0,
            "start_frame": 0,
            "end_frame": 19,
            "start_stamp": 0.0,
            "frame_intervals_s": [0.4] * 19,
            "collection_seed": 101,
            "scenario_name": "agents_training_intersection_safe_teacher_low.yaml",
            "agents_wait_for_goal": True,
            "pedestrians_avoid_robot": False,
            "human_avoidance_mode": "yield",
            "human_yield_supervisor": True,
            "final_goal": [4.0, 0.0],
        }],
    }
    raw_path = tmp_path / "raw.pkl"
    with raw_path.open("wb") as stream:
        pickle.dump(raw, stream)

    distance = float(np.min(np.linalg.norm(human - robot, axis=1)))
    unsafe = int(distance < 1.2)
    summary = {
        "input_samples": 1,
        "clean_all_samples": 1 - unsafe,
        "clean_social_samples": 1 - unsafe,
        "clean_avoidance_samples": clean_avoidance_samples if not unsafe else 0,
        "basic_failure_reasons": (
            {"human_clearance_violation": 1} if unsafe else {}
        ),
        "route_failure_reasons": {},
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))
    return raw_path, summary_path


def test_safe_recording_passes(tmp_path):
    raw_path, summary_path = _write_recording(tmp_path, human_y=2.0)

    report = quality_gate.evaluate_recording(
        raw_path,
        summary_path,
        expected_pedestrians_avoid_robot=False,
    )

    assert report["passed"]
    assert report["metrics"]["windows_below_social_distance"] == 0


def test_any_unsafe_window_rejects_whole_recording(tmp_path):
    raw_path, summary_path = _write_recording(tmp_path, human_y=0.5)

    report = quality_gate.evaluate_recording(
        raw_path,
        summary_path,
        expected_pedestrians_avoid_robot=False,
    )

    assert not report["passed"]
    assert not report["checks"]["no_social_clearance_violation_window"]
    assert not report["checks"]["no_unsafe_postprocess_failure"]


def test_required_reactive_avoidance_window_is_enforced(tmp_path):
    raw_path, summary_path = _write_recording(
        tmp_path,
        human_y=2.0,
        clean_avoidance_samples=0,
    )

    report = quality_gate.evaluate_recording(
        raw_path,
        summary_path,
        minimum_clean_avoidance_windows=1,
        expected_pedestrians_avoid_robot=False,
    )

    assert not report["passed"]
    assert not report["checks"]["contains_clean_avoidance_window"]


def test_expected_human_yield_supervisor_is_recorded(tmp_path):
    raw_path, summary_path = _write_recording(tmp_path, human_y=2.0)

    report = quality_gate.evaluate_recording(
        raw_path,
        summary_path,
        expected_human_yield_supervisor=True,
    )

    assert report["passed"]
    assert report["metrics"]["human_yield_supervisor_values"] == [True]


def test_expected_human_avoidance_mode_is_recorded(tmp_path):
    raw_path, summary_path = _write_recording(tmp_path, human_y=2.0)

    report = quality_gate.evaluate_recording(
        raw_path,
        summary_path,
        expected_human_avoidance_mode="yield",
    )

    assert report["passed"]
    assert report["metrics"]["human_avoidance_mode_values"] == ["yield"]


def test_continuous_motion_gate_rejects_stop_and_wait(tmp_path):
    raw_path, summary_path = _write_recording(
        tmp_path,
        human_y=2.0,
        clean_avoidance_samples=1,
    )
    with raw_path.open("rb") as stream:
        raw = pickle.load(stream)
    robot = raw["all_trajs"][0][0]
    robot[7:14] = robot[7]
    with raw_path.open("wb") as stream:
        pickle.dump(raw, stream)

    report = quality_gate.evaluate_recording(
        raw_path,
        summary_path,
        minimum_clean_avoidance_windows=1,
        max_robot_stop_duration_s=0.8,
    )

    assert not report["passed"]
    assert not report["checks"]["no_extended_robot_stop"]
    assert report["metrics"]["longest_robot_stop_duration_s"] > 0.8


def test_lateral_avoidance_gate_rejects_straight_teacher(tmp_path):
    raw_path, summary_path = _write_recording(
        tmp_path,
        human_y=2.0,
        clean_avoidance_samples=1,
    )

    report = quality_gate.evaluate_recording(
        raw_path,
        summary_path,
        minimum_clean_avoidance_windows=1,
        minimum_route_deviation_m=0.8,
    )

    assert not report["passed"]
    assert not report["checks"]["sufficient_lateral_avoidance"]


def test_aggregate_requires_ten_unique_passing_seeds():
    reports = []
    for seed in range(101, 111):
        reports.append({
            "passed": True,
            "raw_path": f"seed_{seed}.pkl",
            "metrics": {
                "collection_seeds": [seed],
                "minimum_human_distance_m": 1.3,
                "raw_window_count": 3,
                "encounter_profiles": [[float(seed), 0.0, -6.0, 0.0]],
            },
            "postprocess": {
                "clean_social_samples": 2,
                "clean_avoidance_samples": 1,
            },
        })

    aggregate = quality_gate.aggregate_reports(reports, required_runs=10)

    assert aggregate["passed"]
    assert aggregate["report_count"] == 10
    assert aggregate["total_clean_social_windows"] == 20
    assert aggregate["total_clean_avoidance_windows"] == 10

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SAFETY_FAILURE_REASONS = {
    "human_clearance_violation",
    "observed_human_clearance_violation",
    "speed_limit",
    "acceleration_limit",
    "yaw_rate_limit",
    "target_map_collision",
    "irregular_frame_interval",
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _finite_min(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return min(finite) if finite else None


def _robot_motion_metrics(
    payload: dict[str, Any],
    stop_speed_threshold_mps: float,
) -> dict[str, float | None]:
    trajectories = payload.get("all_trajs", [])
    sample_meta = payload.get("sample_meta", [])
    default_dt = float(payload.get("metadata", {}).get("dt", 0.4))
    episode_frames: dict[int, dict[int, tuple[float, np.ndarray]]] = {}
    episode_goals: dict[int, np.ndarray] = {}

    for window_index, trajectory in enumerate(trajectories):
        points = np.asarray(trajectory, dtype=np.float64)
        if points.ndim != 3 or points.shape[0] < 1 or points.shape[2] != 2:
            continue
        meta = sample_meta[window_index] if window_index < len(sample_meta) else {}
        meta = meta if isinstance(meta, dict) else {}
        episode_id = int(meta.get("episode_id", 0))
        start_frame = int(meta.get("start_frame", window_index * points.shape[1]))
        start_stamp = float(meta.get("start_stamp", start_frame * default_dt))
        intervals = [float(value) for value in meta.get("frame_intervals_s", [])]
        stamps = [start_stamp]
        for offset in range(1, points.shape[1]):
            dt = intervals[offset - 1] if offset - 1 < len(intervals) else default_dt
            stamps.append(stamps[-1] + dt)

        frames = episode_frames.setdefault(episode_id, {})
        for offset, (stamp, position) in enumerate(zip(stamps, points[0, :, :2])):
            if np.isfinite(position).all():
                frames.setdefault(start_frame + offset, (stamp, position.copy()))

        final_goal = meta.get("final_goal")
        if isinstance(final_goal, (list, tuple)) and len(final_goal) >= 2:
            goal = np.asarray(final_goal[:2], dtype=np.float64)
            if np.isfinite(goal).all():
                episode_goals[episode_id] = goal

    speeds: list[float] = []
    longest_stop_duration_s = 0.0
    maximum_route_deviation_m = 0.0
    episode_duration_s = 0.0
    for episode_id, frames in episode_frames.items():
        ordered = [frames[index] for index in sorted(frames)]
        if len(ordered) < 2:
            continue
        episode_duration_s += max(ordered[-1][0] - ordered[0][0], 0.0)
        current_stop_duration_s = 0.0
        for previous, current in zip(ordered, ordered[1:]):
            dt = float(current[0] - previous[0])
            if not math.isfinite(dt) or dt <= 1e-6:
                continue
            speed = float(np.linalg.norm(current[1] - previous[1]) / dt)
            speeds.append(speed)
            if speed < stop_speed_threshold_mps:
                current_stop_duration_s += dt
                longest_stop_duration_s = max(
                    longest_stop_duration_s,
                    current_stop_duration_s,
                )
            else:
                current_stop_duration_s = 0.0

        start = ordered[0][1]
        goal = episode_goals.get(episode_id, ordered[-1][1])
        direction = goal - start
        length = float(np.linalg.norm(direction))
        if length > 1e-6:
            offsets = np.stack([position - start for _, position in ordered])
            deviations = np.abs(
                direction[0] * offsets[:, 1] - direction[1] * offsets[:, 0]
            ) / length
            maximum_route_deviation_m = max(
                maximum_route_deviation_m,
                float(np.max(deviations)),
            )

    return {
        "episode_duration_s": episode_duration_s if episode_frames else None,
        "minimum_robot_speed_mps": _finite_min(speeds),
        "mean_robot_speed_mps": (
            float(sum(speeds) / len(speeds)) if speeds else None
        ),
        "longest_robot_stop_duration_s": (
            longest_stop_duration_s if speeds else None
        ),
        "maximum_route_deviation_m": (
            maximum_route_deviation_m if episode_frames else None
        ),
    }


def raw_recording_metrics(
    payload: dict[str, Any],
    social_distance_m: float,
    collision_distance_m: float,
    stop_speed_threshold_mps: float = 0.1,
) -> dict[str, Any]:
    trajectories = payload.get("all_trajs", [])
    sample_meta = payload.get("sample_meta", [])
    window_minima: list[float] = []
    invalid_windows = 0

    for trajectory in trajectories:
        points = np.asarray(trajectory, dtype=np.float32)
        if points.ndim != 3 or points.shape[0] < 2 or points.shape[2] != 2:
            invalid_windows += 1
            continue
        distances = np.linalg.norm(points[1:] - points[0][None, :, :], axis=2)
        finite = distances[np.isfinite(distances)]
        if finite.size == 0:
            invalid_windows += 1
            continue
        window_minima.append(float(np.min(finite)))

    robot_initial_positions = []
    if trajectories:
        first = np.asarray(trajectories[0], dtype=np.float32)
        if first.ndim == 3 and first.shape[0] >= 1 and first.shape[1] >= 1:
            robot_initial_positions.append([
                round(float(first[0, 0, 0]), 2),
                round(float(first[0, 0, 1]), 2),
            ])
    final_goals = sorted(
        {
            (
                round(float(meta["final_goal"][0]), 2),
                round(float(meta["final_goal"][1]), 2),
            )
            for meta in sample_meta
            if isinstance(meta, dict)
            and isinstance(meta.get("final_goal"), (list, tuple))
            and len(meta["final_goal"]) >= 2
        }
    )

    episode_ids = sorted(
        {
            int(meta["episode_id"])
            for meta in sample_meta
            if isinstance(meta, dict) and meta.get("episode_id") is not None
        }
    )
    seeds = sorted(
        {
            int(meta["collection_seed"])
            for meta in sample_meta
            if isinstance(meta, dict) and meta.get("collection_seed") is not None
        }
    )
    scenarios = sorted(
        {
            str(meta["scenario_name"])
            for meta in sample_meta
            if isinstance(meta, dict) and meta.get("scenario_name")
        }
    )
    synchronized = bool(sample_meta) and all(
        _as_bool(meta.get("agents_wait_for_goal", False))
        for meta in sample_meta
        if isinstance(meta, dict)
    )
    coupling_values = sorted(
        {
            _as_bool(meta.get("pedestrians_avoid_robot", False))
            for meta in sample_meta
            if isinstance(meta, dict)
        }
    )
    yield_supervisor_values = sorted(
        {
            _as_bool(meta["human_yield_supervisor"])
            for meta in sample_meta
            if isinstance(meta, dict) and "human_yield_supervisor" in meta
        }
    )
    avoidance_mode_values = sorted(
        {
            str(meta["human_avoidance_mode"]).strip().lower()
            for meta in sample_meta
            if isinstance(meta, dict) and meta.get("human_avoidance_mode")
        }
    )

    return {
        "raw_window_count": len(trajectories),
        "valid_distance_window_count": len(window_minima),
        "invalid_window_count": invalid_windows,
        "episode_ids": episode_ids,
        "collection_seeds": seeds,
        "scenario_names": scenarios,
        "robot_initial_positions": robot_initial_positions,
        "final_goals": [list(goal) for goal in final_goals],
        "encounter_profiles": [
            [*start, *goal]
            for start in robot_initial_positions
            for goal in final_goals
        ],
        "agents_wait_for_goal": synchronized,
        "pedestrians_avoid_robot_values": coupling_values,
        "human_avoidance_mode_values": avoidance_mode_values,
        "human_yield_supervisor_values": yield_supervisor_values,
        "minimum_human_distance_m": _finite_min(window_minima),
        "windows_below_social_distance": sum(
            distance < social_distance_m for distance in window_minima
        ),
        "windows_below_collision_distance": sum(
            distance < collision_distance_m for distance in window_minima
        ),
        **_robot_motion_metrics(payload, stop_speed_threshold_mps),
    }


def evaluate_recording(
    raw_path: Path,
    processed_summary_path: Path,
    social_distance_m: float = 1.2,
    collision_distance_m: float = 0.65,
    minimum_clean_social_windows: int = 1,
    minimum_clean_avoidance_windows: int = 0,
    expected_pedestrians_avoid_robot: bool | None = None,
    expected_human_avoidance_mode: str | None = None,
    expected_human_yield_supervisor: bool | None = None,
    max_robot_stop_duration_s: float | None = None,
    stop_speed_threshold_mps: float = 0.1,
    minimum_route_deviation_m: float | None = None,
) -> dict[str, Any]:
    with raw_path.open("rb") as stream:
        payload = pickle.load(stream)
    summary = json.loads(processed_summary_path.read_text())
    metrics = raw_recording_metrics(
        payload,
        social_distance_m=social_distance_m,
        collision_distance_m=collision_distance_m,
        stop_speed_threshold_mps=stop_speed_threshold_mps,
    )
    failure_counts = summary.get("basic_failure_reasons", {})
    unsafe_failures = {
        reason: int(failure_counts.get(reason, 0))
        for reason in sorted(SAFETY_FAILURE_REASONS)
        if int(failure_counts.get(reason, 0)) > 0
    }
    route_failures = sum(
        int(count) for count in summary.get("route_failure_reasons", {}).values()
    )

    checks = {
        "contains_windows": metrics["raw_window_count"] > 0,
        "all_windows_well_formed": (
            metrics["valid_distance_window_count"] == metrics["raw_window_count"]
        ),
        "single_completed_episode": len(metrics["episode_ids"]) == 1,
        "pedestrians_started_with_goal": metrics["agents_wait_for_goal"],
        "no_physical_collision_window": metrics["windows_below_collision_distance"] == 0,
        "no_social_clearance_violation_window": metrics["windows_below_social_distance"] == 0,
        "no_unsafe_postprocess_failure": not unsafe_failures,
        "no_route_guidance_failure": route_failures == 0,
        "contains_clean_social_window": (
            int(summary.get("clean_social_samples", 0))
            >= int(minimum_clean_social_windows)
        ),
        "contains_clean_avoidance_window": (
            int(summary.get("clean_avoidance_samples", 0))
            >= int(minimum_clean_avoidance_windows)
        ),
    }
    if expected_pedestrians_avoid_robot is not None:
        checks["expected_coupling_policy"] = (
            metrics["pedestrians_avoid_robot_values"]
            == [expected_pedestrians_avoid_robot]
        )
    if expected_human_avoidance_mode is not None:
        checks["expected_human_avoidance_mode"] = (
            metrics["human_avoidance_mode_values"]
            == [expected_human_avoidance_mode]
        )
    if expected_human_yield_supervisor is not None:
        checks["expected_human_yield_supervisor"] = (
            metrics["human_yield_supervisor_values"]
            == [expected_human_yield_supervisor]
        )
    if max_robot_stop_duration_s is not None:
        stop_duration = metrics["longest_robot_stop_duration_s"]
        checks["no_extended_robot_stop"] = (
            stop_duration is not None
            and stop_duration <= float(max_robot_stop_duration_s)
        )
    if minimum_route_deviation_m is not None:
        route_deviation = metrics["maximum_route_deviation_m"]
        checks["sufficient_lateral_avoidance"] = (
            route_deviation is not None
            and route_deviation >= float(minimum_route_deviation_m)
        )

    return {
        "passed": all(checks.values()),
        "raw_path": str(raw_path),
        "processed_summary_path": str(processed_summary_path),
        "thresholds": {
            "social_distance_m": float(social_distance_m),
            "collision_distance_m": float(collision_distance_m),
            "minimum_clean_social_windows": int(minimum_clean_social_windows),
            "minimum_clean_avoidance_windows": int(minimum_clean_avoidance_windows),
            "expected_human_avoidance_mode": expected_human_avoidance_mode,
            "stop_speed_threshold_mps": float(stop_speed_threshold_mps),
            "max_robot_stop_duration_s": max_robot_stop_duration_s,
            "minimum_route_deviation_m": minimum_route_deviation_m,
        },
        "checks": checks,
        "metrics": metrics,
        "postprocess": {
            "input_samples": int(summary.get("input_samples", 0)),
            "clean_all_samples": int(summary.get("clean_all_samples", 0)),
            "clean_social_samples": int(summary.get("clean_social_samples", 0)),
            "clean_avoidance_samples": int(summary.get("clean_avoidance_samples", 0)),
            "unsafe_failure_reasons": unsafe_failures,
            "route_failure_count": route_failures,
        },
    }


def aggregate_reports(reports: list[dict[str, Any]], required_runs: int) -> dict[str, Any]:
    seeds = [
        seed
        for report in reports
        for seed in report.get("metrics", {}).get("collection_seeds", [])
    ]
    distances = [
        report.get("metrics", {}).get("minimum_human_distance_m")
        for report in reports
    ]
    profiles = {
        tuple(profile)
        for report in reports
        for profile in report.get("metrics", {}).get("encounter_profiles", [])
    }
    checks = {
        "required_run_count": len(reports) >= required_runs,
        "unique_seed_count": len(set(seeds)) >= required_runs,
        "unique_encounter_profile_count": len(profiles) >= required_runs,
        "all_runs_passed": bool(reports) and all(report.get("passed", False) for report in reports),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "required_runs": required_runs,
        "report_count": len(reports),
        "passed_run_count": sum(report.get("passed", False) for report in reports),
        "collection_seeds": sorted(set(seeds)),
        "encounter_profiles": [list(profile) for profile in sorted(profiles)],
        "minimum_human_distance_m": _finite_min(
            value for value in distances if value is not None
        ),
        "total_raw_windows": sum(
            int(report.get("metrics", {}).get("raw_window_count", 0))
            for report in reports
        ),
        "total_clean_social_windows": sum(
            int(report.get("postprocess", {}).get("clean_social_samples", 0))
            for report in reports
        ),
        "total_clean_avoidance_windows": sum(
            int(report.get("postprocess", {}).get("clean_avoidance_samples", 0))
            for report in reports
        ),
        "failed_reports": [
            report.get("raw_path")
            for report in reports
            if not report.get("passed", False)
        ],
    }


def _write_report(report: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")
    print(rendered)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reject unsafe HuNavSim teacher recordings before training."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--raw", type=Path, required=True)
    evaluate.add_argument("--processed-summary", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--social-distance", type=float, default=1.2)
    evaluate.add_argument("--collision-distance", type=float, default=0.65)
    evaluate.add_argument("--minimum-clean-social-windows", type=int, default=1)
    evaluate.add_argument("--minimum-clean-avoidance-windows", type=int, default=0)
    evaluate.add_argument("--stop-speed-threshold", type=float, default=0.1)
    evaluate.add_argument("--max-robot-stop-duration", type=float)
    evaluate.add_argument("--minimum-route-deviation", type=float)
    evaluate.add_argument(
        "--expected-coupling",
        choices=["any", "one-way", "reciprocal"],
        default="any",
    )
    evaluate.add_argument(
        "--expected-human-avoidance-mode",
        choices=["any", "off", "yield", "continuous"],
        default="any",
    )
    evaluate.add_argument(
        "--expected-human-yield-supervisor",
        choices=["any", "false", "true"],
        default="any",
    )

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--reports", type=Path, nargs="+", required=True)
    aggregate.add_argument("--required-runs", type=int, default=10)
    aggregate.add_argument("--output", type=Path)

    args = parser.parse_args()
    if args.command == "evaluate":
        expected = {
            "any": None,
            "one-way": False,
            "reciprocal": True,
        }[args.expected_coupling]
        expected_yield_supervisor = {
            "any": None,
            "false": False,
            "true": True,
        }[args.expected_human_yield_supervisor]
        expected_avoidance_mode = (
            None
            if args.expected_human_avoidance_mode == "any"
            else args.expected_human_avoidance_mode
        )
        report = evaluate_recording(
            raw_path=args.raw,
            processed_summary_path=args.processed_summary,
            social_distance_m=args.social_distance,
            collision_distance_m=args.collision_distance,
            minimum_clean_social_windows=args.minimum_clean_social_windows,
            minimum_clean_avoidance_windows=args.minimum_clean_avoidance_windows,
            expected_pedestrians_avoid_robot=expected,
            expected_human_avoidance_mode=expected_avoidance_mode,
            expected_human_yield_supervisor=expected_yield_supervisor,
            max_robot_stop_duration_s=args.max_robot_stop_duration,
            stop_speed_threshold_mps=args.stop_speed_threshold,
            minimum_route_deviation_m=args.minimum_route_deviation,
        )
    else:
        reports = [json.loads(path.read_text()) for path in args.reports]
        report = aggregate_reports(reports, required_runs=args.required_runs)

    _write_report(report, args.output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

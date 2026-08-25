#!/usr/bin/env python3
"""Build balanced route-GP splits from joint-interaction pilot recordings."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from itertools import product
import json
import pickle
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_BATCHES = (
    "head_on_batch_10seeds_20260822",
    "crossing_90_batch_10seeds_20260822",
    "staggered_two_batch_10seeds_20260822",
    "overtaking_batch_10seeds_20260822",
    "cut_in_batch_10seeds_20260822",
    "detour_oncoming_batch_10seeds_20260822",
)
PREFIX = "pmb2_route_gp_v2_clean_social"
ROUTE_GUIDANCE_POLICY = "inflated_occupancy_grid_route_lookahead"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch", action="append", default=None)
    parser.add_argument("--scenario-cap", type=int, default=60)
    parser.add_argument("--seed", type=int, default=21)
    return parser.parse_args()


def sample_identity(sample: dict[str, Any]) -> tuple[Any, ...]:
    meta = sample.get("meta", {})
    return (
        meta.get("collection_seed"),
        meta.get("episode_id"),
        meta.get("start_frame"),
        meta.get("end_frame"),
        meta.get("source_index"),
    )


def read_samples(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    return list(payload.get("samples", []))


def validate_sample(sample: dict[str, Any], source: Path) -> None:
    expected_shapes = {
        "target_past": (8, 2),
        "target_future": (12, 2),
        "guidance_point": (2,),
        "guidance_traj": (12, 2),
        "local_map": (32, 32),
    }
    for key, expected in expected_shapes.items():
        value = np.asarray(sample.get(key))
        if value.shape != expected:
            raise ValueError(
                f"{source}: {key} shape {value.shape} does not match {expected}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{source}: {key} contains non-finite values")

    meta = sample.get("meta", {})
    if meta.get("guidance_policy") != ROUTE_GUIDANCE_POLICY:
        raise ValueError(
            f"{source}: expected route guidance policy {ROUTE_GUIDANCE_POLICY}, "
            f"got {meta.get('guidance_policy')}"
        )
    if int(meta.get("route_path_pose_count", 0)) < 2:
        raise ValueError(f"{source}: route path is missing or too short")


def load_collected_samples(
    recordings_root: Path,
    batches: Iterable[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    social_samples: list[dict[str, Any]] = []
    avoidance_samples: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []

    for batch_name in batches:
        batch_dir = recordings_root / batch_name
        if not batch_dir.is_dir():
            raise FileNotFoundError(f"missing collection batch: {batch_dir}")
        social_paths = sorted(batch_dir.glob("seed_*/processed/*_clean_social.pkl"))
        for social_path in social_paths:
            seed_dir = social_path.parent.parent
            seed_name = seed_dir.name
            avoidance_path = social_path.with_name(
                social_path.name.replace("_clean_social.pkl", "_clean_avoidance.pkl")
            )
            avoidance_run = (
                read_samples(avoidance_path) if avoidance_path.is_file() else []
            )
            avoidance_ids = {sample_identity(sample) for sample in avoidance_run}
            qualification_path = seed_dir / "qualification.json"
            qualification = (
                json.loads(qualification_path.read_text(encoding="utf-8"))
                if qualification_path.is_file()
                else {}
            )
            qualification_passed = qualification.get("passed") is True

            run_samples = read_samples(social_path)
            for sample in run_samples:
                validate_sample(sample, social_path)
                item = dict(sample)
                meta = dict(item.get("meta", {}))
                meta.update(
                    {
                        "recording_id": f"{batch_name}/{seed_name}",
                        "source_batch": batch_name,
                        "source_file": str(social_path),
                        "source_qualification_passed": qualification_passed,
                        "clean_avoidance": sample_identity(sample) in avoidance_ids,
                        "map_name": Path(str(meta.get("map_yaml_path", ""))).stem,
                    }
                )
                item["meta"] = meta
                social_samples.append(item)
                if meta["clean_avoidance"]:
                    avoidance_samples.append(item)

            source_records.append(
                {
                    "batch": batch_name,
                    "seed": seed_name,
                    "clean_social": len(run_samples),
                    "clean_avoidance": len(avoidance_run),
                    "qualification_passed": qualification_passed,
                    "source": str(social_path),
                }
            )

    if not social_samples:
        raise ValueError("no clean-social samples were found")
    return social_samples, avoidance_samples, source_records


def round_robin_select(
    samples: list[dict[str, Any]],
    cap: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if cap <= 0 or len(samples) <= cap:
        return list(samples)

    by_recording: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_recording[str(sample["meta"]["recording_id"])].append(sample)
    recording_ids = sorted(by_recording)
    rng.shuffle(recording_ids)

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    for avoidance_only in (True, False):
        queues: dict[str, list[dict[str, Any]]] = {}
        for recording_id in recording_ids:
            queue = [
                sample
                for sample in by_recording[recording_id]
                if bool(sample["meta"]["clean_avoidance"]) is avoidance_only
            ]
            rng.shuffle(queue)
            queues[recording_id] = queue

        while len(selected) < cap and any(queues.values()):
            for recording_id in recording_ids:
                if queues[recording_id] and len(selected) < cap:
                    sample = queues[recording_id].pop()
                    marker = id(sample)
                    if marker not in selected_ids:
                        selected.append(sample)
                        selected_ids.add(marker)
        if len(selected) >= cap:
            break
    return selected


def balanced_samples(
    samples: list[dict[str, Any]],
    scenario_cap: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        scenario = Path(str(sample["meta"].get("scenario_name", "unknown"))).stem
        by_scenario[scenario].append(sample)

    selected: list[dict[str, Any]] = []
    for index, scenario in enumerate(sorted(by_scenario)):
        selected.extend(
            round_robin_select(
                by_scenario[scenario],
                scenario_cap,
                random.Random(seed + index),
            )
        )
    return selected


def split_recordings(
    samples: list[dict[str, Any]],
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    scenario_recordings: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        scenario = Path(str(sample["meta"].get("scenario_name", "unknown"))).stem
        scenario_recordings[scenario].add(str(sample["meta"]["recording_id"]))

    recording_split: dict[str, str] = {}
    assignments = {"train": [], "val": [], "test": []}
    for index, scenario in enumerate(sorted(scenario_recordings)):
        recordings = sorted(scenario_recordings[scenario])
        random.Random(seed + 100 + index).shuffle(recordings)
        if len(recordings) < 3:
            raise ValueError(f"scenario {scenario} has fewer than 3 recordings")
        recording_sizes = Counter(
            str(sample["meta"]["recording_id"])
            for sample in samples
            if Path(str(sample["meta"].get("scenario_name", "unknown"))).stem
            == scenario
        )
        total_samples = sum(recording_sizes.values())
        sample_targets = (0.70 * total_samples, 0.15 * total_samples, 0.15 * total_samples)
        group_targets = (
            0.70 * len(recordings),
            0.15 * len(recordings),
            0.15 * len(recordings),
        )
        best_assignment: tuple[int, ...] | None = None
        best_score = float("inf")
        for candidate in product(range(3), repeat=len(recordings)):
            if len(set(candidate)) != 3:
                continue
            sample_counts = [0, 0, 0]
            group_counts = [0, 0, 0]
            for recording_id, split_index in zip(recordings, candidate):
                sample_counts[split_index] += recording_sizes[recording_id]
                group_counts[split_index] += 1
            sample_error = sum(
                ((count - target) / max(target, 1.0)) ** 2
                for count, target in zip(sample_counts, sample_targets)
            )
            group_error = sum(
                ((count - target) / max(target, 1.0)) ** 2
                for count, target in zip(group_counts, group_targets)
            )
            candidate_score = sample_error + 0.05 * group_error
            if candidate_score < best_score:
                best_score = candidate_score
                best_assignment = candidate
        if best_assignment is None:
            raise RuntimeError(f"could not split scenario {scenario}")

        split_lists = {"train": [], "val": [], "test": []}
        split_names = ("train", "val", "test")
        for recording_id, split_index in zip(recordings, best_assignment):
            split_lists[split_names[split_index]].append(recording_id)
        for split, values in split_lists.items():
            for recording_id in values:
                if recording_id in recording_split:
                    raise RuntimeError(f"duplicate recording id: {recording_id}")
                recording_split[recording_id] = split
                assignments[split].append(recording_id)

    split_samples = {"train": [], "val": [], "test": []}
    for sample in samples:
        split = recording_split[str(sample["meta"]["recording_id"])]
        split_samples[split].append(sample)
    for index, split in enumerate(("train", "val", "test")):
        random.Random(seed + 200 + index).shuffle(split_samples[split])
    return split_samples, assignments


def scenario_counts(samples: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        scenario = Path(str(sample["meta"].get("scenario_name", "unknown"))).stem
        counts[scenario]["samples"] += 1
        if sample["meta"].get("clean_avoidance"):
            counts[scenario]["clean_avoidance"] += 1
        if sample["meta"].get("source_qualification_passed"):
            counts[scenario]["from_passed_episode"] += 1
    return {scenario: dict(value) for scenario, value in sorted(counts.items())}


def write_payload(path: Path, samples: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    with path.open("wb") as stream:
        pickle.dump(
            {"samples": samples, "metadata": metadata},
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def main() -> int:
    args = parse_args()
    batches = tuple(args.batch or DEFAULT_BATCHES)
    if args.scenario_cap <= 0:
        raise ValueError("--scenario-cap must be positive")

    social, avoidance, sources = load_collected_samples(
        args.recordings_root.resolve(), batches
    )
    balanced = balanced_samples(social, args.scenario_cap, args.seed)
    splits, assignments = split_recordings(balanced, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    common_metadata = {
        "format": "moai_joint_route_gp_v1",
        "guidance_policy": ROUTE_GUIDANCE_POLICY,
        "source_policy": "per_window_clean_social",
        "unsafe_windows_included": False,
        "scenario_cap": args.scenario_cap,
        "split_strategy": "scenario_stratified_recording_id",
        "split_seed": args.seed,
        "batches": list(batches),
        "raw_clean_social_samples": len(social),
        "raw_clean_avoidance_samples": len(avoidance),
        "balanced_samples": len(balanced),
    }

    write_payload(
        args.out_dir / "joint_route_gp_clean_social_all.pkl",
        social,
        {**common_metadata, "data_partition": "all_clean_social"},
    )
    write_payload(
        args.out_dir / "joint_route_gp_clean_avoidance_all.pkl",
        avoidance,
        {**common_metadata, "data_partition": "all_clean_avoidance"},
    )
    write_payload(
        args.out_dir / f"{PREFIX}_all.pkl",
        balanced,
        {**common_metadata, "data_partition": "balanced_all"},
    )
    for split, split_samples in splits.items():
        write_payload(
            args.out_dir / f"{PREFIX}_{split}.pkl",
            split_samples,
            {
                **common_metadata,
                "data_partition": split,
                "sample_count": len(split_samples),
                "recording_ids": assignments[split],
            },
        )

    split_scenarios = {
        split: scenario_counts(split_samples)
        for split, split_samples in splits.items()
    }
    manifest = {
        **common_metadata,
        "source_record_count": len(sources),
        "passed_source_record_count": sum(
            bool(item["qualification_passed"]) for item in sources
        ),
        "scenario_counts_before_balancing": scenario_counts(social),
        "scenario_counts_balanced": scenario_counts(balanced),
        "split_counts": {split: len(value) for split, value in splits.items()},
        "split_scenario_counts": split_scenarios,
        "split_recordings": assignments,
        "source_records": sources,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    with (args.out_dir / "scenario_counts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["split", "scenario", "samples", "clean_avoidance", "passed_episode"]
        )
        for split, counts in split_scenarios.items():
            for scenario, values in counts.items():
                writer.writerow(
                    [
                        split,
                        scenario,
                        values.get("samples", 0),
                        values.get("clean_avoidance", 0),
                        values.get("from_passed_episode", 0),
                    ]
                )

    readme = f"""# Joint route-GP dataset

Generated from six Gazebo joint-interaction batches.

- Route-GP policy: `{ROUTE_GUIDANCE_POLICY}`
- Raw clean-social samples: {len(social)}
- Raw clean-avoidance samples: {len(avoidance)}
- Balanced samples: {len(balanced)}
- Split samples: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}
- Unsafe windows included: no
- Split leakage unit: recording seed and episode

Training prefix: `{PREFIX}`
"""
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps({
        "out_dir": str(args.out_dir),
        "clean_social": len(social),
        "clean_avoidance": len(avoidance),
        "balanced": len(balanced),
        "splits": {split: len(value) for split, value in splits.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

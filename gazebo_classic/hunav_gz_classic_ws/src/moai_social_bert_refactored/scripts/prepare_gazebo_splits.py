#!/usr/bin/env python3
"""Merge processed Gazebo runs and split without episode leakage."""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import product
import json
import pickle
import random
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory recursively containing *_clean_social.pkl run files.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="pmb2_route_gp_v2_clean_social")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--held-out-map",
        action="append",
        default=None,
        help=(
            "Map basename that must be excluded from train/validation data. "
            "May be repeated; defaults to training_dual_route."
        ),
    )
    parser.add_argument(
        "--allow-missing-qualification",
        action="store_true",
        help=(
            "Include legacy runs that have no qualification.json. Runs with "
            "an explicit failed qualification are always excluded."
        ),
    )
    parser.add_argument(
        "--minimum-pilot-runs",
        type=int,
        default=10,
        help=(
            "Require a passed parent pilot_summary.json with at least this "
            "many required, reported, and passed runs. Set 0 for legacy data."
        ),
    )
    return parser.parse_args()


def sample_map_name(sample: dict[str, Any], source: Path) -> str:
    meta = sample.get("meta", {})
    raw_path = str(meta.get("map_yaml_path", "")).strip()
    if raw_path:
        return Path(raw_path).stem
    scenario = Path(str(meta.get("scenario_name", ""))).stem
    if scenario.startswith("agents_"):
        scenario = scenario[len("agents_") :]
        for suffix in (
            "_safe_teacher_low",
            "_safe_teacher_medium",
            "_safe_teacher_high",
            "_low",
            "_medium",
            "_high",
        ):
            if scenario.endswith(suffix):
                return scenario[: -len(suffix)]
    raise KeyError(f"sample in {source} has no usable map identity")


def load_runs(
    input_dir: Path,
    held_out_maps: set[str],
    require_qualification: bool = False,
    minimum_pilot_runs: int = 0,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    paths = sorted(input_dir.rglob("*_clean_social.pkl"))
    if not paths:
        raise FileNotFoundError(f"no *_clean_social.pkl files below {input_dir}")

    samples: list[dict[str, Any]] = []
    sources: list[str] = []
    excluded_sources: list[dict[str, Any]] = []
    for path in paths:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        run_samples = list(payload.get("samples", []))
        if not run_samples:
            print(f"warning: skipping empty processed run: {path}")
            continue

        map_names = sorted({sample_map_name(sample, path) for sample in run_samples})
        held_out_in_run = sorted(set(map_names) & held_out_maps)
        if held_out_in_run:
            if len(map_names) != len(held_out_in_run):
                raise ValueError(
                    f"processed run mixes training and held-out maps: {path}: {map_names}"
                )
            excluded_sources.append(
                {
                    "path": str(path),
                    "maps": map_names,
                    "sample_count": len(run_samples),
                    "reason": "held_out_map",
                }
            )
            print(
                "held-out exclusion: "
                f"{path} maps={map_names} samples={len(run_samples)}"
            )
            continue

        run_dir = path.parent.parent
        qualification_candidates = (
            path.parent / "qualification.json",
            run_dir / "qualification.json",
        )
        qualification_path = next(
            (candidate for candidate in qualification_candidates if candidate.is_file()),
            qualification_candidates[-1],
        )
        pilot_summary_path = run_dir.parent / "pilot_summary.json"
        qualification: dict[str, Any] | None = None
        pilot_summary: dict[str, Any] | None = None
        qualification_error: str | None = None
        pilot_summary_error: str | None = None
        if qualification_path.is_file():
            try:
                qualification = json.loads(qualification_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                qualification_error = str(exc)
        if pilot_summary_path.is_file():
            try:
                pilot_summary = json.loads(pilot_summary_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                pilot_summary_error = str(exc)

        exclusion_reason: str | None = None
        if qualification_error is not None:
            exclusion_reason = "invalid_qualification"
        elif qualification is not None and qualification.get("passed") is not True:
            exclusion_reason = "qualification_failed"
        elif qualification is None and require_qualification:
            exclusion_reason = "qualification_missing"
        elif pilot_summary_error is not None:
            exclusion_reason = "invalid_pilot_summary"
        elif pilot_summary is not None and pilot_summary.get("passed") is not True:
            exclusion_reason = "pilot_summary_failed"
        elif pilot_summary is None and minimum_pilot_runs > 0:
            exclusion_reason = "pilot_summary_missing"
        elif pilot_summary is not None and minimum_pilot_runs > 0:
            pilot_counts = (
                int(pilot_summary.get("required_runs", 0)),
                int(pilot_summary.get("report_count", 0)),
                int(pilot_summary.get("passed_run_count", 0)),
            )
            if min(pilot_counts) < minimum_pilot_runs:
                exclusion_reason = "pilot_run_count_below_minimum"

        if exclusion_reason is not None:
            excluded_sources.append(
                {
                    "path": str(path),
                    "maps": map_names,
                    "sample_count": len(run_samples),
                    "reason": exclusion_reason,
                    "qualification_path": str(qualification_path),
                    "qualification_error": qualification_error,
                    "pilot_summary_path": str(pilot_summary_path),
                    "pilot_summary_error": pilot_summary_error,
                }
            )
            print(
                "quality exclusion: "
                f"{path} reason={exclusion_reason} samples={len(run_samples)}"
            )
            continue

        relative_path = path.relative_to(input_dir)
        recording_id = str(
            relative_path.parent
            / relative_path.name[: -len("_clean_social.pkl")]
        )
        for source_index, source_sample in enumerate(run_samples):
            sample = dict(source_sample)
            meta = dict(sample.get("meta", {}))
            meta["recording_id"] = recording_id
            meta.setdefault("processed_source_index", source_index)
            meta["map_name"] = sample_map_name(sample, path)
            sample["meta"] = meta
            samples.append(sample)
        sources.append(str(path))

    if not samples:
        raise ValueError(
            "no training samples remain after held-out and qualification exclusions"
        )
    return samples, sources, excluded_sources


def group_samples(
    samples: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for index, sample in enumerate(samples):
        meta = sample.get("meta", {})
        recording_id = str(meta.get("recording_id", ""))
        episode = meta.get("episode_id", meta.get("goal_stamp"))
        if not recording_id:
            raise KeyError(f"sample {index} has no recording_id")
        if episode is None:
            raise KeyError(
                f"sample {index} has no episode_id or goal_stamp; "
                "episode-safe splitting is impossible"
            )
        groups[(recording_id, repr(episode))].append(sample)
    if len(groups) < 3:
        raise ValueError(
            "train/val/test splitting needs at least 3 independent RViz-goal "
            f"episodes, but found {len(groups)}"
        )
    return groups


def score(counts: list[int], targets: list[float]) -> float:
    return sum(
        ((count - target) / max(target, 1.0)) ** 2
        for count, target in zip(counts, targets)
    )


def assign_groups(
    groups: dict[tuple[str, str], list[dict[str, Any]]],
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[tuple[str, str]]]]:
    rng = random.Random(seed)
    ordered = list(groups)
    rng.shuffle(ordered)
    targets = [sum(map(len, groups.values())) * ratio for ratio in ratios]
    assignments: dict[str, list[tuple[str, str]]] = {
        split: [] for split in SPLITS
    }

    if len(ordered) <= 12:
        sizes = [len(groups[group]) for group in ordered]
        best: tuple[int, ...] | None = None
        best_score = float("inf")
        for candidate_assignment in product(range(3), repeat=len(ordered)):
            if len(set(candidate_assignment)) != 3:
                continue
            counts = [0, 0, 0]
            for split_index, size in zip(candidate_assignment, sizes):
                counts[split_index] += size
            candidate_score = score(counts, targets)
            if candidate_score < best_score:
                best = candidate_assignment
                best_score = candidate_score
        if best is None:
            raise RuntimeError("could not create non-empty train/val/test splits")
        for group, split_index in zip(ordered, best):
            assignments[SPLITS[split_index]].append(group)
    else:
        ordered.sort(key=lambda key: len(groups[key]), reverse=True)
        counts = [0, 0, 0]
        for position, group in enumerate(ordered):
            remaining = len(ordered) - position
            empty = [idx for idx, count in enumerate(counts) if count == 0]
            candidates = empty if remaining == len(empty) else list(range(3))
            group_size = len(groups[group])
            selected = min(
                candidates,
                key=lambda idx: score(
                    [
                        count + (group_size if candidate == idx else 0)
                        for candidate, count in enumerate(counts)
                    ],
                    targets,
                ),
            )
            assignments[SPLITS[selected]].append(group)
            counts[selected] += group_size

    split_samples = {split: [] for split in SPLITS}
    for split_index, split in enumerate(SPLITS):
        for group in assignments[split]:
            split_samples[split].extend(groups[group])
        random.Random(seed + split_index).shuffle(split_samples[split])
    return split_samples, assignments


def validate_no_leakage(
    assignments: dict[str, list[tuple[str, str]]],
) -> None:
    group_sets = {split: set(values) for split, values in assignments.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = group_sets[left] & group_sets[right]
        if overlap:
            raise RuntimeError(f"episode leakage between {left} and {right}: {overlap}")


def main() -> int:
    args = parse_args()
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    ratios = (args.train_ratio, args.val_ratio, test_ratio)
    if any(ratio <= 0.0 for ratio in ratios):
        raise ValueError(f"all split ratios must be positive, got {ratios}")

    if args.minimum_pilot_runs < 0:
        raise ValueError("--minimum-pilot-runs must be non-negative")

    held_out_maps = set(args.held_out_map or ["training_dual_route"])
    samples, sources, excluded_sources = load_runs(
        args.input_dir,
        held_out_maps,
        require_qualification=not args.allow_missing_qualification,
        minimum_pilot_runs=args.minimum_pilot_runs,
    )
    groups = group_samples(samples)
    split_samples, assignments = assign_groups(groups, ratios, args.seed)
    validate_no_leakage(assignments)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    held_out_excluded_sources = [
        item for item in excluded_sources if item["reason"] == "held_out_map"
    ]
    quality_excluded_sources = [
        item for item in excluded_sources if item["reason"] != "held_out_map"
    ]
    common_metadata = {
        "format": "moai_gazebo_episode_split_v1",
        "sources": sources,
        "held_out_maps": sorted(held_out_maps),
        "qualification_required": not args.allow_missing_qualification,
        "minimum_pilot_runs": args.minimum_pilot_runs,
        "held_out_excluded_sources": held_out_excluded_sources,
        "quality_excluded_sources": quality_excluded_sources,
        "excluded_sources": excluded_sources,
        "split_strategy": "recording_id_plus_episode_id",
        "split_seed": args.seed,
        "split_ratios": dict(zip(SPLITS, ratios)),
        "total_samples": len(samples),
        "total_episode_groups": len(groups),
    }
    report: dict[str, Any] = {
        "status": "PASS",
        **common_metadata,
        "splits": {},
    }
    for split in SPLITS:
        metadata = {
            **common_metadata,
            "data_partition": split,
            "sample_count": len(split_samples[split]),
            "episode_group_count": len(assignments[split]),
        }
        path = args.out_dir / f"{args.prefix}_{split}.pkl"
        with path.open("wb") as stream:
            pickle.dump(
                {"samples": split_samples[split], "metadata": metadata},
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        report["splits"][split] = {
            "path": str(path),
            "samples": len(split_samples[split]),
            "episode_groups": len(assignments[split]),
        }
        print(
            f"{split}: {len(split_samples[split])} samples, "
            f"{len(assignments[split])} episode groups -> {path}"
        )

    report_path = args.out_dir / "split_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("episode leakage: 0")
    print(f"held-out sources excluded: {len(held_out_excluded_sources)}")
    print(f"quality sources excluded: {len(quality_excluded_sources)}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
    return parser.parse_args()


def load_runs(input_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    paths = sorted(input_dir.rglob("*_clean_social.pkl"))
    if not paths:
        raise FileNotFoundError(f"no *_clean_social.pkl files below {input_dir}")

    samples: list[dict[str, Any]] = []
    sources: list[str] = []
    for path in paths:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        run_samples = list(payload.get("samples", []))
        if not run_samples:
            print(f"warning: skipping empty processed run: {path}")
            continue
        recording_id = path.name.removesuffix("_clean_social.pkl")
        for source_index, source_sample in enumerate(run_samples):
            sample = dict(source_sample)
            meta = dict(sample.get("meta", {}))
            meta["recording_id"] = recording_id
            meta.setdefault("processed_source_index", source_index)
            sample["meta"] = meta
            samples.append(sample)
        sources.append(str(path))

    if not samples:
        raise ValueError("all processed run files were empty")
    return samples, sources


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

    samples, sources = load_runs(args.input_dir)
    groups = group_samples(samples)
    split_samples, assignments = assign_groups(groups, ratios, args.seed)
    validate_no_leakage(assignments)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    common_metadata = {
        "format": "moai_gazebo_episode_split_v1",
        "sources": sources,
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
    print(f"episode leakage: 0")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


CANONICAL_TEST_FILES = (
    "eth_test.pkl",
    "hotel_test.pkl",
    "univ_test.pkl",
    "zara1_test.pkl",
    "zara2_test.pkl",
)
SUPPLEMENTAL_SCENES = (
    "students001",
    "students003",
    "uni_examples",
    "zara3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one duplicate-free ETH/UCY pretraining split."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="ethucy_all")
    return parser.parse_args()


def copy_scene_assets(input_dir: Path, output_dir: Path, scenes: list[str]) -> None:
    shutil.copy2(input_dir / "scales.yml", output_dir / "scales.yml")
    for scene in scenes:
        shutil.copy2(input_dir / f"{scene}_H.txt", output_dir / f"{scene}_H.txt")
        destination = output_dir / scene
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_dir / scene / "oracle.png", destination / "oracle.png")


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical_parts = [
        pd.read_pickle(input_dir / filename) for filename in CANONICAL_TEST_FILES
    ]
    supplemental_source = pd.read_pickle(input_dir / "eth_train.pkl")
    supplemental = supplemental_source[
        supplemental_source["sceneId"].isin(SUPPLEMENTAL_SCENES)
    ].copy()
    combined = pd.concat([*canonical_parts, supplemental], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["sceneId", "metaId", "trackId", "frame", "x", "y"]
    ).reset_index(drop=True)

    scenes = sorted(combined["sceneId"].unique().tolist())
    expected_scenes = sorted(
        ["eth", "hotel", "univ", "zara1", "zara2", *SUPPLEMENTAL_SCENES]
    )
    if scenes != expected_scenes:
        raise ValueError(f"Unexpected scene set: {scenes}; expected {expected_scenes}")

    train_path = output_dir / f"{args.prefix}_train.pkl"
    combined.to_pickle(train_path)
    copy_scene_assets(input_dir, output_dir, scenes)

    report = {
        "format": "ethucy_all_pretrain_v1",
        "source_dir": str(input_dir),
        "output": str(train_path),
        "rows": len(combined),
        "scenes": scenes,
        "scene_rows": {
            str(scene): int(count)
            for scene, count in combined.groupby("sceneId").size().items()
        },
        "duplicate_rows_after_merge": int(
            combined.duplicated(
                subset=["sceneId", "metaId", "trackId", "frame", "x", "y"]
            ).sum()
        ),
    }
    (output_dir / "ethucy_all_pretrain_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

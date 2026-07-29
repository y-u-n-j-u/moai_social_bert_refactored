#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.loader import load_runtime_namespace
from src.data_loader import _build_dataset
from src.utils import set_workdir


EXPECTED_SHAPES = {
    "mgp_spatial_ids": (58, 2),
    "mgp_segment_ids": (58,),
    "mgp_temporal_ids": (58,),
    "mgp_attn_mask": (58,),
    "tgp_spatial_ids": (58, 2),
    "tgp_segment_ids": (58,),
    "tgp_temporal_ids": (58,),
    "tgp_attn_mask": (58,),
    "traj_lbl": (12, 2),
    "goal_lbl": (2,),
    "guidance_lbl": (2,),
    "env_spatial_ids": (4, 256),
    "env_segment_ids": (4,),
    "env_temporal_ids": (4,),
    "env_attn_mask": (4,),
    "envs": (32, 32),
    "envs_params": (6,),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate every Gazebo sample against the guided SPU-BERT contract."
    )
    parser.add_argument(
        "--config",
        default="configs/spubert/moai_social_nav_ext_scene_smoke_fs.yaml",
    )
    return parser.parse_args()


def validate_sample(sample: dict[str, torch.Tensor], view_range: float) -> dict[str, bool]:
    for key, expected in EXPECTED_SHAPES.items():
        if key not in sample:
            raise KeyError(f"missing key: {key}")
        if tuple(sample[key].shape) != expected:
            raise AssertionError(f"{key}: expected {expected}, got {tuple(sample[key].shape)}")
        if sample[key].is_floating_point() and not torch.isfinite(sample[key]).all():
            raise AssertionError(f"{key} contains NaN or Inf")

    pad = sample["mgp_spatial_ids"].new_tensor([-view_range, -view_range])
    mask = sample["mgp_spatial_ids"].new_tensor([view_range, view_range])
    if not torch.allclose(sample["goal_lbl"], sample["traj_lbl"][-1]):
        raise AssertionError("goal_lbl != traj_lbl[-1]")
    if not torch.allclose(sample["mgp_spatial_ids"][9:20], pad.expand(11, -1)):
        raise AssertionError("MGP PAD contract failed")
    if not torch.allclose(sample["mgp_spatial_ids"][20], mask):
        raise AssertionError("MGP endpoint MASK contract failed")
    if not torch.allclose(sample["mgp_spatial_ids"][21], sample["guidance_lbl"]):
        raise AssertionError("MGP guidance token contract failed")
    if not torch.allclose(sample["tgp_spatial_ids"][9:21], mask.expand(12, -1)):
        raise AssertionError("TGP MASK contract failed")
    if not torch.allclose(sample["tgp_spatial_ids"][21], sample["goal_lbl"]):
        raise AssertionError("TGP endpoint contract failed")

    map_values = set(torch.unique(sample["envs"]).tolist())
    if not map_values.issubset({0.0, 1.0, 2.0}):
        raise AssertionError(f"unexpected map classes: {sorted(map_values)}")
    half_width = -float(sample["envs_params"][0])
    half_height = -float(sample["envs_params"][1])
    guidance_outside = bool(
        abs(float(sample["guidance_lbl"][0])) >= half_width
        or abs(float(sample["guidance_lbl"][1])) >= half_height
    )
    endpoint_outside = bool(
        abs(float(sample["goal_lbl"][0])) >= half_width
        or abs(float(sample["goal_lbl"][1])) >= half_height
    )
    return {
        "guidance_differs_from_endpoint": not torch.allclose(
            sample["guidance_lbl"], sample["goal_lbl"]
        ),
        "guidance_outside_map": guidance_outside,
        "endpoint_outside_map": endpoint_outside,
    }


def main() -> int:
    cli = parse_args()
    set_workdir()
    args = load_runtime_namespace(cli.config, command="train", cli_dry_run=False)
    report: dict[str, object] = {
        "status": "PASS",
        "contract": {
            "mgp_mask_index": 20,
            "mgp_guidance_index": 21,
            "tgp_goal_index": 21,
            "map_values": {"unknown": 0, "free": 1, "occupied": 2},
        },
        "splits": {},
    }
    total = 0
    differs = 0
    guidance_outside = 0
    endpoint_outside = 0
    for split in ("train", "val", "test"):
        dataset = _build_dataset(args, split=split)
        split_stats = {
            "samples": len(dataset),
            "guidance_differs_from_endpoint": 0,
            "guidance_outside_map": 0,
            "endpoint_outside_map": 0,
        }
        for index in range(len(dataset)):
            flags = validate_sample(dataset[index], view_range=float(args.view_range))
            for key, value in flags.items():
                split_stats[key] += int(value)
        report["splits"][split] = split_stats
        total += split_stats["samples"]
        differs += split_stats["guidance_differs_from_endpoint"]
        guidance_outside += split_stats["guidance_outside_map"]
        endpoint_outside += split_stats["endpoint_outside_map"]

    report["totals"] = {
        "samples": total,
        "guidance_differs_from_endpoint": differs,
        "guidance_outside_map": guidance_outside,
        "endpoint_outside_map": endpoint_outside,
    }
    print("[GUIDED_DATASET_VALIDATION]")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

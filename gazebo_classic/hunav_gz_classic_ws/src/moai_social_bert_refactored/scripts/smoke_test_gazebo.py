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
from src.data_loader import build_loaders
from src.trainer import build_trainer_class
from src.utils import NullWriter, _to_device, bootstrap_paths, set_seed, set_workdir

bootstrap_paths()

from spubert.training import _spubert_finetune_batch_kwargs


EXPECTED_BATCH_SHAPES = {
    "mgp_spatial_ids": (58, 2),
    "tgp_spatial_ids": (58, 2),
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
        description="Run one real Gazebo scene-aware SPU-BERT training step."
    )
    parser.add_argument(
        "--config",
        default="configs/spubert/moai_social_nav_ext_scene_smoke_fs.yaml",
    )
    return parser.parse_args()


def assert_batch_contract(batch: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    batch_size = int(batch["mgp_spatial_ids"].shape[0])
    for key, sample_shape in EXPECTED_BATCH_SHAPES.items():
        if key not in batch:
            raise KeyError(f"Missing required batch key: {key}")
        actual = tuple(batch[key].shape)
        expected = (batch_size, *sample_shape)
        if actual != expected:
            raise AssertionError(f"{key}: expected {expected}, got {actual}")
        if batch[key].is_floating_point() and not torch.isfinite(batch[key]).all():
            raise AssertionError(f"{key} contains NaN or Inf")
        shapes[key] = list(actual)

    map_values = set(torch.unique(batch["envs"]).detach().cpu().tolist())
    if not map_values.issubset({0.0, 1.0, 2.0}):
        raise AssertionError(f"Unexpected map classes: {sorted(map_values)}")

    view_range = float(batch["mgp_spatial_ids"].new_tensor(20.0).item())
    pad_token = batch["mgp_spatial_ids"].new_tensor([-view_range, -view_range])
    mask_token = batch["mgp_spatial_ids"].new_tensor([view_range, view_range])
    if not torch.allclose(batch["goal_lbl"], batch["traj_lbl"][:, -1]):
        raise AssertionError("goal_lbl must equal the 12-step trajectory endpoint")
    expected_mgp_pad = pad_token.view(1, 1, -1).expand(batch_size, 11, -1)
    expected_tgp_mask = mask_token.view(1, 1, -1).expand(batch_size, 12, -1)
    if not torch.allclose(batch["mgp_spatial_ids"][:, 9:20], expected_mgp_pad):
        raise AssertionError("guided MGP indexes 9:20 must contain 11 PAD tokens")
    if not torch.allclose(batch["mgp_spatial_ids"][:, 20], mask_token):
        raise AssertionError("guided MGP index 20 must contain the endpoint MASK")
    if not torch.allclose(batch["mgp_spatial_ids"][:, 21], batch["guidance_lbl"]):
        raise AssertionError("guided MGP index 21 must contain guidance_lbl")
    if not torch.allclose(batch["tgp_spatial_ids"][:, 9:21], expected_tgp_mask):
        raise AssertionError("TGP indexes 9:21 must contain 12 MASK tokens")
    if not torch.allclose(batch["tgp_spatial_ids"][:, 21], batch["goal_lbl"]):
        raise AssertionError("TGP index 21 must contain the 12-step endpoint")
    return shapes


def gradient_stats(module: torch.nn.Module) -> dict[str, float | int]:
    tensors = 0
    elements = 0
    nonzero = 0
    max_abs = 0.0
    for parameter in module.parameters():
        grad = parameter.grad
        if grad is None:
            continue
        if not torch.isfinite(grad).all():
            raise AssertionError("Model gradient contains NaN or Inf")
        tensors += 1
        elements += grad.numel()
        nonzero += int(torch.count_nonzero(grad).item())
        max_abs = max(max_abs, float(grad.detach().abs().max().item()))
    if tensors == 0 or nonzero == 0:
        raise AssertionError("No finite nonzero gradients were produced")
    return {
        "tensors": tensors,
        "elements": elements,
        "nonzero": nonzero,
        "max_abs": max_abs,
    }


def main() -> int:
    cli = parse_args()
    set_workdir()
    args = load_runtime_namespace(cli.config, command="train", cli_dry_run=False)
    if args.mode != "finetune" or args.framework != "spubert":
        raise ValueError("Smoke test requires a SPU-BERT finetune configuration")
    if args.train_mode != "fs":
        raise ValueError("Smoke test must use train_mode=fs and require no checkpoint")
    if not args.scene:
        raise ValueError("Smoke test must use scene.enabled=true")
    if args.col_weight != 0:
        raise ValueError("Keep col_weight=0 until collision-map semantics are fixed")
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("runtime.cuda=true but CUDA is unavailable")

    set_seed(args.seed, use_cuda=args.cuda)
    train_loader, _, _ = build_loaders(args, for_test=False)
    trainer_class = build_trainer_class(args)
    trainer = trainer_class(
        train_dataloader=train_loader,
        val_dataloader=None,
        args=args,
        tb_writer=NullWriter(),
    )

    batch = _to_device(next(iter(train_loader)), trainer.device)
    input_shapes = assert_batch_contract(batch)

    trainer.mgp_optim.zero_grad(set_to_none=True)
    trainer.tgp_optim.zero_grad(set_to_none=True)
    outputs = trainer.model(
        **_spubert_finetune_batch_kwargs(
            batch,
            scene=args.scene,
            kld_weight=torch.tensor(args.kld_weight, device=trainer.device),
            traj_weight=torch.tensor(args.traj_weight, device=trainer.device),
            goal_weight=torch.tensor(args.goal_weight, device=trainer.device),
        )
    )

    expected_output_shapes = {
        "pred_goals": (args.batch_size, args.k_sample, args.goal_dim),
        "pred_trajs": (args.batch_size, args.pred_len, args.output_dim),
    }
    for key, expected in expected_output_shapes.items():
        actual = tuple(outputs[key].shape)
        if actual != expected:
            raise AssertionError(f"{key}: expected {expected}, got {actual}")
        if not torch.isfinite(outputs[key]).all():
            raise AssertionError(f"{key} contains NaN or Inf")

    losses = {
        key: float(outputs[key].mean().detach().cpu().item())
        for key in ("mgp_loss", "tgp_loss", "kld_loss", "gde_loss", "ade_loss", "fde_loss")
    }
    if not all(torch.isfinite(outputs[key]).all() for key in losses):
        raise AssertionError(f"Non-finite loss detected: {losses}")

    total_loss = outputs["mgp_loss"].mean() + outputs["tgp_loss"].mean()
    total_loss.backward()
    mgp_gradients = gradient_stats(trainer.model.mgp_model)
    tgp_gradients = gradient_stats(trainer.model.tgp_model)
    trainer.mgp_optim.step()
    trainer.tgp_optim.step()

    inference_model = trainer.model.module if trainer.parallel else trainer.model
    inference_model.eval()
    with torch.no_grad():
        guided_outputs = inference_model.inference_guided(
            mgp_spatial_ids=batch["mgp_spatial_ids"],
            mgp_temporal_ids=batch["mgp_temporal_ids"],
            mgp_segment_ids=batch["mgp_segment_ids"],
            mgp_attn_mask=batch["mgp_attn_mask"],
            tgp_temporal_ids=batch["tgp_temporal_ids"],
            tgp_segment_ids=batch["tgp_segment_ids"],
            tgp_attn_mask=batch["tgp_attn_mask"],
            guidance_points=batch["guidance_lbl"],
            env_spatial_ids=batch["env_spatial_ids"],
            env_temporal_ids=batch["env_temporal_ids"],
            env_segment_ids=batch["env_segment_ids"],
            env_attn_mask=batch["env_attn_mask"],
            envs=batch["envs"],
            envs_params=batch["envs_params"],
            d_sample=args.d_sample,
            reject_unknown=args.reject_unknown_goals,
        )
        multi_candidate_outputs = inference_model.inference(
            mgp_spatial_ids=batch["mgp_spatial_ids"],
            mgp_temporal_ids=batch["mgp_temporal_ids"],
            mgp_segment_ids=batch["mgp_segment_ids"],
            mgp_attn_mask=batch["mgp_attn_mask"],
            tgp_temporal_ids=batch["tgp_temporal_ids"],
            tgp_segment_ids=batch["tgp_segment_ids"],
            tgp_attn_mask=batch["tgp_attn_mask"],
            env_spatial_ids=batch["env_spatial_ids"],
            env_temporal_ids=batch["env_temporal_ids"],
            env_segment_ids=batch["env_segment_ids"],
            env_attn_mask=batch["env_attn_mask"],
            envs=batch["envs"],
            d_sample=args.d_sample,
        )
    guided_expected_shapes = {
        "pred_trajs": (args.batch_size, args.pred_len, args.output_dim),
        "pred_goals": (args.batch_size, args.goal_dim),
        "candidate_goals": (args.batch_size, args.k_sample, args.goal_dim),
        "candidate_safe_mask": (args.batch_size, args.k_sample),
        "selected_indices": (args.batch_size,),
        "selected_goal_valid": (args.batch_size,),
        "trajectory_point_safe_mask": (args.batch_size, args.pred_len),
        "trajectory_map_safe": (args.batch_size,),
        "execution_valid": (args.batch_size,),
    }
    for key, expected in guided_expected_shapes.items():
        actual = tuple(guided_outputs[key].shape)
        if actual != expected:
            raise AssertionError(f"guided {key}: expected {expected}, got {actual}")
    if not torch.isfinite(guided_outputs["pred_trajs"]).all():
        raise AssertionError("guided pred_trajs contains NaN or Inf")
    if not torch.isfinite(guided_outputs["pred_goals"]).all():
        raise AssertionError("guided pred_goals contains NaN or Inf")
    multi_candidate_expected_shapes = {
        "pred_goals": (
            args.batch_size,
            args.k_sample,
            args.goal_dim,
        ),
        "pred_trajs": (
            args.batch_size,
            args.k_sample,
            args.pred_len,
            args.output_dim,
        ),
    }
    for key, expected in multi_candidate_expected_shapes.items():
        actual = tuple(multi_candidate_outputs[key].shape)
        if actual != expected:
            raise AssertionError(
                f"multi-candidate {key}: expected {expected}, got {actual}"
            )
        if not torch.isfinite(multi_candidate_outputs[key]).all():
            raise AssertionError(f"multi-candidate {key} contains NaN or Inf")
    valid_rows = guided_outputs["selected_goal_valid"]
    if valid_rows.any():
        selected = guided_outputs["selected_indices"][valid_rows]
        safe = guided_outputs["candidate_safe_mask"][valid_rows]
        if not safe[torch.arange(len(selected), device=safe.device), selected].all():
            raise AssertionError("guided selector chose an unsafe candidate")
    invalid_rows = ~valid_rows
    if invalid_rows.any():
        if not torch.all(guided_outputs["selected_indices"][invalid_rows] == -1):
            raise AssertionError("all-invalid rows must use selected index -1")
        if not torch.allclose(
            guided_outputs["pred_goals"][invalid_rows],
            torch.zeros_like(guided_outputs["pred_goals"][invalid_rows]),
        ):
            raise AssertionError("all-invalid rows must return the stop goal [0,0]")
        if not torch.allclose(
            guided_outputs["pred_trajs"][invalid_rows],
            torch.zeros_like(guided_outputs["pred_trajs"][invalid_rows]),
        ):
            raise AssertionError("all-invalid rows must return a zero stop trajectory")
    expected_execution_valid = (
        guided_outputs["selected_goal_valid"]
        & guided_outputs["trajectory_map_safe"]
    )
    if not torch.equal(guided_outputs["execution_valid"], expected_execution_valid):
        raise AssertionError(
            "execution_valid must require both a valid goal and a map-safe trajectory"
        )
    stopped_rows = ~guided_outputs["execution_valid"]
    if stopped_rows.any() and not torch.allclose(
        guided_outputs["pred_trajs"][stopped_rows],
        torch.zeros_like(guided_outputs["pred_trajs"][stopped_rows]),
    ):
        raise AssertionError("execution-invalid rows must return a zero stop trajectory")

    if trainer.device.type == "cuda":
        torch.cuda.synchronize()
    report = {
        "status": "PASS",
        "device": str(trainer.device),
        "gpu": torch.cuda.get_device_name(0) if trainer.device.type == "cuda" else None,
        "dataset_samples": len(train_loader.dataset),
        "batch_size": int(batch["mgp_spatial_ids"].shape[0]),
        "input_shapes": input_shapes,
        "output_shapes": {
            key: list(outputs[key].shape) for key in ("pred_goals", "pred_trajs")
        },
        "guided_output_shapes": {
            key: list(guided_outputs[key].shape)
            for key in ("candidate_goals", "pred_goals", "pred_trajs")
        },
        "multi_candidate_output_shapes": {
            key: list(multi_candidate_outputs[key].shape)
            for key in ("pred_goals", "pred_trajs")
        },
        "guided_selection": {
            "safe_candidates": int(guided_outputs["candidate_safe_mask"].sum().item()),
            "total_candidates": int(guided_outputs["candidate_safe_mask"].numel()),
            "valid_selections": int(guided_outputs["selected_goal_valid"].sum().item()),
            "all_invalid": int(guided_outputs["all_candidates_invalid"].sum().item()),
            "map_safe_trajectories": int(
                guided_outputs["trajectory_map_safe"].sum().item()
            ),
            "execution_valid": int(guided_outputs["execution_valid"].sum().item()),
        },
        "losses": losses,
        "gradients": {
            "mgp": mgp_gradients,
            "tgp": tgp_gradients,
        },
        "cuda_memory_allocated_mib": (
            round(torch.cuda.max_memory_allocated() / 1024**2, 2)
            if trainer.device.type == "cuda"
            else 0.0
        ),
    }
    print("[GAZEBO_SMOKE_TEST]")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

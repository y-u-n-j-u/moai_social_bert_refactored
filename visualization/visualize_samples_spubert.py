import argparse
import math
import random
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils import bootstrap_paths

bootstrap_paths()

from spubert.datasets.ethucy import ETHUCYDataset
from spubert.datasets.ethucy_sbert import ETHUCYSBertDataset
from spubert.datasets.ethucy_star import ETHUCYSTARDataset
from spubert.datasets.ethucy_tpp import ETHUCYTPPDataset
from spubert.datasets.jrdb import JRDBDataset as SPUBertJRDBDataset
from spubert.datasets.sdd_sbert import SDDSBertDataset
from spubert.datasets.util import rotate_trajs, transform_to_target
from spubert.model import (
    SBertPlusFTConfig,
    SBertPlusFTModel,
    SBertPlusMGPConfig,
    SBertPlusTGPConfig,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Sample trajectory visualization for SPU-BERT datasets.")
    parser.add_argument("--dataset_path", default="./data")
    parser.add_argument("--dataset_name", default="ethucy_sbert")
    parser.add_argument("--dataset_split", default="univ")
    parser.add_argument("--split", default="test")

    parser.add_argument("--obs_len", type=int, default=8)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--min_obs_len", type=int, default=2)
    parser.add_argument("--num_nbr", type=int, default=4)
    parser.add_argument("--view_range", type=float, default=20.0)
    parser.add_argument("--view_angle", type=float, default=2.09)
    parser.add_argument("--social_range", type=float, default=2.0)
    parser.add_argument("--num_samples", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20)

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--head", type=int, default=4)
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--act_fn", default="gelu")
    parser.add_argument("--goal_hidden", type=int, default=64)
    parser.add_argument("--goal_latent", type=int, default=32)
    parser.add_argument("--goal_dim", type=int, default=2)
    parser.add_argument("--k_sample", type=int, default=20)
    parser.add_argument("--d_sample", type=int, default=400)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--normal", action="store_true")
    parser.add_argument("--best_by", choices=["ade", "fde", "first"], default="ade")
    parser.add_argument("--plot_k", action="store_true")
    parser.add_argument("--plot_gridmap", action="store_true")
    parser.add_argument("--cuda", action="store_true")

    parser.add_argument("--scene", action="store_true")
    parser.add_argument("--env_range", type=float, default=10.0)
    parser.add_argument("--env_resol", type=float, default=0.2)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--num_patch", type=int, default=0)
    parser.add_argument("--binary_scene", action="store_true")

    parser.add_argument("--subsample_stride", type=int, default=3, help="JRDB only")
    parser.add_argument("--traj_scale", type=float, default=1.0, help="JRDB only")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_dataset_args(args):
    return SimpleNamespace(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        min_obs_len=args.min_obs_len,
        num_nbr=args.num_nbr,
        view_range=args.view_range,
        view_angle=args.view_angle,
        social_range=args.social_range,
        env_range=args.env_range,
        env_resol=args.env_resol,
        patch_size=args.patch_size,
        binary_scene=args.binary_scene,
        mode="finetune",
        train_mode="fs",
        input_dim=2,
        goal_dim=args.goal_dim,
        output_dim=2,
        aug=False,
        viz=False,
        scene=args.scene,
        subsample_stride=args.subsample_stride,
        traj_scale=args.traj_scale,
    )


def _load_dataset(args):
    ds_args = _build_dataset_args(args)
    name = args.dataset_name

    if name == "ethucy":
        dataset = ETHUCYDataset(split=args.split, args=ds_args)
    elif name == "ethucy_tpp":
        dataset = ETHUCYTPPDataset(split=args.split, args=ds_args)
    elif name == "ethucy_star":
        dataset = ETHUCYSTARDataset(split=args.split, args=ds_args)
    elif name == "ethucy_sbert":
        dataset = ETHUCYSBertDataset(split=args.split, args=ds_args)
    elif name == "sdd_sbert":
        ds_args.dataset_split = "default"
        dataset = SDDSBertDataset(split=args.split, args=ds_args)
    elif name == "JRDB":
        dataset = SPUBertJRDBDataset(split=args.split, args=ds_args)
    else:
        raise ValueError(f"Unsupported dataset_name for spubert: {name}")

    if len(dataset) == 0:
        raise RuntimeError("Loaded dataset has zero samples.")
    return dataset


def _load_model(args, device):
    state_dict = torch.load(args.checkpoint, map_location=device)
    num_patch = args.num_patch
    if args.scene and num_patch <= 0:
        token_type_key = "tgp_model.sbert.hf_encoder.backbone.model.embeddings.token_type_embeddings.weight"
        if token_type_key in state_dict:
            type_vocab_size = int(state_dict[token_type_key].shape[0])
            num_patch = max(0, type_vocab_size - args.num_nbr - 2)
    tgp_cfg = SBertPlusTGPConfig(
        input_dim=2,
        output_dim=2,
        goal_dim=args.goal_dim,
        hidden_size=args.hidden,
        num_layer=args.layer,
        num_head=args.head,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        num_nbr=args.num_nbr,
        scene=args.scene,
        num_patch=num_patch,
        patch_size=args.patch_size,
        dropout_prob=args.dropout_prob,
        col_weight=0.0,
        traj_weight=1.0,
        act_fn=args.act_fn,
        view_range=args.view_range,
        view_angle=args.view_angle,
        social_range=args.social_range,
    )
    mgp_cfg = SBertPlusMGPConfig(
        input_dim=2,
        output_dim=2,
        goal_dim=args.goal_dim,
        hidden_size=args.hidden,
        num_layer=args.layer,
        num_head=args.head,
        k_sample=args.k_sample,
        goal_hidden_size=args.goal_hidden,
        goal_latent_size=args.goal_latent,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        num_nbr=args.num_nbr,
        scene=args.scene,
        num_patch=num_patch,
        patch_size=args.patch_size,
        dropout_prob=args.dropout_prob,
        col_weight=0.0,
        goal_weight=1.0,
        kld_weight=1.0,
        cvae_sigma=1.0,
        kld_clamp=None,
        act_fn=args.act_fn,
        normal=args.normal,
        view_range=args.view_range,
        view_angle=args.view_angle,
        social_range=args.social_range,
    )
    ft_cfg = SBertPlusFTConfig(traj_cfgs=tgp_cfg, goal_cfgs=mgp_cfg, share=args.share)
    model = SBertPlusFTModel(tgp_cfg, mgp_cfg, ft_cfg)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _prepare_plus_inputs_from_spatial(item, args):
    mgp_spatial = item["spatial_ids"].clone()
    mgp_segment = item["segment_ids"].clone()
    mgp_temporal = item["temporal_ids"].clone()
    mgp_attn = item["attn_mask"].clone()

    tgp_segment = item["segment_ids"].clone()
    tgp_temporal = item["temporal_ids"].clone()
    tgp_attn = item["attn_mask"].clone()

    start = 1 + args.obs_len
    end = start + args.pred_len
    pad_token = mgp_spatial.new_full((mgp_spatial.shape[-1],), -args.view_range)
    msk_token = mgp_spatial.new_full((mgp_spatial.shape[-1],), args.view_range)

    if args.pred_len > 1:
        mgp_spatial[start : end - 1] = pad_token
        mgp_segment[start : end - 1] = 0
        mgp_temporal[start : end - 1] = 0
        mgp_attn[start : end - 1] = 0

    mgp_spatial[end - 1] = msk_token
    mgp_segment[end - 1] = 1
    mgp_temporal[end - 1] = args.obs_len + args.pred_len
    mgp_attn[end - 1] = 1.0

    return {
        "mgp_spatial_ids": mgp_spatial,
        "mgp_temporal_ids": mgp_temporal,
        "mgp_segment_ids": mgp_segment,
        "mgp_attn_mask": mgp_attn,
        "tgp_temporal_ids": tgp_temporal,
        "tgp_segment_ids": tgp_segment,
        "tgp_attn_mask": tgp_attn,
    }


def _pick_best(pred_trajs: np.ndarray, gt_local: np.ndarray, mode: str):
    if mode == "first":
        return 0
    d = np.linalg.norm(pred_trajs - gt_local[None, :, :], axis=-1)
    if mode == "fde":
        return int(np.argmin(d[:, -1]))
    return int(np.argmin(d.mean(axis=1)))


def _local_to_world(raw_sample: np.ndarray, pred_local: np.ndarray, obs_len: int):
    _, center, theta = transform_to_target(
        trajs=np.array(raw_sample, dtype=np.float32, copy=True),
        obs_len=obs_len,
        traj_dim=raw_sample.shape[2],
    )
    pred_world = rotate_trajs(
        trajs=np.expand_dims(pred_local, axis=0),
        theta=-theta,
        traj_dim=2,
    )[0] - center[np.newaxis, :]
    return pred_world


def _collect_samples(args):
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    dataset = _load_dataset(args)
    random.seed(args.seed)
    n = min(args.num_samples, len(dataset))
    sampled_idx = random.sample(list(range(len(dataset))), n)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    model = _load_model(args, device)

    samples = []
    with torch.no_grad():
        for idx in sampled_idx:
            item = dataset[idx]
            raw_sample = np.array(dataset.all_trajs[idx], dtype=np.float32, copy=True)
            raw_tgt = raw_sample[0].copy()
            obs_world = raw_tgt[: args.obs_len].copy()
            gt_world = raw_tgt[args.obs_len : args.obs_len + args.pred_len].copy()
            local_sample, _, _ = transform_to_target(
                trajs=np.array(raw_sample, dtype=np.float32, copy=True),
                obs_len=args.obs_len,
                traj_dim=raw_sample.shape[2],
            )
            local_tgt = local_sample[0].copy()
            scale = float(item["scales"].item())
            obs_local = local_tgt[: args.obs_len].copy() * scale
            gt_local_plot = local_tgt[args.obs_len : args.obs_len + args.pred_len].copy() * scale

            if "mgp_spatial_ids" in item:
                plus_inputs = {
                    "mgp_spatial_ids": item["mgp_spatial_ids"],
                    "mgp_temporal_ids": item["mgp_temporal_ids"],
                    "mgp_segment_ids": item["mgp_segment_ids"],
                    "mgp_attn_mask": item["mgp_attn_mask"],
                    "tgp_temporal_ids": item["tgp_temporal_ids"],
                    "tgp_segment_ids": item["tgp_segment_ids"],
                    "tgp_attn_mask": item["tgp_attn_mask"],
                }
            else:
                plus_inputs = _prepare_plus_inputs_from_spatial(item, args)

            batched = {k: v.unsqueeze(0).to(device) for k, v in plus_inputs.items()}
            env_spatial_ids = item["env_spatial_ids"].unsqueeze(0).to(device) if "env_spatial_ids" in item else None
            env_temporal_ids = item["env_temporal_ids"].unsqueeze(0).to(device) if "env_temporal_ids" in item else None
            env_segment_ids = item["env_segment_ids"].unsqueeze(0).to(device) if "env_segment_ids" in item else None
            env_attn_mask = item["env_attn_mask"].unsqueeze(0).to(device) if "env_attn_mask" in item else None
            envs = item["envs"].unsqueeze(0).to(device) if "envs" in item else None

            outputs = model.inference(
                mgp_spatial_ids=batched["mgp_spatial_ids"],
                mgp_temporal_ids=batched["mgp_temporal_ids"],
                mgp_segment_ids=batched["mgp_segment_ids"],
                mgp_attn_mask=batched["mgp_attn_mask"],
                tgp_temporal_ids=batched["tgp_temporal_ids"],
                tgp_segment_ids=batched["tgp_segment_ids"],
                tgp_attn_mask=batched["tgp_attn_mask"],
                env_spatial_ids=env_spatial_ids,
                env_temporal_ids=env_temporal_ids,
                env_segment_ids=env_segment_ids,
                env_attn_mask=env_attn_mask,
                envs=envs,
                d_sample=args.d_sample,
                output_attentions=False,
            )

            pred_local_all = outputs["pred_trajs"][0].detach().cpu().numpy() * scale
            gt_local = item["traj_lbl"].detach().cpu().numpy() * scale
            best_idx = _pick_best(pred_local_all, gt_local, args.best_by)
            pred_world = _local_to_world(raw_sample, pred_local_all[best_idx], args.obs_len)

            pred_all_world = None
            if args.plot_k:
                pred_all_world = [
                    _local_to_world(raw_sample, pred_local_all[k], args.obs_len)
                    for k in range(pred_local_all.shape[0])
                ]

            obs_plot = obs_world
            gt_plot = gt_world
            pred_plot = pred_world
            pred_all_plot = pred_all_world
            gridmap = None
            grid_extent = None
            if args.plot_gridmap:
                if "envs" not in item or "envs_params" not in item:
                    raise ValueError("--plot_gridmap requires scene-enabled samples with envs and envs_params")
                min_x, min_y, width, height, resolution = item["envs_params"].detach().cpu().numpy()[:5]
                gridmap = item["envs"].detach().cpu().numpy()
                grid_extent = [
                    float(min_x * scale),
                    float((min_x + width * resolution) * scale),
                    float(min_y * scale),
                    float((min_y + height * resolution) * scale),
                ]
                obs_plot = obs_local
                gt_plot = gt_local_plot
                pred_plot = pred_local_all[best_idx]
                pred_all_plot = [pred_local_all[k] for k in range(pred_local_all.shape[0])] if args.plot_k else None

            scene_id = dataset.all_scenes[idx] if idx < len(dataset.all_scenes) else "unknown_scene"
            title = f"spubert | {args.dataset_name} | scene={scene_id} | idx={idx}"
            if args.plot_gridmap:
                title += " | local+map"
            samples.append(
                {
                    "obs": obs_plot,
                    "gt": gt_plot,
                    "pred": pred_plot,
                    "pred_all": pred_all_plot,
                    "gridmap": gridmap,
                    "grid_extent": grid_extent,
                    "title": title,
                }
            )
    return samples


def plot_samples(samples, out_path: Path):
    n = len(samples)
    cols = min(3, max(1, n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.5 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    else:
        axes = [ax for row_axes in axes for ax in row_axes]

    for i, sample in enumerate(samples):
        ax = axes[i]
        obs = sample["obs"]
        gt = sample["gt"]
        pred = sample["pred"]
        pred_all = sample.get("pred_all")
        gridmap = sample.get("gridmap")
        grid_extent = sample.get("grid_extent")

        if gridmap is not None and grid_extent is not None:
            ax.imshow(
                gridmap,
                extent=grid_extent,
                origin="lower",
                cmap="gray_r",
                alpha=0.35,
                interpolation="nearest",
                vmin=0.0,
                vmax=2.0,
                zorder=0,
            )

        ax.plot(obs[:, 0], obs[:, 1], color="royalblue", linewidth=2.0, marker="o", markersize=3, label="Obs")
        ax.plot(gt[:, 0], gt[:, 1], color="crimson", linewidth=2.0, marker="o", markersize=3, label="GT Future")

        if pred_all is not None:
            for j, traj in enumerate(pred_all):
                label = "K samples" if j == 0 else None
                ax.plot(traj[:, 0], traj[:, 1], color="gray", alpha=0.2, linewidth=1.0, label=label)

        ax.plot(pred[:, 0], pred[:, 1], color="black", linewidth=2.0, marker="o", markersize=3, label="Prediction")
        ax.scatter([obs[0, 0]], [obs[0, 1]], color="royalblue", s=35)
        ax.scatter([gt[-1, 0]], [gt[-1, 1]], color="crimson", s=35)
        ax.scatter([pred[-1, 0]], [pred[-1, 1]], color="black", s=35)

        ax.set_title(sample["title"], fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        if grid_extent is not None:
            ax.set_xlim(grid_extent[0], grid_extent[1])
            ax.set_ylim(grid_extent[2], grid_extent[3])
        ax.grid(True, alpha=0.25)

    for i in range(n, len(axes)):
        axes[i].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        uniq = {}
        for h, l in zip(handles, labels):
            if l not in uniq:
                uniq[l] = h
        fig.legend(uniq.values(), uniq.keys(), loc="upper right")
    fig.suptitle("Sampled Trajectories (SPU-BERT)", y=0.995)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved visualization: {out_path}")


def main():
    args = parse_args()
    _set_seed(args.seed)

    if args.out:
        out_path = Path(args.out)
    else:
        base = Path(args.dataset_path) / args.dataset_name
        out_path = base / f"sampled_{args.dataset_name}_spubert.png"

    samples = _collect_samples(args)
    plot_samples(samples, out_path)


if __name__ == "__main__":
    main()

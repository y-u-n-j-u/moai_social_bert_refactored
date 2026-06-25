import argparse
import math
import random
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, Wedge

from src.utils import bootstrap_paths

bootstrap_paths()

from sbert.datasets.ethucy_sbert import ETHUCYSBertDataset
from sbert.datasets.ethucy_star import ETHUCYSTARDataset
from sbert.datasets.jrdb import JRDBDataset
from sbert.datasets.sdd_sbert import SDDSBertDataset
from sbert.datasets.util import rotate_trajs, transform_to_target
from sbert.model import SBertConfig, SBertFTModel


def parse_args():
    parser = argparse.ArgumentParser(description="Sample trajectory visualization for SBert datasets.")
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

    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--head", type=int, default=4)
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--act_fn", default="gelu")
    parser.add_argument("--sip", action="store_true")
    parser.add_argument("--cuda", action="store_true")

    parser.add_argument("--subsample_stride", type=int, default=3, help="JRDB only")
    parser.add_argument("--traj_scale", type=float, default=1.0, help="JRDB only")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_dataset_args(args):
    ds_args = SimpleNamespace(
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
        mode="finetune",
        train_mode="fs",
        input_dim=2,
        output_dim=2,
        aug=False,
        viz=False,
        subsample_stride=args.subsample_stride,
        traj_scale=args.traj_scale,
    )
    return ds_args


def _load_dataset(args):
    ds_args = _build_dataset_args(args)
    name = args.dataset_name

    if name == "ethucy_star":
        dataset = ETHUCYSTARDataset(split=args.split, args=ds_args)
    elif name == "ethucy_sbert":
        dataset = ETHUCYSBertDataset(split=args.split, args=ds_args)
    elif name == "sdd_sbert":
        ds_args.dataset_split = "default"
        dataset = SDDSBertDataset(split=args.split, args=ds_args)
    elif name == "JRDB":
        dataset = JRDBDataset(split=args.split, args=ds_args)
    else:
        raise ValueError(f"Unsupported dataset_name for sbert: {name}")

    if len(dataset) == 0:
        raise RuntimeError("Loaded dataset has zero samples.")
    return dataset


def _load_model(args, device):
    cfg = SBertConfig(
        input_dim=2,
        output_dim=2,
        hidden_size=args.hidden,
        num_layer=args.layer,
        num_head=args.head,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        num_nbr=args.num_nbr,
        dropout_prob=args.dropout_prob,
        act_fn=args.act_fn,
        view_range=args.view_range,
        view_angle=args.view_angle,
        social_range=args.social_range,
        sip=args.sip,
    )
    model = SBertFTModel(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    return model


def _transform_params(raw_sample: np.ndarray, obs_len: int):
    return transform_to_target(
        trajs=np.array(raw_sample, dtype=np.float32, copy=True),
        obs_len=obs_len,
        traj_dim=raw_sample.shape[2],
    )


def _local_to_world(raw_sample: np.ndarray, pred_local: np.ndarray, obs_len: int):
    _, center, theta = _transform_params(raw_sample, obs_len)
    pred_world = rotate_trajs(
        trajs=np.expand_dims(np.array(pred_local, dtype=np.float32, copy=True), axis=0),
        theta=-theta,
        traj_dim=2,
    )[0] - center[np.newaxis, :]
    return pred_world


def _local_trajs_to_world(raw_sample: np.ndarray, trajs_local: np.ndarray, obs_len: int):
    trajs_local = np.array(trajs_local, dtype=np.float32, copy=True)
    if trajs_local.size == 0:
        return trajs_local.reshape(0, obs_len, 2)
    _, center, theta = _transform_params(raw_sample, obs_len)
    world = rotate_trajs(trajs=trajs_local, theta=-theta, traj_dim=2)
    world[:, :, :2] = world[:, :, :2] - center[np.newaxis, np.newaxis, :]
    return world


def _target_center_heading_world(raw_sample: np.ndarray, obs_len: int):
    center = np.array(raw_sample[0, obs_len - 1, :2], dtype=np.float32)
    if np.any(np.isnan(raw_sample[0, obs_len - 2 : obs_len, :2])):
        theta = 0.0
    else:
        diff = raw_sample[0, obs_len - 1, :2] - raw_sample[0, obs_len - 2, :2]
        theta = float(np.arctan2(diff[1], diff[0]))
    return center, theta


def _select_neighbors_local(local_sample: np.ndarray, args):
    nbr_trajs = np.array(local_sample[1:, : args.obs_len + args.pred_len, :2], dtype=np.float32, copy=True)
    if len(nbr_trajs) == 0:
        return np.empty((0, args.obs_len, 2), dtype=np.float32)

    valid_obs = ~np.all(np.isnan(nbr_trajs[:, : args.obs_len, 0]), axis=1)
    nbr_trajs = nbr_trajs[valid_obs]
    if len(nbr_trajs) == 0:
        return np.empty((0, args.obs_len, 2), dtype=np.float32)

    nbr_dist = np.linalg.norm(nbr_trajs[:, :, :2], axis=2)
    nbr_dist[np.isnan(nbr_dist)] = args.view_range
    nbr_trajs[nbr_dist >= args.view_range] = np.nan

    nbr_curr_pos = nbr_trajs[:, args.obs_len - 1].copy()
    nbr_curr_dist = nbr_dist[:, args.obs_len - 1].copy()
    nbr_angles = np.abs(np.arctan2(nbr_curr_pos[:, 1], nbr_curr_pos[:, 0]))
    nbr_angles[np.isnan(nbr_angles)] = args.view_angle

    in_interest = (nbr_curr_dist <= args.social_range) | (nbr_angles <= args.view_angle / 2.0)
    in_interest &= ~np.isnan(nbr_curr_pos[:, 0])
    nbr_trajs = nbr_trajs[in_interest]
    nbr_curr_dist = nbr_curr_dist[in_interest]
    if len(nbr_trajs) == 0:
        return np.empty((0, args.obs_len, 2), dtype=np.float32)

    order = np.argsort(nbr_curr_dist)
    nbr_trajs = nbr_trajs[order][: args.num_nbr]
    return nbr_trajs[:, : args.obs_len, :2]


def _last_valid_point(traj: np.ndarray):
    valid = ~np.isnan(traj[:, 0])
    if not np.any(valid):
        return None
    return traj[np.where(valid)[0][-1], :2]


def _draw_context(ax, center, heading, view_range, view_angle, social_range):
    wedge = Wedge(
        center=tuple(center),
        r=view_range,
        theta1=np.degrees(heading - view_angle / 2.0),
        theta2=np.degrees(heading + view_angle / 2.0),
        fill=False,
        linestyle=(0, (4, 4)),
        linewidth=1.2,
        edgecolor="dimgray",
        alpha=0.7,
        zorder=1,
    )
    circle = Circle(
        xy=tuple(center),
        radius=social_range,
        fill=False,
        linestyle=(0, (4, 4)),
        linewidth=1.2,
        edgecolor="darkslategray",
        alpha=0.7,
        zorder=1,
    )
    ax.add_patch(wedge)
    ax.add_patch(circle)


def _plot_neighbors(ax, neighbors):
    cmap = plt.get_cmap("tab10")
    for nbr_idx, traj in enumerate(neighbors, start=1):
        color = cmap((nbr_idx - 1) % 10)
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color=color,
            linewidth=1.6,
            linestyle="-",
            marker="o",
            markersize=2.5,
            alpha=0.9,
            label=f"Neighbor {nbr_idx}",
        )
        last_pt = _last_valid_point(traj)
        if last_pt is not None:
            ax.text(last_pt[0], last_pt[1], f"N{nbr_idx}", color=color, fontsize=7, weight="bold")


def _collect_samples(args):
    dataset = _load_dataset(args)
    random.seed(args.seed)
    n = min(args.num_samples, len(dataset))
    sampled_idx = random.sample(list(range(len(dataset))), n)

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        if not ckpt.is_file():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        model = _load_model(args, device)
    else:
        model = None
        device = torch.device("cpu")

    samples = []
    with torch.no_grad():
        for idx in sampled_idx:
            item = dataset[idx]
            raw_sample = np.array(dataset.all_trajs[idx], dtype=np.float32, copy=True)
            raw_tgt = raw_sample[0].copy()
            obs_world = raw_tgt[: args.obs_len].copy()
            gt_world = raw_tgt[args.obs_len : args.obs_len + args.pred_len].copy()

            local_sample, _, _ = _transform_params(raw_sample, args.obs_len)
            neighbors_local = _select_neighbors_local(local_sample, args)
            neighbors_world = _local_trajs_to_world(raw_sample, neighbors_local, args.obs_len)
            context_center, context_heading = _target_center_heading_world(raw_sample, args.obs_len)

            if model is not None:
                batched = {k: v.unsqueeze(0).to(device) for k, v in item.items()}
                outputs = model(
                    train=False,
                    spatial_ids=batched["spatial_ids"],
                    temporal_ids=batched["temporal_ids"],
                    segment_ids=batched["segment_ids"],
                    attn_mask=batched["attn_mask"],
                )
                scale = float(item["scales"].item())
                pred_local = outputs["pred_traj"][0].detach().cpu().numpy() * scale
                pred_world = _local_to_world(raw_sample, pred_local, args.obs_len)
            else:
                pred_world = None

            scene_id = dataset.all_scenes[idx] if idx < len(dataset.all_scenes) else "unknown_scene"
            samples.append(
                {
                    "obs": obs_world,
                    "gt": gt_world,
                    "pred": pred_world,
                    "neighbors": neighbors_world,
                    "context_center": context_center,
                    "context_heading": context_heading,
                    "view_range": args.view_range,
                    "view_angle": args.view_angle,
                    "social_range": args.social_range,
                    "title": f"sbert | {args.dataset_name} | scene={scene_id} | idx={idx}",
                }
            )
    return samples


def plot_samples(samples, out_path: Path):
    n = len(samples)
    cols = min(3, max(1, n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.8 * rows))
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

        _draw_context(
            ax,
            center=sample["context_center"],
            heading=sample["context_heading"],
            view_range=sample["view_range"],
            view_angle=sample["view_angle"],
            social_range=sample["social_range"],
        )
        _plot_neighbors(ax, sample["neighbors"])

        ax.plot(obs[:, 0], obs[:, 1], color="royalblue", linewidth=2.0, marker="o", markersize=3, label="Obs")
        ax.plot(gt[:, 0], gt[:, 1], color="crimson", linewidth=2.0, marker="o", markersize=3, label="GT Future")
        if pred is not None:
            ax.plot(pred[:, 0], pred[:, 1], color="black", linewidth=2.0, marker="o", markersize=3, label="Prediction")

        ax.scatter([obs[0, 0]], [obs[0, 1]], color="royalblue", s=35)
        ax.scatter([obs[-1, 0]], [obs[-1, 1]], color="gold", edgecolors="black", linewidths=0.7, s=46, zorder=6, label="Target @ t_obs")
        ax.scatter([gt[-1, 0]], [gt[-1, 1]], color="crimson", s=35)
        if pred is not None:
            ax.scatter([pred[-1, 0]], [pred[-1, 1]], color="black", s=35)

        ax.set_title(sample["title"], fontsize=9)
        ax.set_aspect("equal", adjustable="box")
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
    fig.suptitle("Sampled Trajectories (SBert)", y=0.995)
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
        out_path = base / f"sampled_{args.dataset_name}_sbert.png"

    samples = _collect_samples(args)
    plot_samples(samples, out_path)


if __name__ == "__main__":
    main()

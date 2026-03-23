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
            scale = float(item["scales"].item())

            if model is not None:
                batched = {k: v.unsqueeze(0).to(device) for k, v in item.items()}
                outputs = model(
                    train=False,
                    spatial_ids=batched["spatial_ids"],
                    temporal_ids=batched["temporal_ids"],
                    segment_ids=batched["segment_ids"],
                    attn_mask=batched["attn_mask"],
                )
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
                    "title": f"sbert | {args.dataset_name} | scene={scene_id} | idx={idx}",
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

        ax.plot(obs[:, 0], obs[:, 1], color="royalblue", linewidth=2.0, marker="o", markersize=3, label="Obs")
        ax.plot(gt[:, 0], gt[:, 1], color="crimson", linewidth=2.0, marker="o", markersize=3, label="GT Future")
        if pred is not None:
            ax.plot(pred[:, 0], pred[:, 1], color="black", linewidth=2.0, marker="o", markersize=3, label="Prediction")

        ax.scatter([obs[0, 0]], [obs[0, 1]], color="royalblue", s=35)
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

"""MoAI/Gazebo social-nav dataset adapter for colleague SPU-BERT finetuning.

    The canonical repository location is:

    SPU-BERT/spubert/datasets/moai_social_nav_extended_goal.py

It reads the social-nav PKLs created in this project and returns the same
finetune keys used by SBertPlusFTModel:

    mgp_* token stream, tgp_* token stream, traj_lbl, goal_lbl, guidance_lbl

If args.scene=True, this adapter also converts each sample's local_map into the
env_* keys expected by the colleague scene encoder.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class MoAISocialNavExtendedGoalDataset(Dataset):
    """Dataset adapter from this project's PKL format to SPU-BERT tokens.

    Expected input PKL:
        {
          "samples": [
            {
              "target_past": (8, 2),
              "target_future": (12, 2),
              "neighbor_past": (N, 8, 2),
              "neighbor_future": (N, 12, 2), optional but recommended,
              "neighbor_mask": (N,),
              "guidance_point": (2,),
              "local_map": (32, 32), optional unless scene=True
            },
            ...
          ],
          "metadata": {...}
        }

    Returned finetune keys match colleague SBertPlusFTModel.forward().
    """

    def __init__(self, split: str, args: Any) -> None:
        super().__init__()
        self.split = split
        self.args = args
        self.obs_len = int(getattr(args, "obs_len", 8))
        self.pred_len = int(getattr(args, "pred_len", 12))
        self.seq_len = self.obs_len + self.pred_len
        self.num_nbr = int(getattr(args, "num_nbr", 4))
        self.view_range = float(getattr(args, "view_range", 20.0))
        self.view_angle = float(getattr(args, "view_angle", math.pi / 3.0))
        self.social_range = float(getattr(args, "social_range", 2.0))
        self.guidance_conditioned = bool(getattr(args, "guidance_conditioned", False))
        if not self.guidance_conditioned:
            raise ValueError(
                "MoAI Gazebo data requires model.heads.spubert.guidance_conditioned=true"
            )
        self.scene = bool(getattr(args, "scene", False))
        self.patch_size = int(getattr(args, "patch_size", 16))
        self.map_align_to_target = bool(getattr(args, "map_align_to_target", True))
        self.map_source_unknown_value = float(
            getattr(args, "map_source_unknown_value", getattr(args, "map_unknown_value", 0.5))
        )

        self.seq_input_len = (self.obs_len + 1) * (self.num_nbr + 1) + self.pred_len + 1
        self.spatial = _SpatialTokens(self.view_range)
        self.payload_path = resolve_split_path(args, split)
        with self.payload_path.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict) or "samples" not in payload:
            raise ValueError(f"expected a social-nav payload with samples: {self.payload_path}")
        self.samples = list(payload["samples"])
        self.metadata = dict(payload.get("metadata", {}))
        self.local_map_size_m = float(
            getattr(args, "local_map_size_m", self.metadata.get("local_map_size_m", self.metadata.get("crop_size_m", 8.0)))
        )
        self.local_map_grid_size = int(
            getattr(args, "local_map_grid_size", self.metadata.get("local_map_grid_size", self.metadata.get("map_size", 32)))
        )
        self.env_resol = float(getattr(args, "env_resol", self.local_map_size_m / max(self.local_map_grid_size, 1)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        raw_trajs = build_all_trajs(sample, self.obs_len, self.pred_len)
        trajs, center, theta = transform_to_target(raw_trajs, self.obs_len)
        trajs = neighbor_filtering(
            trajs,
            num_nbr=self.num_nbr,
            obs_len=self.obs_len,
            pred_len=self.pred_len,
            view_range=self.view_range,
            view_angle=self.view_angle,
            social_range=self.social_range,
        )

        traj_lbl = trajs[0, self.obs_len : self.seq_len].astype(np.float32)
        goal_lbl = traj_lbl[-1].copy()
        human_future_lbl, human_future_mask = build_human_future_labels(
            trajs,
            obs_len=self.obs_len,
            pred_len=self.pred_len,
            num_nbr=self.num_nbr,
        )
        guidance_world = xy(sample, "guidance_point", "final_goal").reshape(-1, 2)[0]
        guidance_lbl = transform_point(guidance_world, center, theta).astype(np.float32)

        streams = build_mgp_tgp_streams(
            trajs=trajs,
            goal_lbl=goal_lbl,
            guidance_lbl=guidance_lbl,
            obs_len=self.obs_len,
            pred_len=self.pred_len,
            num_nbr=self.num_nbr,
            seq_input_len=self.seq_input_len,
            spatial=self.spatial,
        )
        out = {
            key: torch.tensor(value, dtype=dtype)
            for key, value, dtype in [
                ("mgp_spatial_ids", streams["mgp_spatial_ids"], torch.float),
                ("mgp_segment_ids", streams["mgp_segment_ids"], torch.long),
                ("mgp_temporal_ids", streams["mgp_temporal_ids"], torch.long),
                ("mgp_attn_mask", streams["mgp_attn_mask"], torch.float),
                ("tgp_spatial_ids", streams["tgp_spatial_ids"], torch.float),
                ("tgp_segment_ids", streams["tgp_segment_ids"], torch.long),
                ("tgp_temporal_ids", streams["tgp_temporal_ids"], torch.long),
                ("tgp_attn_mask", streams["tgp_attn_mask"], torch.float),
                ("traj_lbl", traj_lbl, torch.float),
                ("goal_lbl", goal_lbl, torch.float),
                ("guidance_lbl", guidance_lbl, torch.float),
                ("human_future_lbl", human_future_lbl, torch.float),
                ("human_future_mask", human_future_mask, torch.float),
            ]
        }
        if self.scene:
            map_streams = build_local_map_streams(
                sample=sample,
                patch_size=self.patch_size,
                num_nbr=self.num_nbr,
                obs_len=self.obs_len,
                local_map_size_m=self.local_map_size_m,
                env_resol=self.env_resol,
                theta=theta,
                align_to_target=self.map_align_to_target,
                source_unknown_value=self.map_source_unknown_value,
            )
            for key, value, dtype in [
                ("env_spatial_ids", map_streams["env_spatial_ids"], torch.float),
                ("env_segment_ids", map_streams["env_segment_ids"], torch.long),
                ("env_temporal_ids", map_streams["env_temporal_ids"], torch.long),
                ("env_attn_mask", map_streams["env_attn_mask"], torch.float),
                ("envs", map_streams["envs"], torch.float),
                ("envs_params", map_streams["envs_params"], torch.float),
            ]:
                out[key] = torch.tensor(value, dtype=dtype)
        out["scales"] = torch.tensor(1.0, dtype=torch.float)
        return out


class _SpatialTokens:
    def __init__(self, view_range: float) -> None:
        self.pad_id = [-view_range, -view_range]
        self.msk_id = [view_range, view_range]
        self.sep_id = [view_range, -view_range]
        self.sot_id = [-view_range, view_range]
        self.true_val = 1
        self.false_val = 0


def resolve_split_path(args: Any, split: str) -> Path:
    dataset_path = Path(str(getattr(args, "dataset_path", ""))).expanduser()
    dataset_name = str(getattr(args, "dataset_name", ""))
    dataset_split = str(getattr(args, "dataset_split", ""))
    candidates = []
    if dataset_path.is_file():
        candidates.append(dataset_path)
    if dataset_split:
        candidates.extend(
            [
                dataset_path / f"{dataset_split}_{split}.pkl",
                dataset_path / "splits" / f"{dataset_split}_{split}.pkl",
            ]
        )
    if dataset_name:
        candidates.extend(
            [
                dataset_path / dataset_name / f"{dataset_split}_{split}.pkl",
                dataset_path / dataset_name / "splits" / f"{dataset_split}_{split}.pkl",
            ]
        )
    existing = [path for path in candidates if path.exists()]
    if existing:
        return existing[0]
    tried = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"could not resolve MoAI social-nav split for split={split}. Tried:\n  {tried}")


def xy(sample: dict[str, Any], *keys: str) -> np.ndarray:
    for key in keys:
        if key in sample:
            return np.asarray(sample[key], dtype=np.float32)[..., :2]
    raise KeyError(f"sample has none of these keys: {keys}")


def build_all_trajs(sample: dict[str, Any], obs_len: int, pred_len: int) -> np.ndarray:
    target_past = xy(sample, "target_past", "robot_past")
    target_future = xy(sample, "target_future", "robot_future")
    target = np.concatenate([target_past[:obs_len], target_future[:pred_len]], axis=0)

    neighbor_past = xy(sample, "neighbor_past")
    neighbor_mask = np.asarray(sample.get("neighbor_mask", np.ones(neighbor_past.shape[0])), dtype=np.float32)
    if "neighbor_future" in sample:
        neighbor_future = xy(sample, "neighbor_future")
    else:
        neighbor_future = np.full((neighbor_past.shape[0], pred_len, 2), np.nan, dtype=np.float32)

    rows = [target.astype(np.float32)]
    for idx in range(neighbor_past.shape[0]):
        if idx >= len(neighbor_mask) or float(neighbor_mask[idx]) <= 0.5:
            continue
        future = neighbor_future[idx, :pred_len] if idx < neighbor_future.shape[0] else np.full((pred_len, 2), np.nan)
        rows.append(np.concatenate([neighbor_past[idx, :obs_len], future], axis=0).astype(np.float32))
    return np.stack(rows, axis=0)


def build_human_future_labels(
    trajs: np.ndarray,
    obs_len: int,
    pred_len: int,
    num_nbr: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad filtered human futures and mark only finite synchronized steps."""
    labels = np.zeros((num_nbr, pred_len, 2), dtype=np.float32)
    mask = np.zeros((num_nbr, pred_len), dtype=np.float32)
    for nbr_idx, nbr_traj in enumerate(trajs[1 : num_nbr + 1]):
        future = np.asarray(
            nbr_traj[obs_len : obs_len + pred_len, :2],
            dtype=np.float32,
        )
        valid = np.isfinite(future).all(axis=-1)
        labels[nbr_idx, valid] = future[valid]
        mask[nbr_idx, valid] = 1.0
    return labels, mask


def rotate_xy(xy_value: np.ndarray, theta: float) -> np.ndarray:
    ct = math.cos(theta)
    st = math.sin(theta)
    rot = np.array([[ct, st], [-st, ct]], dtype=np.float32)
    return xy_value.dot(rot.T)


def transform_to_target(trajs: np.ndarray, obs_len: int) -> tuple[np.ndarray, np.ndarray, float]:
    center = -trajs[0, obs_len - 1, :2]
    shifted = trajs.copy()
    shifted[..., :2] = shifted[..., :2] + center
    if np.any(np.isnan(shifted[0, obs_len - 2 : obs_len, 0])):
        theta = 0.0
    else:
        diff = shifted[0, obs_len - 1, :2] - shifted[0, obs_len - 2, :2]
        theta = float(math.atan2(float(diff[1]), float(diff[0])))
    transformed = shifted.copy()
    transformed[..., :2] = rotate_xy(shifted[..., :2].reshape(-1, 2), theta).reshape(shifted.shape)
    return transformed.astype(np.float32), center.astype(np.float32), theta


def transform_point(point: np.ndarray, center: np.ndarray, theta: float) -> np.ndarray:
    return rotate_xy((np.asarray(point, dtype=np.float32)[:2] + center).reshape(1, 2), theta)[0]


def neighbor_filtering(
    trajs: np.ndarray,
    num_nbr: int,
    obs_len: int,
    pred_len: int,
    view_range: float,
    view_angle: float,
    social_range: float,
) -> np.ndarray:
    nbr_trajs = trajs[1:, : obs_len + pred_len].copy()
    nbr_trajs = nbr_trajs[~np.all(np.isnan(nbr_trajs[:, :obs_len, 0]), axis=1)]
    if len(nbr_trajs) == 0:
        return np.expand_dims(trajs[0], axis=0)

    nbr_dist = np.sqrt(np.sum(np.square(nbr_trajs[:, :, :2]), axis=2))
    nbr_dist[np.isnan(nbr_dist)] = view_range
    nbr_trajs[nbr_dist >= view_range, :] = np.nan

    nbr_curr_pos = nbr_trajs[:, obs_len - 1].copy()
    nbr_curr_dist = nbr_dist[:, obs_len - 1].copy()
    out_social = nbr_curr_dist > social_range
    angle = np.abs(np.arctan2(nbr_curr_pos[:, 1], nbr_curr_pos[:, 0]))
    angle[np.isnan(angle)] = view_angle
    out_sight = angle > view_angle / 2.0
    nbr_curr_pos[out_social & out_sight, :] = np.nan
    valid = np.array([not np.isnan(pos[0]) for pos in nbr_curr_pos])

    nbr_trajs = nbr_trajs[valid]
    nbr_curr_dist = nbr_curr_dist[valid]
    if len(nbr_trajs) > 0:
        nbr_trajs = nbr_trajs[np.argsort(nbr_curr_dist)][:num_nbr]

    out = np.full((len(nbr_trajs) + 1, obs_len + pred_len, 2), np.nan, dtype=np.float32)
    out[0] = trajs[0]
    out[1:] = nbr_trajs
    return out[~np.all(np.isnan(out[:, :obs_len, 0]), axis=1)]


def build_mgp_tgp_streams(
    trajs: np.ndarray,
    goal_lbl: np.ndarray,
    guidance_lbl: np.ndarray,
    obs_len: int,
    pred_len: int,
    num_nbr: int,
    seq_input_len: int,
    spatial: _SpatialTokens,
) -> dict[str, np.ndarray]:
    target_obs = trajs[0, :obs_len].astype(np.float32)

    # Guidance-conditioned MGP:
    # [SOT] + obs(8) + PAD(11) + MASK(t+12 goal) + guidance + neighbors.
    mgp_spatial = (
        [spatial.sot_id]
        + target_obs.tolist()
        + [spatial.pad_id] * (pred_len - 1)
        + [spatial.msk_id]
        + [guidance_lbl.tolist()]
    )
    mgp_segment = [1] + [1] * obs_len + [0] * (pred_len - 1) + [1, 1]
    mgp_temporal = (
        [0]
        + list(range(1, obs_len + 1))
        + [0] * (pred_len - 1)
        + [obs_len + pred_len, obs_len + pred_len + 1]
    )
    mgp_attn = [1] + [1] * obs_len + [0] * (pred_len - 1) + [1, 1]

    tgp_spatial = [spatial.sot_id] + target_obs.tolist() + [spatial.msk_id] * pred_len + [goal_lbl.tolist()]
    tgp_segment = [1] + [1] * obs_len + [1] * pred_len + [1]
    tgp_temporal = [0] + list(range(1, obs_len + 1)) + list(range(obs_len + 1, obs_len + pred_len + 1)) + [obs_len + pred_len + 1]
    tgp_attn = [1] + [1] * obs_len + [1] * pred_len + [1]

    nbr_spatial: list[list[float]] = []
    nbr_segment: list[int] = []
    nbr_temporal: list[int] = []
    nbr_attn: list[int] = []
    for nbr_idx, nbr_traj in enumerate(trajs[1:]):
        obs = nbr_traj[:obs_len].copy()
        nbr_nan = np.isnan(obs[:, 0])
        obs[nbr_nan] = np.asarray(spatial.pad_id, dtype=np.float32)
        seg_id = nbr_idx + 2
        nbr_spatial += [spatial.sep_id] + obs.tolist()
        nbr_segment += [seg_id] + [0 if nan else seg_id for nan in nbr_nan]
        nbr_temporal += [0] + [0 if nan else t for t, nan in zip(range(1, obs_len + 1), nbr_nan)]
        nbr_attn += [1] + [0 if nan else 1 for nan in nbr_nan]

    ext_len = seq_input_len - len(mgp_spatial) - len(nbr_spatial)
    if ext_len < 0:
        raise ValueError(
            f"SPU-BERT token overflow: seq_input_len={seq_input_len}, "
            f"target_tokens={len(mgp_spatial)}, neighbor_tokens={len(nbr_spatial)}. "
            f"Reduce num_nbr or increase seq_input_len contract."
        )
    nbr_spatial += [spatial.pad_id] * ext_len
    nbr_segment += [0] * ext_len
    nbr_temporal += [0] * ext_len
    nbr_attn += [0] * ext_len

    return {
        "mgp_spatial_ids": np.asarray(mgp_spatial + nbr_spatial, dtype=np.float32),
        "mgp_segment_ids": np.asarray(mgp_segment + nbr_segment, dtype=np.int64),
        "mgp_temporal_ids": np.asarray(mgp_temporal + nbr_temporal, dtype=np.int64),
        "mgp_attn_mask": np.asarray(mgp_attn + nbr_attn, dtype=np.float32),
        "tgp_spatial_ids": np.asarray(tgp_spatial + nbr_spatial, dtype=np.float32),
        "tgp_segment_ids": np.asarray(tgp_segment + nbr_segment, dtype=np.int64),
        "tgp_temporal_ids": np.asarray(tgp_temporal + nbr_temporal, dtype=np.int64),
        "tgp_attn_mask": np.asarray(tgp_attn + nbr_attn, dtype=np.float32),
    }


def align_local_map_to_target(
    local_map: np.ndarray,
    theta: float,
    unknown_value: float = 0.5,
) -> np.ndarray:
    """Rotate a robot-centered world-axis map into the target-heading frame.

    Trajectories use ``p_target = R(-theta) p_world``.  For every output map
    cell in the target frame, this function samples the corresponding source
    position ``p_world = R(theta) p_target``.  Map row 0 is treated as +y,
    matching the local-map convention used by the training/evaluation code.
    """
    grid = np.asarray(local_map, dtype=np.float32)
    if grid.ndim != 2:
        raise ValueError(f"local_map must be 2D, got shape={grid.shape}")
    if not np.isfinite(theta):
        raise ValueError(f"map rotation theta must be finite, got {theta}")
    if abs(float(theta)) <= 1e-8:
        return grid.copy()

    height, width = grid.shape
    center_x = 0.5 * (width - 1)
    center_y = 0.5 * (height - 1)
    rows, cols = np.indices((height, width), dtype=np.float32)

    target_x = cols - center_x
    target_y = center_y - rows
    cos_theta = math.cos(float(theta))
    sin_theta = math.sin(float(theta))
    world_x = cos_theta * target_x - sin_theta * target_y
    world_y = sin_theta * target_x + cos_theta * target_y

    source_cols = np.rint(center_x + world_x).astype(np.int64)
    source_rows = np.rint(center_y - world_y).astype(np.int64)
    valid = (
        (source_rows >= 0)
        & (source_rows < height)
        & (source_cols >= 0)
        & (source_cols < width)
    )

    aligned = np.full((height, width), float(unknown_value), dtype=np.float32)
    aligned[valid] = grid[source_rows[valid], source_cols[valid]]
    return aligned


def encode_local_map_for_colleague(
    local_map: np.ndarray,
    source_unknown_value: float = 0.5,
    occupied_threshold: float = 0.55,
) -> np.ndarray:
    """Convert our occupancy encoding to the colleague scene-map encoding.

    Source Gazebo maps use 0=free and 1=occupied.  Rotation/padding cells use
    ``source_unknown_value``.  Colleague maps use 0=unknown/padding, 1=free,
    and 2=occupied.
    """
    grid = np.asarray(local_map, dtype=np.float32)
    if grid.ndim != 2:
        raise ValueError(f"local_map must be 2D, got shape={grid.shape}")
    encoded = np.zeros(grid.shape, dtype=np.float32)
    unknown = np.isclose(grid, float(source_unknown_value), atol=1e-6)
    occupied = (grid > float(occupied_threshold)) & ~unknown
    free = ~unknown & ~occupied
    encoded[free] = 1.0
    encoded[occupied] = 2.0
    return encoded


def build_local_map_streams(
    sample: dict[str, Any],
    patch_size: int,
    num_nbr: int,
    obs_len: int,
    local_map_size_m: float,
    env_resol: float,
    theta: float = 0.0,
    align_to_target: bool = True,
    source_unknown_value: float = 0.5,
) -> dict[str, np.ndarray]:
    if "local_map" not in sample:
        raise KeyError("scene=True requires each sample to contain local_map")

    source_grid = np.asarray(sample["local_map"], dtype=np.float32)
    if source_grid.ndim != 2:
        raise ValueError(f"local_map must be 2D, got shape={source_grid.shape}")
    if source_grid.shape[0] != source_grid.shape[1]:
        raise ValueError(f"local_map must be square, got shape={source_grid.shape}")
    if not np.isfinite(source_grid).all():
        raise ValueError("local_map contains NaN or Inf")

    source_grid_size = int(source_grid.shape[0])
    inferred_resolution = float(local_map_size_m) / max(source_grid_size, 1)
    if env_resol > 0 and not math.isclose(float(env_resol), inferred_resolution, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError(
            "local-map resolution mismatch: "
            f"local_map_size_m/grid_size={inferred_resolution:.6f}, env_resol={float(env_resol):.6f}"
        )

    grid = (
        align_local_map_to_target(source_grid, theta=theta, unknown_value=source_unknown_value)
        if align_to_target
        else source_grid.copy()
    )
    grid = encode_local_map_for_colleague(grid, source_unknown_value=source_unknown_value)
    # Gazebo local maps use row 0 = +y. Yunju's grid lookup uses row 0 = min_y.
    # Flip once here so scene tokens and collision/goal lookup share one convention.
    grid = np.flipud(grid).copy()

    patch_size = max(1, int(patch_size))
    pad_h = (patch_size - grid.shape[0] % patch_size) % patch_size
    pad_w = (patch_size - grid.shape[1] % patch_size) % patch_size
    if pad_h or pad_w:
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        grid = np.pad(
            grid,
            ((top, bottom), (left, right)),
            mode="constant",
            constant_values=0.0,
        )

    patches = []
    patch_masks = []
    for row in range(0, grid.shape[0], patch_size):
        for col in range(0, grid.shape[1], patch_size):
            patch = grid[row : row + patch_size, col : col + patch_size]
            patches.append(patch.reshape(-1))
            known = patch > 0.0
            patch_masks.append(float(np.any(known)))
    env_spatial_ids = np.asarray(patches, dtype=np.float32)
    env_segment_ids = np.arange(
        start=num_nbr + 2,
        stop=num_nbr + len(env_spatial_ids) + 2,
        dtype=np.int64,
    )
    env_temporal_ids = np.ones(len(env_spatial_ids), dtype=np.int64) * int(obs_len)
    env_attn_mask = np.asarray(patch_masks, dtype=np.float32)

    resolution = inferred_resolution
    padded_width_m = float(grid.shape[1]) * resolution
    padded_height_m = float(grid.shape[0]) * resolution
    envs_params = np.asarray(
        [
            -0.5 * padded_width_m,
            -0.5 * padded_height_m,
            float(grid.shape[1]),
            float(grid.shape[0]),
            resolution,
            2.0,
        ],
        dtype=np.float32,
    )
    return {
        "env_spatial_ids": env_spatial_ids,
        "env_segment_ids": env_segment_ids,
        "env_temporal_ids": env_temporal_ids,
        "env_attn_mask": env_attn_mask,
        "envs": grid.astype(np.float32),
        "envs_params": envs_params,
    }

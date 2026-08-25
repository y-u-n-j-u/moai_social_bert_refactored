from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ADELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def forward(self, input, target):
        return torch.sqrt(torch.sum(((input - target) ** 2.0) + self.eps, dim=-1)).mean()


class FDELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def forward(self, input, target):
        if input.dim() == 3:
            input = input[:, -1]
        if target.dim() == 3:
            target = target[:, -1]
        return torch.sqrt(torch.sum(((input - target) ** 2.0) + self.eps, dim=-1)).mean()


class MaskedADELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def forward(self, input, target, mask, weight=None):
        l2_sum = torch.sqrt(torch.sum(((input - target) ** 2.0) * mask + self.eps, dim=2)).sum()
        num_mask = torch.sum(mask) / 2
        return l2_sum / num_mask


class MGPCVAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def forward(self, pred_goals, gt_goals, k_sample, output_dim=2, best=True):
        gt_goals = gt_goals.unsqueeze(1).repeat(1, k_sample, 1)
        goal_rmse = torch.sqrt(
            torch.sum(((pred_goals[:, :, :output_dim] - gt_goals[:, :, :output_dim]) ** 2) + self.eps, dim=-1)
        )
        best_idx = torch.argmin(goal_rmse, dim=-1)
        goal_loss = goal_rmse[range(len(best_idx)), best_idx].mean() if best else goal_rmse.mean()
        return goal_loss, best_idx


def cal_idx_from_pos(pos, min_pos, max_idx, res):
    pos = torch.floor((pos - min_pos) / res).to(dtype=torch.long, device=pos.device)
    valid = ((0 <= pos) & (pos < max_idx)).to(pos.device)
    return pos, valid


def _differentiable_collision_loss(pred_positions, envs, envs_params):
    """Sample map risk continuously so collision loss reaches predictions."""
    if pred_positions.ndim < 3 or pred_positions.shape[-1] < 2:
        raise ValueError(
            f"pred_positions must have shape [B, ..., 2], got {pred_positions.shape}"
        )
    batch_size = pred_positions.shape[0]
    points = pred_positions[..., :2].reshape(batch_size, -1, 2)
    maps = envs.to(device=points.device, dtype=points.dtype)
    params = envs_params.to(device=points.device, dtype=points.dtype)
    if maps.ndim != 3 or maps.shape[0] != batch_size:
        raise ValueError(f"envs must have shape [B, H, W], got {maps.shape}")

    min_x = params[:, 0:1]
    min_y = params[:, 1:2]
    width_m = params[:, 2:3] * params[:, 4:5]
    height_m = params[:, 3:4] * params[:, 4:5]
    x_normalized = 2.0 * (points[..., 0] - min_x) / width_m.clamp_min(1e-6) - 1.0
    y_normalized = 2.0 * (points[..., 1] - min_y) / height_m.clamp_min(1e-6) - 1.0
    sample_grid = torch.stack((x_normalized, y_normalized), dim=-1).unsqueeze(2)

    # Dataset maps use 0=unknown, 1=free, and >=threshold=occupied. Convert
    # both unknown and occupied space to risk 1 while preserving continuous
    # interpolation across cell boundaries.
    occupied_threshold = params[:, 5].reshape(batch_size, 1, 1)
    unknown_risk = torch.relu(1.0 - maps)
    occupied_risk = torch.relu(maps - 1.0) / (occupied_threshold - 1.0).clamp_min(1e-6)
    risk_maps = torch.maximum(unknown_risk, occupied_risk).clamp(0.0, 1.0)
    sampled_risk = F.grid_sample(
        risk_maps.unsqueeze(1),
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).squeeze(1).squeeze(-1)
    outside = (
        (x_normalized < -1.0)
        | (x_normalized >= 1.0)
        | (y_normalized < -1.0)
        | (y_normalized >= 1.0)
    ).to(sampled_risk.dtype)
    outside_distance = (
        F.relu(-1.0 - x_normalized)
        + F.relu(x_normalized - 1.0)
        + F.relu(-1.0 - y_normalized)
        + F.relu(y_normalized - 1.0)
    )
    # Keep the legacy unit penalty for any out-of-map point, then increase it
    # with normalized distance so gradient descent has a direction back in-map.
    outside_risk = outside + outside_distance
    return torch.maximum(sampled_risk, outside_risk).mean()


def goal_collision_loss(pred_goals, envs, envs_params):
    return _differentiable_collision_loss(pred_goals, envs, envs_params)


def pos_collision_loss(pred_trajs, envs, envs_params):
    return _differentiable_collision_loss(pred_trajs, envs, envs_params)


def dynamic_social_collision_loss(
    pred_trajs,
    human_futures,
    human_future_mask,
    safe_distance,
):
    """Penalize synchronized robot-human future positions inside a safety margin.

    ``pred_trajs`` and ``human_futures`` must already use the same local frame
    and timestep convention. Missing human future steps are ignored by the
    mask, so legacy or partially observed tracks do not create false losses.
    """
    if pred_trajs.ndim != 3 or pred_trajs.shape[-1] < 2:
        raise ValueError(
            f"pred_trajs must have shape [B,T,2], got {pred_trajs.shape}"
        )
    if human_futures.ndim != 4 or human_futures.shape[-1] < 2:
        raise ValueError(
            "human_futures must have shape [B,N,T,2], "
            f"got {human_futures.shape}"
        )
    if human_future_mask.ndim != 3:
        raise ValueError(
            "human_future_mask must have shape [B,N,T], "
            f"got {human_future_mask.shape}"
        )
    if pred_trajs.shape[0] != human_futures.shape[0]:
        raise ValueError("robot and human future batch sizes must match")
    if pred_trajs.shape[1] != human_futures.shape[2]:
        raise ValueError("robot and human future lengths must match")
    if human_future_mask.shape != human_futures.shape[:3]:
        raise ValueError("human_future_mask shape must match [B,N,T]")
    if safe_distance <= 0:
        raise ValueError("safe_distance must be positive")

    humans = human_futures[..., :2].to(
        device=pred_trajs.device,
        dtype=pred_trajs.dtype,
    )
    finite = torch.isfinite(humans).all(dim=-1)
    valid = human_future_mask.to(device=pred_trajs.device) > 0
    valid = valid & finite
    humans = torch.nan_to_num(humans)

    distances = torch.linalg.vector_norm(
        pred_trajs[:, None, :, :2] - humans,
        dim=-1,
    )
    intrusion = F.relu(float(safe_distance) - distances).square()
    # A single dangerous synchronized step matters. Averaging over all 12
    # steps would dilute a brief collision with eleven otherwise-safe steps.
    pair_penalty = intrusion.masked_fill(~valid, 0.0).max(dim=-1).values
    valid_pairs = valid.any(dim=-1).to(dtype=pred_trajs.dtype)
    return (pair_penalty * valid_pairs).sum() / valid_pairs.sum().clamp_min(1.0)


def bom_loss_3(pred_goals, pred_trajs, gt_goals, gt_trajs, k_sample, output_dim=2):
    gt_trajs = gt_trajs.unsqueeze(1).repeat(1, k_sample, 1, 1)
    gt_goals = gt_goals.unsqueeze(1).repeat(1, k_sample, 1)
    goal_rmse = torch.sqrt(torch.sum((pred_goals[:, :, :output_dim] - gt_goals[:, :, :output_dim]) ** 2, dim=-1))
    traj_rmse = torch.sqrt(torch.sum((pred_trajs - gt_trajs) ** 2, dim=-1))
    traj_fde_rmse = traj_rmse[:, :, -1]
    traj_ade_rmse = traj_rmse.mean(dim=-1)

    best_gde = torch.min(goal_rmse, dim=-1)[0].sum()
    best_fde = torch.min(traj_fde_rmse, dim=-1)[0].sum()
    best_ade = torch.min(traj_ade_rmse, dim=-1)[0].sum()
    return best_gde, best_ade, best_fde


def ADError(pred_traj, gt_traj):
    loss = gt_traj - pred_traj
    loss = loss ** 2
    loss = torch.sqrt(loss.sum(dim=2)).mean(dim=1)
    return torch.sum(loss)


def FDError(pred_final_pos, gt_final_pos):
    loss = gt_final_pos - pred_final_pos
    loss = loss ** 2
    loss = torch.sqrt(loss.sum(dim=1))
    return torch.sum(loss)

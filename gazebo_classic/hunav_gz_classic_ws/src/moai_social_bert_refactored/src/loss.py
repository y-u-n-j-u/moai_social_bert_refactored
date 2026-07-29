from __future__ import annotations

import torch
from torch import nn


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


def goal_collision_loss(pred_goals, envs, envs_params):
    num_goal = 0
    num_col_goal = torch.zeros((), dtype=torch.double, device=pred_goals.device)
    for bidx, bgoal in enumerate(pred_goals):
        num_goal += bgoal.size(dim=0)
        bgoal = bgoal.reshape(-1, 2)
        x_ids, x_valid = cal_idx_from_pos(bgoal[:, 0], envs_params[bidx][0], envs_params[bidx][2], envs_params[bidx][4])
        y_ids, y_valid = cal_idx_from_pos(bgoal[:, 1], envs_params[bidx][1], envs_params[bidx][3], envs_params[bidx][4])
        valid = (x_valid & y_valid).to(pred_goals.device)
        num_col_goal += torch.sum(~valid).double()
        vals = envs[bidx][y_ids[valid], x_ids[valid]]
        unsafe = (vals <= 0) | (vals >= envs_params[bidx][5])
        num_col_goal += torch.sum(unsafe).double()
    return torch.div(num_col_goal, max(num_goal, 1))


def pos_collision_loss(pred_trajs, envs, envs_params):
    num_pos = 0
    num_col_pos = torch.zeros((), dtype=torch.double, device=pred_trajs.device)
    for bidx, btraj in enumerate(pred_trajs):
        btraj = btraj.reshape(-1, 2)
        num_pos += btraj.size(dim=0)
        x_ids, x_valid = cal_idx_from_pos(btraj[:, 0], envs_params[bidx][0], envs_params[bidx][2], envs_params[bidx][4])
        y_ids, y_valid = cal_idx_from_pos(btraj[:, 1], envs_params[bidx][1], envs_params[bidx][3], envs_params[bidx][4])
        valid = (x_valid & y_valid).to(pred_trajs.device)
        num_col_pos += torch.sum(~valid).double()
        vals = envs[bidx][y_ids[valid], x_ids[valid]]
        unsafe = (vals <= 0) | (vals >= envs_params[bidx][5])
        num_col_pos += torch.sum(unsafe).double()
    return torch.div(num_col_pos, max(num_pos, 1))


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

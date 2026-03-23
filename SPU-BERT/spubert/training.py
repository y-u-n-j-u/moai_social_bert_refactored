from __future__ import annotations

import torch

from src.loss import bom_loss_3
from src.utils import (
    CyclicScherduler,
    SimpleTrainerBase,
    _data_parallel,
    _make_eval_pbar,
    _make_pbar,
    _postfix,
    _resolve_pretrain_checkpoint,
    _safe_avg,
    _scheduler_steps_per_epoch,
    _to_device,
    build_adamw,
    build_scheduler_from_args,
    estimate_map_length,
    estimate_num_patch,
    frange_cycle_cosine,
    frange_cycle_sigmoid,
)

from .model import (
    SBertPlusFTConfig,
    SBertPlusFTModel,
    SBertPlusMGPConfig,
    SBertPlusPTConfig,
    SBertPlusPTModel,
    SBertPlusTGPConfig,
)



def _num_patch(args):
    return estimate_num_patch(estimate_map_length(args.env_range * 2, args.env_resol), args.patch_size)



def _spubert_shared_cfg_kwargs(args):
    return {
        "input_dim": args.input_dim,
        "output_dim": args.output_dim,
        "goal_dim": args.goal_dim,
        "hidden_size": args.hidden,
        "num_layer": args.layer,
        "num_head": args.head,
        "obs_len": args.obs_len,
        "pred_len": args.pred_len,
        "num_nbr": args.num_nbr,
        "scene": args.scene,
        "num_patch": _num_patch(args),
        "dropout_prob": args.dropout_prob,
        "patch_size": args.patch_size,
        "act_fn": args.act_fn,
        "view_range": args.view_range,
        "view_angle": args.view_angle,
        "social_range": args.social_range,
        "backbone_type": args.backbone_type,
        "scene_encoder_type": args.scene_encoder_type,
        "scene_num_splits": args.scene_num_splits,
        "scene_image_size": args.scene_image_size,
        "scene_patch_size": args.scene_patch_size,
        "scene_num_channels": args.scene_num_channels,
        "scene_hidden_size": args.scene_hidden_size,
        "scene_num_hidden_layers": args.scene_num_hidden_layers,
        "scene_num_attention_heads": args.scene_num_attention_heads,
        "scene_intermediate_size": args.scene_intermediate_size,
        "binary_scene": args.binary_scene,
    }



def _build_spubert_pt_config(args):
    kwargs = _spubert_shared_cfg_kwargs(args)
    kwargs.update(
        {
            "sip": args.sip,
            "col_weight": args.col_weight,
            "traj_weight": args.traj_weight,
        }
    )
    return SBertPlusPTConfig(**kwargs)



def _build_spubert_tgp_config(args):
    kwargs = _spubert_shared_cfg_kwargs(args)
    kwargs.update(
        {
            "col_weight": args.col_weight,
            "traj_weight": args.traj_weight,
        }
    )
    return SBertPlusTGPConfig(**kwargs)



def _build_spubert_mgp_config(args):
    kwargs = _spubert_shared_cfg_kwargs(args)
    kwargs.update(
        {
            "k_sample": args.k_sample,
            "goal_hidden_size": args.goal_hidden,
            "goal_latent_size": args.goal_latent,
            "kld_weight": args.kld_weight,
            "col_weight": args.col_weight,
            "kld_clamp": args.kld_clamp,
            "cvae_sigma": args.cvae_sigma,
            "goal_weight": args.goal_weight,
            "normal": args.normal,
        }
    )
    return SBertPlusMGPConfig(**kwargs)



def _build_train_stack(model, train_dataloader, args, *, eps):
    optim = build_adamw(model.parameters(), lr=args.lr, eps=eps)
    lr_scheduler = build_scheduler_from_args(args, optim, len(train_dataloader))
    return optim, lr_scheduler



def _build_dual_train_stack(model, train_dataloader, args, *, eps):
    mgp_optim = build_adamw(model.mgp_model.parameters(), lr=args.lr, eps=eps)
    tgp_optim = build_adamw(model.tgp_model.parameters(), lr=args.lr, eps=eps)
    steps_per_epoch = len(train_dataloader)
    mgp_lr_scheduler = build_scheduler_from_args(args, mgp_optim, steps_per_epoch)
    tgp_lr_scheduler = build_scheduler_from_args(args, tgp_optim, steps_per_epoch)
    return mgp_optim, tgp_optim, mgp_lr_scheduler, tgp_lr_scheduler



def _scene_batch_kwargs(data, *, include_env_params: bool):
    kwargs = {
        "env_spatial_ids": data["env_spatial_ids"],
        "env_temporal_ids": data["env_temporal_ids"],
        "env_segment_ids": data["env_segment_ids"],
        "env_attn_mask": data["env_attn_mask"],
        "envs": data["envs"],
    }
    if include_env_params:
        kwargs["envs_params"] = data["envs_params"]
    return kwargs



def _spubert_pretrain_batch_kwargs(data, *, scene: bool):
    kwargs = {
        "spatial_ids": data["spatial_ids"],
        "temporal_ids": data["temporal_ids"],
        "segment_ids": data["segment_ids"],
        "attn_mask": data["attn_mask"],
        "traj_mask": data["traj_mask"],
        "traj_lbl": data["traj_lbl"],
        "near_lbl": data["near_lbl"],
    }
    if scene:
        kwargs.update(_scene_batch_kwargs(data, include_env_params=True))
    return kwargs



def _spubert_finetune_batch_kwargs(data, *, scene: bool, kld_weight=None, traj_weight=None, goal_weight=None):
    kwargs = {
        "mgp_spatial_ids": data["mgp_spatial_ids"],
        "mgp_temporal_ids": data["mgp_temporal_ids"],
        "mgp_segment_ids": data["mgp_segment_ids"],
        "mgp_attn_mask": data["mgp_attn_mask"],
        "tgp_spatial_ids": data["tgp_spatial_ids"],
        "tgp_temporal_ids": data["tgp_temporal_ids"],
        "tgp_segment_ids": data["tgp_segment_ids"],
        "tgp_attn_mask": data["tgp_attn_mask"],
        "traj_lbl": data["traj_lbl"],
        "goal_lbl": data["goal_lbl"],
    }
    if kld_weight is not None:
        kwargs["kld_weight"] = kld_weight
    if traj_weight is not None:
        kwargs["traj_weight"] = traj_weight
    if goal_weight is not None:
        kwargs["goal_weight"] = goal_weight
    if scene:
        kwargs.update(_scene_batch_kwargs(data, include_env_params=True))
    return kwargs



def _spubert_inference_kwargs(data, *, scene: bool, d_sample):
    kwargs = {
        "mgp_spatial_ids": data["mgp_spatial_ids"],
        "mgp_temporal_ids": data["mgp_temporal_ids"],
        "mgp_segment_ids": data["mgp_segment_ids"],
        "mgp_attn_mask": data["mgp_attn_mask"],
        "tgp_temporal_ids": data["tgp_temporal_ids"],
        "tgp_segment_ids": data["tgp_segment_ids"],
        "tgp_attn_mask": data["tgp_attn_mask"],
        "d_sample": d_sample,
    }
    if scene:
        kwargs.update(_scene_batch_kwargs(data, include_env_params=False))
    return kwargs



class SBertPlusPTTrainer(SimpleTrainerBase):
    def __init__(self, train_dataloader=None, val_dataloader=None, tb_writer=None, args=None):
        super().__init__(train_dataloader=train_dataloader, val_dataloader=val_dataloader, tb_writer=tb_writer, args=args)
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.sbert_cfgs = _build_spubert_pt_config(args)
        self.model = SBertPlusPTModel(self.sbert_cfgs)
        self.parallel, self.model = _data_parallel(self.model, args.cuda)
        self.model.to(self.device)
        self.optim, self.lr_scheduler = _build_train_stack(self.model, self.train_dataloader, args, eps=1e-6)

    def val_iteration(self, epoch, data_loader):
        total_loss = 0.0
        total_mtp_loss = 0.0
        total_sip_loss = 0.0
        for _, it_data in _make_eval_pbar(data_loader, epoch, "val"):
            data = _to_device(it_data, self.device)
            outputs = self.model(**_spubert_pretrain_batch_kwargs(data, scene=self.sbert_cfgs.scene))
            total_mtp_loss += outputs["mtp_loss"].mean().item()
            if self.sbert_cfgs.sip and outputs["sip_loss"] is not None:
                total_sip_loss += outputs["sip_loss"].mean().item()
            total_loss += outputs["total_loss"].mean().item()

        n = len(data_loader)
        total_loss /= n
        total_mtp_loss /= n
        total_sip_loss /= n
        print(f"total={total_loss:.6f}, mtp={total_mtp_loss:.6f}, sip={total_sip_loss:.6f}")
        return total_loss, {"lr": self.optim.param_groups[0]["lr"]}


class SBertPlusFTTrainer(SimpleTrainerBase):
    def __init__(self, train_dataloader=None, val_dataloader=None, tb_writer=None, args=None):
        super().__init__(train_dataloader=train_dataloader, val_dataloader=val_dataloader, tb_writer=tb_writer, args=args)
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.sbert_tgp_cfgs = _build_spubert_tgp_config(args)
        self.sbert_mgp_cfgs = _build_spubert_mgp_config(args)
        self.sbert_cfgs = SBertPlusFTConfig(self.sbert_tgp_cfgs, self.sbert_mgp_cfgs, share=args.share)
        self.model = SBertPlusFTModel(self.sbert_tgp_cfgs, self.sbert_mgp_cfgs, self.sbert_cfgs)

        if args.train_mode == "pt":
            state_dict = torch.load(_resolve_pretrain_checkpoint(args), map_location="cpu")
            self.model.tgp_model.sbert.load_state_dict(state_dict)
            self.model.mgp_model.sbert.load_state_dict(state_dict)
        else:
            print("from_scratch")

        self.parallel, self.model = _data_parallel(self.model, args.cuda)
        self.model.to(self.device)
        (
            self.mgp_optim,
            self.tgp_optim,
            self.mgp_lr_scheduler,
            self.tgp_lr_scheduler,
        ) = _build_dual_train_stack(self.model, self.train_dataloader, args, eps=1e-6)
        self.kld_weight_scheduler = CyclicScherduler(
            name="kld_weight",
            end_val=args.kld_weight,
            epoch=args.epoch,
            num_gpu=torch.cuda.device_count(),
            func=frange_cycle_sigmoid,
            num_cycle=0,
        )
        self.traj_weight_scheduler = CyclicScherduler(
            name="traj_weight",
            end_val=args.traj_weight,
            epoch=args.epoch,
            num_gpu=torch.cuda.device_count(),
            func=frange_cycle_sigmoid,
            num_cycle=0,
        )
        self.goal_weight_scheduler = CyclicScherduler(
            name="goal_weight",
            end_val=args.goal_weight,
            epoch=args.epoch,
            num_gpu=torch.cuda.device_count(),
            func=frange_cycle_cosine,
            num_cycle=0,
            reverse=True,
        )

    def test(self, epoch, data_loader, d_sample, k_sample):
        self.model.eval()
        with torch.no_grad():
            total_aderror = 0
            total_fderror = 0
            total_gderror = 0
            total_data = 0
            for _, it_data in _make_eval_pbar(data_loader, epoch, "test"):
                data = _to_device(it_data, self.device)
                outputs = self.model.inference(**_spubert_inference_kwargs(data, scene=self.args.scene, d_sample=d_sample))
                outputs["pred_trajs"] = torch.einsum("bkts,b->bkts", outputs["pred_trajs"], data["scales"])
                outputs["pred_goals"] = torch.einsum("bks,b->bks", outputs["pred_goals"], data["scales"])
                data["traj_lbl"] = torch.einsum("bts,b->bts", data["traj_lbl"], data["scales"])
                data["goal_lbl"] = torch.einsum("bs,b->bs", data["goal_lbl"], data["scales"])
                gderror, aderror, fderror = bom_loss_3(
                    outputs["pred_goals"],
                    outputs["pred_trajs"],
                    data["goal_lbl"],
                    data["traj_lbl"],
                    k_sample,
                    output_dim=self.args.output_dim,
                )
                total_aderror += aderror
                total_fderror += fderror
                total_gderror += gderror
                total_data += len(data["mgp_spatial_ids"])

            return total_aderror / total_data, total_fderror / total_data, total_gderror / total_data

    def val_iteration(self, epoch, data_loader):
        total_mgp_loss = 0.0
        total_tgp_loss = 0.0
        total_tgp_col_loss = 0.0
        total_mgp_col_loss = 0.0
        total_kld_loss = 0.0
        total_gde_loss = 0.0
        total_ade_loss = 0.0
        total_fde_loss = 0.0

        for _, it_data in _make_eval_pbar(data_loader, epoch, "val"):
            data = _to_device(it_data, self.device)
            kld_weight = self.kld_weight_scheduler.get_param(train=False).to(self.device)
            traj_weight = self.traj_weight_scheduler.get_param(train=False).to(self.device)
            goal_weight = self.goal_weight_scheduler.get_param(train=False).to(self.device)
            outputs = self.model(
                **_spubert_finetune_batch_kwargs(
                    data,
                    scene=self.args.scene,
                    kld_weight=kld_weight,
                    traj_weight=traj_weight,
                    goal_weight=goal_weight,
                )
            )
            if self.args.scene and self.args.col_weight > 0:
                total_mgp_col_loss += outputs["mgp_col_loss"].mean().item()
                total_tgp_col_loss += outputs["tgp_col_loss"].mean().item()

            total_kld_loss += outputs["kld_loss"].mean().item()
            total_ade_loss += outputs["ade_loss"].mean().item()
            total_fde_loss += outputs["fde_loss"].mean().item()
            total_gde_loss += outputs["gde_loss"].mean().item()
            total_mgp_loss += outputs["mgp_loss"].mean().item()
            total_tgp_loss += outputs["tgp_loss"].mean().item()

        n = len(data_loader)
        total_kld_loss /= n
        total_ade_loss /= n
        total_fde_loss /= n
        total_gde_loss /= n
        total_mgp_loss /= n
        total_tgp_loss /= n
        if self.args.scene and self.args.col_weight > 0:
            total_mgp_col_loss /= n
            total_tgp_col_loss /= n
        total_loss = total_mgp_loss + total_tgp_loss
        print(f"[MGP] total_mgp={total_mgp_loss:.6f}, kld={total_kld_loss:.6f}, gde={total_gde_loss:.6f}, col={total_mgp_col_loss:.6f}")
        print(f"[TGP] total_tgp={total_tgp_loss:.6f}, ade={total_ade_loss:.6f}, fde={total_fde_loss:.6f}, col={total_tgp_col_loss:.6f}")
        return total_loss, {
            "total_loss": total_loss,
            "traj_weight": traj_weight,
            "goal_weight": goal_weight,
            "kld_weight": kld_weight,
            "kld_loss": total_kld_loss,
            "gde_loss": total_gde_loss,
            "mgp_col_loss": total_mgp_col_loss,
            "ade_loss": total_ade_loss,
            "fde_loss": total_fde_loss,
            "tgp_col_loss": total_tgp_col_loss,
        }



def _train_spubert_pretrain(trainer, args, epoch: int):
    trainer.model.train()
    total_loss = 0.0
    total_mse = 0.0

    pbar = _make_pbar(trainer.train_dataloader, epoch, args.epoch)
    for step, it_data in pbar:
        data = _to_device(it_data, trainer.device)
        trainer.optim.zero_grad(set_to_none=True)

        outputs = trainer.model(**_spubert_pretrain_batch_kwargs(data, scene=trainer.sbert_cfgs.scene))
        loss = outputs["total_loss"].mean()
        mse = outputs["mtp_loss"].mean()

        loss.backward()
        if trainer.args.clip_grads:
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
        trainer.optim.step()
        if not _scheduler_steps_per_epoch(trainer.args.lr_scheduler):
            trainer.lr_scheduler.step()

        loss_val = float(loss.item())
        mse_val = float(mse.item())
        total_loss += loss_val
        total_mse += mse_val
        _postfix(pbar, step=step, total_loss=total_loss, total_mse=total_mse, loss=loss_val, mse=mse_val)

    avg_loss = _safe_avg(total_loss, len(trainer.train_dataloader))
    if _scheduler_steps_per_epoch(trainer.args.lr_scheduler):
        trainer.lr_scheduler.step()
    return avg_loss, {"lr": trainer.optim.param_groups[0]["lr"]}



def _train_spubert_finetune(trainer, args, epoch: int):
    trainer.model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_kld_loss = 0.0
    total_gde_loss = 0.0
    total_ade_loss = 0.0
    total_fde_loss = 0.0

    pbar = _make_pbar(trainer.train_dataloader, epoch, args.epoch)
    for step, it_data in pbar:
        kld_weight = trainer.kld_weight_scheduler.get_param(train=True).to(trainer.device)
        traj_weight = trainer.traj_weight_scheduler.get_param(train=True).to(trainer.device)
        goal_weight = trainer.goal_weight_scheduler.get_param(train=True).to(trainer.device)

        data = _to_device(it_data, trainer.device)
        trainer.mgp_optim.zero_grad(set_to_none=True)
        trainer.tgp_optim.zero_grad(set_to_none=True)

        outputs = trainer.model(
            **_spubert_finetune_batch_kwargs(
                data,
                scene=args.scene,
                kld_weight=kld_weight,
                traj_weight=traj_weight,
                goal_weight=goal_weight,
            )
        )

        mgp_loss = outputs["mgp_loss"].mean()
        tgp_loss = outputs["tgp_loss"].mean()
        loss = mgp_loss + tgp_loss
        mse = outputs["ade_loss"].mean()

        mgp_loss.backward()
        tgp_loss.backward()
        if trainer.args.clip_grads:
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)

        trainer.mgp_optim.step()
        trainer.tgp_optim.step()
        if "it_" in trainer.args.lr_scheduler:
            trainer.mgp_lr_scheduler.step()
            trainer.tgp_lr_scheduler.step()

        loss_val = float(loss.item())
        mse_val = float(mse.item())
        total_loss += loss_val
        total_mse += mse_val
        total_kld_loss += float(outputs["kld_loss"].mean().item())
        total_gde_loss += float(outputs["gde_loss"].mean().item())
        total_ade_loss += float(outputs["ade_loss"].mean().item())
        total_fde_loss += float(outputs["fde_loss"].mean().item())
        _postfix(pbar, step=step, total_loss=total_loss, total_mse=total_mse, loss=loss_val, mse=mse_val)

    if "ep_" in trainer.args.lr_scheduler or _scheduler_steps_per_epoch(trainer.args.lr_scheduler):
        trainer.mgp_lr_scheduler.step()
        trainer.tgp_lr_scheduler.step()

    trainer.kld_weight_scheduler.step()
    trainer.traj_weight_scheduler.step()
    trainer.goal_weight_scheduler.step()

    n = len(trainer.train_dataloader)
    avg_loss = _safe_avg(total_loss, n)
    return avg_loss, {
        "total_loss": avg_loss,
        "kld_loss": _safe_avg(total_kld_loss, n),
        "gde_loss": _safe_avg(total_gde_loss, n),
        "ade_loss": _safe_avg(total_ade_loss, n),
        "fde_loss": _safe_avg(total_fde_loss, n),
        "mgp_lr": trainer.mgp_optim.param_groups[0]["lr"],
        "tgp_lr": trainer.tgp_optim.param_groups[0]["lr"],
    }



def run_train_epoch(trainer, args, epoch: int):
    if args.mode == "pretrain":
        return _train_spubert_pretrain(trainer, args, epoch)
    return _train_spubert_finetune(trainer, args, epoch)

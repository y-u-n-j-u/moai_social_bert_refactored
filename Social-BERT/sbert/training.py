from __future__ import annotations

import torch

from src.loss import ADError, FDError
from src.utils import (
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
)

from .model import SBertConfig, SBertFTModel, SBertPTModel



def _build_sbert_cfg(args):
    return SBertConfig(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
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
        backbone_type=args.backbone_type,
    )



def _build_train_stack(model, train_dataloader, args, *, eps):
    optim = build_adamw(model.parameters(), lr=args.lr, eps=eps)
    lr_scheduler = build_scheduler_from_args(args, optim, len(train_dataloader))
    return optim, lr_scheduler



def _base_batch_kwargs(data):
    return {
        "spatial_ids": data["spatial_ids"],
        "temporal_ids": data["temporal_ids"],
        "segment_ids": data["segment_ids"],
        "attn_mask": data["attn_mask"],
    }



def _pretrain_batch_kwargs(data):
    kwargs = _base_batch_kwargs(data)
    kwargs.update(
        {
            "traj_mask": data["traj_mask"],
            "traj_lbl": data["traj_lbl"],
            "near_lbl": data["near_lbl"],
        }
    )
    return kwargs



def _finetune_batch_kwargs(data):
    kwargs = _base_batch_kwargs(data)
    kwargs["traj_lbl"] = data["traj_lbl"]
    return kwargs



def _forward_sbert(model, data, *, mode: str, train: bool):
    batch_kwargs = _pretrain_batch_kwargs(data) if mode == "pretrain" else _finetune_batch_kwargs(data)
    return model(train=train, **batch_kwargs)



def _run_single_optimizer_epoch(trainer, args, epoch: int, *, mode: str, metric_key: str):
    trainer.model.train()
    total_loss = 0.0
    total_metric = 0.0

    pbar = _make_pbar(trainer.train_dataloader, epoch, args.epoch)
    for step, it_data in pbar:
        data = _to_device(it_data, trainer.device)
        trainer.optim.zero_grad(set_to_none=True)

        outputs = _forward_sbert(trainer.model, data, mode=mode, train=True)
        loss = outputs["total_loss"].mean()
        metric = outputs[metric_key].mean()

        loss.backward()
        if trainer.args.clip_grads:
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
        trainer.optim.step()
        if not _scheduler_steps_per_epoch(trainer.args.lr_scheduler):
            trainer.lr_scheduler.step()

        loss_val = float(loss.item())
        metric_val = float(metric.item())
        total_loss += loss_val
        total_metric += metric_val
        _postfix(pbar, step=step, total_loss=total_loss, total_mse=total_metric, loss=loss_val, mse=metric_val)

    avg_loss = _safe_avg(total_loss, len(trainer.train_dataloader))
    if _scheduler_steps_per_epoch(trainer.args.lr_scheduler):
        trainer.lr_scheduler.step()
    return avg_loss, {"lr": trainer.optim.param_groups[0]["lr"]}



class SBertPTTrainer(SimpleTrainerBase):
    def __init__(self, train_dataloader=None, val_dataloader=None, tb_writer=None, args=None):
        super().__init__(train_dataloader=train_dataloader, val_dataloader=val_dataloader, tb_writer=tb_writer, args=args)
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.sbert_cfgs = _build_sbert_cfg(args)
        self.model = SBertPTModel(self.sbert_cfgs)
        self.parallel, self.model = _data_parallel(self.model, args.cuda)
        self.model.to(self.device)
        self.optim, self.lr_scheduler = _build_train_stack(self.model, self.train_dataloader, args, eps=1e-8)

    def val_iteration(self, epoch, data_loader):
        total_loss = 0.0
        total_mtp_loss = 0.0
        total_sip_loss = 0.0
        for _, it_data in _make_eval_pbar(data_loader, epoch, "val"):
            data = _to_device(it_data, self.device)
            outputs = _forward_sbert(self.model, data, mode="pretrain", train=True)
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


class SBertFTTrainer(SimpleTrainerBase):
    def __init__(self, train_dataloader=None, val_dataloader=None, tb_writer=None, args=None):
        super().__init__(train_dataloader=train_dataloader, val_dataloader=val_dataloader, tb_writer=tb_writer, args=args)
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.sbert_cfgs = _build_sbert_cfg(args)
        self.model = SBertFTModel(self.sbert_cfgs)
        if args.train_mode == "pt":
            self.model.sbert.load_state_dict(torch.load(_resolve_pretrain_checkpoint(args), map_location="cpu"))
            print("Learning moai_social_bert from the pre-trained.")
        else:
            print("Learning moai_social_bert from scratch.")

        self.parallel, self.model = _data_parallel(self.model, args.cuda)
        self.model.to(self.device)
        self.optim, self.lr_scheduler = _build_train_stack(self.model, self.train_dataloader, args, eps=1e-8)

    def test(self, epoch, data_loader):
        self.model.eval()
        with torch.no_grad():
            total_aderror = 0
            total_fderror = 0
            total_data = 0
            for _, it_data in _make_eval_pbar(data_loader, epoch, "test"):
                data = _to_device(it_data, self.device)
                outputs = _forward_sbert(self.model, data, mode="finetune", train=False)
                data["traj_lbl"] = torch.einsum("ijk,i->ijk", data["traj_lbl"], data["scales"])
                outputs["pred_traj"] = torch.einsum("ijk,i->ijk", outputs["pred_traj"], data["scales"])
                total_aderror += ADError(pred_traj=outputs["pred_traj"], gt_traj=data["traj_lbl"])
                total_fderror += FDError(
                    pred_final_pos=outputs["pred_traj"][:, -1, :].squeeze(dim=1),
                    gt_final_pos=data["traj_lbl"][:, -1, :].squeeze(dim=1),
                )
                total_data += len(data["spatial_ids"])

            return total_aderror / total_data, total_fderror / total_data

    def val_iteration(self, epoch, data_loader):
        total_loss = 0.0
        total_ade_loss = 0.0
        total_fde_loss = 0.0
        for _, it_data in _make_eval_pbar(data_loader, epoch, "val"):
            data = _to_device(it_data, self.device)
            outputs = _forward_sbert(self.model, data, mode="finetune", train=True)
            total_ade_loss += outputs["ade_loss"].mean().item()
            total_fde_loss += outputs["fde_loss"].mean().item()
            total_loss += outputs["total_loss"].mean().item()

        n = len(data_loader)
        total_loss /= n
        total_ade_loss /= n
        total_fde_loss /= n
        print(f"total={total_loss:.6f}, ade={total_ade_loss:.6f}, fde={total_fde_loss:.6f}")
        return total_loss, {"lr": self.optim.param_groups[0]["lr"]}



def _train_sbert_pretrain(trainer, args, epoch: int):
    return _run_single_optimizer_epoch(trainer, args, epoch, mode="pretrain", metric_key="mtp_loss")



def _train_sbert_finetune(trainer, args, epoch: int):
    return _run_single_optimizer_epoch(trainer, args, epoch, mode="finetune", metric_key="ade_loss")



def run_train_epoch(trainer, args, epoch: int):
    if args.mode == "pretrain":
        return _train_sbert_pretrain(trainer, args, epoch)
    return _train_sbert_finetune(trainer, args, epoch)

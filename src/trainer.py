from __future__ import annotations

import argparse
import sys

import torch
import yaml

from configs.loader import load_runtime_namespace
from .data_loader import build_loaders
from .utils import (
    CheckpointTracker,
    NullWriter,
    best_checkpoint_path,
    bootstrap_paths,
    ensure_runtime_dirs,
    epoch_checkpoint_path,
    estimate_map_length,
    estimate_num_patch,
    format_metric_message,
    normalize_framework,
    resolve_checkpoint_path,
    save_state_dict,
    set_seed,
    set_workdir,
)


def _base_stream_length(obs_len: int, pred_len: int, num_nbr: int) -> int:
    return (obs_len + 1) * (num_nbr + 1) + pred_len


def _scene_tokens(args) -> int:
    if not getattr(args, "scene", False):
        return 0
    if hasattr(args, "num_patch"):
        return int(getattr(args, "num_patch"))
    map_length = estimate_map_length(float(args.env_range) * 2.0, float(args.env_resol))
    return int(estimate_num_patch(map_length, int(args.patch_size)))


def format_token_contract_report(args) -> str:
    framework = normalize_framework(args.framework)
    obs_len = int(args.obs_len)
    pred_len = int(args.pred_len)
    num_nbr = int(args.num_nbr)
    base_stream_len = _base_stream_length(obs_len, pred_len, num_nbr)

    lines: list[str] = ["[TOKEN_CONTRACT]"]
    if framework == "sbert":
        lines.extend(
            [
                f"framework=sbert mode={args.mode}",
                f"stream_count=1 total_tokens={base_stream_len}",
                f"target_prefix_tokens={1 + obs_len + pred_len}",
                f"neighbor_tokens={num_nbr * (obs_len + 1)}",
                "current_stream=[SOT] + target + neighbors([SEP] + obs)",
                "encoder_path=spatial linear -> BERT(inputs_embeds, position_ids=temporal_ids, token_type_ids=segment_ids)",
            ]
        )
        if args.mode == "pretrain":
            lines.append("training_objective=masked trajectory prediction + optional SIP classification")
        else:
            lines.append("training_objective=masked future trajectory decoding")
        return "\n".join(lines)

    lines.extend(
        [
            f"framework=spubert mode={args.mode}",
            f"base_stream_tokens={base_stream_len}",
            f"scene_patch_tokens={_scene_tokens(args)}",
        ]
    )
    if args.mode == "pretrain":
        lines.extend(
            [
                "stream_count=1 (+ optional scene patch tokens)",
                "current_stream=[SOT] + target + neighbors([SEP] + obs) + scene_patch_tokens",
            ]
        )
    else:
        lines.extend(
            [
                f"mgp_stream_tokens={base_stream_len}",
                f"tgp_stream_tokens={base_stream_len}",
                "mgp_stream=[SOT] + obs + [PAD]*(pred-1) + [MSK] + neighbors([SEP] + obs)",
                "tgp_stream=[SOT] + obs + [MSK]*(pred-1) + [GOAL] + neighbors([SEP] + obs)",
            ]
        )

    if args.scene:
        lines.extend(
            [
                "scene_input_source=flattened occupancy-map patches",
                "scene_encoder_path=flattened map patches -> linear patch embedding -> shared BERT",
            ]
        )
    lines.append("trajectory_encoder_path=spatial linear -> shared BERT")
    return "\n".join(lines)


def build_trainer_class(args):
    bootstrap_paths()
    framework = normalize_framework(args.framework)
    if framework == "sbert":
        from sbert.training import SBertFTTrainer, SBertPTTrainer

        mapping = {
            ("sbert", "pretrain"): SBertPTTrainer,
            ("sbert", "finetune"): SBertFTTrainer,
        }
        return mapping[(framework, args.mode)]

    from spubert.training import SBertPlusFTTrainer, SBertPlusPTTrainer

    mapping = {
        ("spubert", "pretrain"): SBertPlusPTTrainer,
        ("spubert", "finetune"): SBertPlusFTTrainer,
    }
    return mapping[(framework, args.mode)]


def run_train_epoch(trainer, args, epoch: int):
    bootstrap_paths()
    framework = normalize_framework(args.framework)
    if framework == "sbert":
        from sbert.training import run_train_epoch as run_sbert_train_epoch

        return run_sbert_train_epoch(trainer, args, epoch)

    from spubert.training import run_train_epoch as run_spubert_train_epoch

    return run_spubert_train_epoch(trainer, args, epoch)


def build_parser(command: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=command,
        description=f"{command} pipeline for moai_social_bert_refactored",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def parse_args(command: str, argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    cli_args = build_parser(command).parse_args(argv)
    return load_runtime_namespace(cli_args.config, command=command, cli_dry_run=cli_args.dry_run)


def _print_epoch(prefix: str, epoch: int, train_loss: float, val_loss: float, metric_msg: str = "") -> None:
    message = f"[{prefix}] epoch={epoch + 1:04d} train={train_loss:.6f} val={val_loss:.6f}"
    if metric_msg:
        message = f"{message} {metric_msg}"
    print(message)


def _print_dry_run(command: str, args: argparse.Namespace) -> None:
    print(f"[DRY_RUN] {command} effective configuration")
    print(yaml.safe_dump(args._config, sort_keys=False, allow_unicode=True).strip())
    print(format_token_contract_report(args))


def run_train(argv: list[str] | None = None) -> int:
    bootstrap_paths()
    set_workdir()
    ensure_runtime_dirs()
    args = parse_args(command="train", argv=argv)
    set_seed(args.seed, use_cuda=args.cuda)

    if args.dry_run:
        _print_dry_run("train", args)
        return 0

    train_loader, val_loader, test_loader = build_loaders(args, for_test=False)
    trainer_class = build_trainer_class(args)
    writer = NullWriter()

    trainer = trainer_class(train_dataloader=train_loader, val_dataloader=val_loader, args=args, tb_writer=writer)
    tracker = CheckpointTracker(patience=args.patience)
    checkpoint_interval = max(int(getattr(args, "checkpoint_interval", 20)), 1)

    try:
        for epoch in range(args.epoch):
            train_loss, params = run_train_epoch(trainer, args, epoch)
            writer.add_scalar("loss/train", train_loss, epoch)
            if "lr" in params:
                writer.add_scalar("lr/main", params["lr"], epoch)
            if "mgp_lr" in params:
                writer.add_scalar("lr/mgp", params["mgp_lr"], epoch)
            if "tgp_lr" in params:
                writer.add_scalar("lr/tgp", params["tgp_lr"], epoch)

            val_loss = train_loss
            if val_loader is not None:
                val_loss, _ = trainer.val(epoch)
                writer.add_scalar("loss/val", val_loss, epoch)

            metric_msg = ""
            if args.test and test_loader is not None and ((epoch + 1) % max(args.eval_interval, 1) == 0):
                if args.framework == "spubert" and args.mode == "finetune":
                    metrics = trainer.test(epoch, test_loader, args.d_sample, args.k_sample)
                    writer.add_scalar("test/ade", float(metrics[0]), epoch)
                    writer.add_scalar("test/fde", float(metrics[1]), epoch)
                    writer.add_scalar("test/gde", float(metrics[2]), epoch)
                elif args.mode == "finetune":
                    metrics = trainer.test(epoch, test_loader)
                    writer.add_scalar("test/ade", float(metrics[0]), epoch)
                    writer.add_scalar("test/fde", float(metrics[1]), epoch)
                else:
                    metrics = ()
                metric_msg = format_metric_message(args, metrics)

            if (epoch + 1) % checkpoint_interval == 0:
                save_state_dict(args, trainer, epoch_checkpoint_path(args, epoch))

            monitor = val_loss if val_loader is not None else train_loss
            if tracker.update(monitor):
                save_state_dict(args, trainer, best_checkpoint_path(args))

            _print_epoch("TRAIN", epoch, train_loss, val_loss, metric_msg)

            if tracker.early_stop:
                print("[TRAIN] early stopping triggered")
                break
    finally:
        writer.close()

    return 0


def run_test(argv: list[str] | None = None) -> int:
    bootstrap_paths()
    set_workdir()
    ensure_runtime_dirs()
    args = parse_args(command="test", argv=argv)
    set_seed(args.seed, use_cuda=args.cuda)

    if args.dry_run:
        _print_dry_run("test", args)
        return 0

    if args.mode == "pretrain":
        print("[TEST] pretrain test is not implemented in the simplified pipeline. Use finetune mode.")
        return 2

    _, _, test_loader = build_loaders(args, for_test=True)
    trainer_class = build_trainer_class(args)
    trainer = trainer_class(train_dataloader=test_loader, val_dataloader=None, args=args, tb_writer=None)

    ckpt_path = resolve_checkpoint_path(args)
    if not ckpt_path.exists():
        print(f"[TEST] checkpoint not found: {ckpt_path}")
        return 1

    state = torch.load(str(ckpt_path), map_location=trainer.device)
    model = trainer.model.module if trainer.parallel else trainer.model
    model.load_state_dict(state)

    if args.framework == "spubert":
        ade, fde, gde = trainer.test(epoch=0, data_loader=test_loader, d_sample=args.d_sample, k_sample=args.k_sample)
        print(f"[TEST] ADE={float(ade):.6f} FDE={float(fde):.6f} GDE={float(gde):.6f}")
    else:
        ade, fde = trainer.test(epoch=0, data_loader=test_loader)
        print(f"[TEST] ADE={float(ade):.6f} FDE={float(fde):.6f}")

    return 0


def main_train() -> int:
    return run_train()


def main_test() -> int:
    return run_test()


if __name__ == "__main__":
    command = "test" if "test" in sys.argv[0] else "train"
    if command == "test":
        raise SystemExit(run_test())
    raise SystemExit(run_train())

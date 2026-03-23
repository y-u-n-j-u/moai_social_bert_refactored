from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import tqdm
import transformers
import yaml
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SBERT_ROOT = PROJECT_ROOT / "Social-BERT"
SPUBERT_ROOT = PROJECT_ROOT / "SPU-BERT"
CONFIG_ROOT = PROJECT_ROOT / "configs"
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_ROOT = PROJECT_ROOT / "models"
OUTPUT_ROOT = PROJECT_ROOT / "output"
LOGS_ROOT = PROJECT_ROOT / "logs"

FRAMEWORKS = ("sbert", "spubert")
MODES = ("pretrain", "finetune")


def bootstrap_paths() -> None:
    for path in (PROJECT_ROOT, SBERT_ROOT, SPUBERT_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def normalize_framework(framework: str | None) -> str | None:
    return framework


def canonicalize_framework_path(path_value: str) -> str:
    if not path_value:
        return path_value
    return str(Path(path_value))


def _normalize_path_component(value: str | None, fallback: str) -> str:
    text = str(value or "").strip().replace("/", "_").replace("\\", "_")
    return text or fallback


def default_model_dir(framework: str | None, dataset_name: str | None, dataset_split: str | None) -> Path:
    framework_key = _normalize_path_component(normalize_framework(framework), "framework")
    dataset_key = _normalize_path_component(dataset_name, "dataset")
    scene_key = _normalize_path_component(dataset_split, "default")
    return OUTPUT_ROOT / framework_key / dataset_key / scene_key


def _project_abs_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def resolve_model_dir(path_value: str, framework: str | None, dataset_name: str | None, dataset_split: str | None) -> str:
    default_dir = default_model_dir(framework, dataset_name, dataset_split)
    if not path_value:
        return str(default_dir)

    candidate = Path(canonicalize_framework_path(path_value))
    candidate_abs = _project_abs_path(candidate)
    framework_root = OUTPUT_ROOT / _normalize_path_component(normalize_framework(framework), "framework")
    dataset_root = framework_root / _normalize_path_component(dataset_name, "dataset")
    legacy_roots = {
        MODELS_ROOT.resolve(),
        OUTPUT_ROOT.resolve(),
        framework_root.resolve(),
        dataset_root.resolve(),
    }
    if candidate_abs in legacy_roots:
        return str(default_dir)
    return str(candidate)


def resolve_config_path(config_value: str) -> Path:
    config_path = Path(config_value)
    if config_path.is_absolute():
        return config_path

    project_candidate = PROJECT_ROOT / config_path
    if project_candidate.exists():
        return project_candidate

    config_candidate = CONFIG_ROOT / config_path
    if config_candidate.exists():
        return config_candidate

    matches = sorted(CONFIG_ROOT.rglob(config_path.name))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        joined = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in matches)
        raise ValueError(f"Ambiguous config '{config_value}'. Matches: {joined}")

    return config_candidate


def set_workdir() -> None:
    os.chdir(PROJECT_ROOT)


def set_seed(seed: int, use_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_yaml_config(config_value: str) -> dict:
    if not config_value:
        return {}

    config_path = resolve_config_path(config_value)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data


class NullWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


class CheckpointTracker:
    def __init__(self, patience: int):
        self.patience = patience
        self.best_loss: float | None = None
        self.bad_epochs = 0
        self.early_stop = False

    def update(self, loss: float) -> bool:
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            self.bad_epochs = 0
            return True

        if self.patience != -1:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.early_stop = True
        return False


def ensure_runtime_dirs() -> None:
    for path in (DATA_ROOT, DATA_ROOT / "raw", DATA_ROOT / "processed", OUTPUT_ROOT, LOGS_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def best_checkpoint_path(args) -> Path:
    name = "pretrain_model_best.pth" if args.mode == "pretrain" else "model_best.pth"
    return Path(args.model_dir) / name


def epoch_checkpoint_path(args, epoch: int) -> Path:
    prefix = "pretrain_" if args.mode == "pretrain" else ""
    return Path(args.model_dir) / f"{prefix}checkpoint_epoch_{epoch + 1}.pth"


def resolve_checkpoint_path(args) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)
    return best_checkpoint_path(args)


def save_state_dict(args, trainer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = trainer.model.module if getattr(trainer, "parallel", False) else trainer.model
    target = model.sbert if args.mode == "pretrain" and hasattr(model, "sbert") else model
    torch.save(target.state_dict(), str(path))


def format_metric_message(args, metrics: tuple[float, ...]) -> str:
    framework = normalize_framework(args.framework)
    if framework == "spubert" and args.mode == "finetune":
        ade, fde, gde = metrics
        return f"ade={float(ade):.6f} fde={float(fde):.6f} gde={float(gde):.6f}"
    if args.mode == "finetune":
        ade, fde = metrics
        return f"ade={float(ade):.6f} fde={float(fde):.6f}"
    return ""


def _resolve_pretrain_checkpoint(args) -> str:
    checkpoint = getattr(args, "pretrain_checkpoint", "")
    if checkpoint:
        return checkpoint
    model_dir = getattr(args, "model_dir", getattr(args, "output_path", "./output"))
    return str(Path(model_dir) / "pretrain_model_best.pth")


def estimate_map_length(map_range, map_resol):
    num_idx = map_range / map_resol
    return int(num_idx) if (num_idx % 2) == 0 else int(num_idx + 1)


def estimate_num_patch(map_length, patch_size):
    num_side_patch = (map_length + patch_size - 1) // patch_size
    return num_side_patch ** 2


class CyclicScherduler:
    def __init__(self, name="", start_val=0, end_val=1.0, epoch=100, num_cycle=2, num_gpu=0, ratio=0.5, func=None, monotonic=False, reverse=False):
        self.name = name
        self.epoch = 0
        self.num_gpu = num_gpu

        if num_cycle == 0:
            self.weights = np.full(epoch, end_val)
        else:
            self.weights = end_val * func(start_val, 1.0, epoch, num_cycle, ratio)
            if monotonic:
                start_epoch = int(epoch / num_cycle)
                self.weights[start_epoch:] = end_val
        if reverse:
            self.weights = self.weights[::-1]

    def step(self):
        self.epoch += 1

    def get_param(self, train=True):
        value = self.weights[self.epoch] if train else 1.0
        if self.num_gpu == 0:
            return torch.tensor(value)
        return torch.ones([self.num_gpu]) * value


def frange_cycle_sigmoid(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    values = np.ones(n_epoch)
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)
    for cycle_idx in range(n_cycle):
        value, inner_idx = start, 0
        while value <= stop:
            values[int(inner_idx + cycle_idx * period)] = 1.0 / (1.0 + np.exp(-(value * 12.0 - 6.0)))
            value += step
            inner_idx += 1
    return values


def frange_cycle_cosine(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    values = np.ones(n_epoch)
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)
    for cycle_idx in range(n_cycle):
        value, inner_idx = start, 0
        while value <= stop:
            values[int(inner_idx + cycle_idx * period)] = 0.5 - 0.5 * math.cos(value * math.pi)
            value += step
            inner_idx += 1
    return values


class SimpleTrainerBase:
    def __init__(self, train_dataloader=None, val_dataloader=None, tb_writer=None, args=None):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tb_writer = tb_writer
        self.args = args
        self.parallel = False

    def val(self, epoch):
        self.model.eval()
        with torch.no_grad():
            return self.val_iteration(epoch, self.val_dataloader)


def _data_parallel(model, use_cuda):
    if use_cuda and torch.cuda.device_count() > 1:
        return True, nn.parallel.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    return False, model


def _resolve_warmup_steps(warm_up, total_units: int) -> int:
    warm_up = float(warm_up)
    if warm_up <= 0:
        return 0
    if warm_up < 1:
        return int(total_units * warm_up)
    return int(warm_up)



def _build_scheduler(name, optimizer, warm_up, steps, epochs, num_cycle, decay_step=10, decay_gamma=0.5):
    step_warm_up = _resolve_warmup_steps(warm_up, int(steps))
    epoch_warm_up = _resolve_warmup_steps(warm_up, int(epochs))
    if name in {"linear", "it_linear"}:
        return transformers.get_scheduler("linear", optimizer=optimizer, num_warmup_steps=step_warm_up, num_training_steps=steps)
    if name in {"cosine", "it_cosine"}:
        return transformers.get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=step_warm_up, num_training_steps=steps)
    if name == "cyclic":
        return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=step_warm_up,
            num_cycles=max(num_cycle, 1),
            num_training_steps=steps,
        )
    if name == "ep_linear":
        return transformers.get_scheduler("linear", optimizer=optimizer, num_warmup_steps=epoch_warm_up, num_training_steps=epochs)
    if name == "ep_cosine":
        return transformers.get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=epoch_warm_up, num_training_steps=epochs)
    if name == "epl_reduce":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.2, patience=5, min_lr=1e-10)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=int(decay_step),
            gamma=float(decay_gamma),
        )
    if name == "exp":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=float(decay_gamma))
    if name == "lambda":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda epoch: float(decay_gamma) ** epoch,
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def build_adamw(parameters, *, lr: float, eps: float, weight_decay: float = 0.01):
    return torch.optim.AdamW(parameters, lr=lr, eps=eps, betas=(0.9, 0.999), weight_decay=weight_decay)


def build_scheduler_from_args(args, optimizer, steps_per_epoch: int):
    steps_per_epoch = max(int(steps_per_epoch), 1)
    return _build_scheduler(
        args.lr_scheduler,
        optimizer,
        args.warm_up,
        steps_per_epoch * int(args.epoch),
        int(args.epoch),
        int(args.num_cycle),
        getattr(args, "decay_step", 10),
        getattr(args, "decay_gamma", 0.5),
    )


def _make_eval_pbar(data_loader, epoch: int, desc: str):
    return tqdm.tqdm(
        enumerate(data_loader),
        desc=f"EP_{desc}:{epoch}",
        total=len(data_loader),
        bar_format="{l_bar}{bar:20}{r_bar}",
        dynamic_ncols=True,
        mininterval=0.1,
        leave=True,
        ascii=True,
    )


def _scheduler_steps_per_epoch(name: str) -> bool:
    return name in {"step", "exp", "lambda"}


def _to_device(batch, target_device):
    if isinstance(batch, dict):
        return {key: _to_device(value, target_device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [_to_device(value, target_device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(_to_device(value, target_device) for value in batch)
    return batch.to(target_device) if hasattr(batch, "to") else batch


def _make_pbar(data_loader, epoch: int, total_epochs: int):
    return tqdm.tqdm(
        enumerate(data_loader, start=1),
        total=len(data_loader),
        desc=f"Epoch {epoch + 1}/{total_epochs}",
        dynamic_ncols=True,
        mininterval=0.1,
        leave=True,
    )


def _safe_avg(value: float, denom: int) -> float:
    return value / max(denom, 1)


def _postfix(pbar, *, step: int, total_loss: float, total_mse: float, loss: float, mse: float):
    pbar.set_postfix(
        {
            "avg_loss": f"{_safe_avg(total_loss, step):.6f}",
            "avg_mse": f"{_safe_avg(total_mse, step):.6f}",
            "loss": f"{loss:.6f}",
            "mse": f"{mse:.6f}",
        }
    )


class MultiKMeans:
    def __init__(self, n_clusters, n_kmeans, max_iter=100, tol=0.0001, verbose=0, mode="euclidean", minibatch=None):
        self.n_clusters = n_clusters
        self.n_kmeans = n_kmeans
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.mode = mode
        self.minibatch = minibatch
        self.centroids = None
        self.num_points_in_clusters = None

    @staticmethod
    def cos_sim(a, b):
        a_norm = a.norm(dim=-1, keepdim=True)
        b_norm = b.norm(dim=-1, keepdim=True)
        a = a / (a_norm + 1e-8)
        b = b / (b_norm + 1e-8)
        return a @ b.transpose(-2, -1)

    @staticmethod
    def euc_sim(a, b):
        return 2 * a @ b.transpose(-2, -1) - (a ** 2).sum(dim=-1)[..., :, None] - (b ** 2).sum(dim=-1)[..., None, :]

    def max_sim(self, a, b):
        sim_func = self.cos_sim if self.mode == "cosine" else self.euc_sim
        sim = sim_func(a, b)
        return sim.max(dim=-1)

    def fit_predict(self, X):
        _, batch_size, _ = X.shape
        device_obj = X.device
        self.centroids = X[:, np.random.choice(batch_size, size=[self.n_clusters]), :]
        self.num_points_in_clusters = torch.ones(self.n_kmeans, self.n_clusters, device=device_obj)

        for _ in range(self.max_iter):
            closest = self.max_sim(a=X, b=self.centroids)[1]
            uniques = [closest[i].unique(return_counts=True) for i in range(self.n_kmeans)]
            expanded_closest = closest[:, None].expand(-1, self.n_clusters, -1)
            mask = (expanded_closest == torch.arange(self.n_clusters, device=device_obj)[None, :, None]).float()
            c_grad = mask @ X / mask.sum(-1, keepdim=True)
            c_grad[c_grad != c_grad] = 0
            error = (c_grad - self.centroids).pow(2).sum()
            for j in range(self.n_kmeans):
                self.num_points_in_clusters[j, uniques[j][0]] += uniques[j][1]
            self.centroids = c_grad
            if error <= self.tol * self.n_kmeans:
                break

        return self.centroids

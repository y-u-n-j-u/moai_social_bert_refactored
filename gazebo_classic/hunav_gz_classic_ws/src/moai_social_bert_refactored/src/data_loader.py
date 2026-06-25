from __future__ import annotations

import copy

from torch.utils.data import DataLoader

from .utils import bootstrap_paths, normalize_framework


def dataset_class(framework: str, dataset_name: str):
    bootstrap_paths()
    framework = normalize_framework(framework)

    if framework == "sbert":
        from sbert.datasets.ethucy_star import ETHUCYSTARDataset
        from sbert.datasets.ethucy_sbert import ETHUCYSBertDataset
        from sbert.datasets.jrdb import JRDBDataset
        from sbert.datasets.sdd_sbert import SDDSBertDataset

        mapping = {
            "ethucy_star": ETHUCYSTARDataset,
            "ethucy_sbert": ETHUCYSBertDataset,
            "sdd_sbert": SDDSBertDataset,
            "JRDB": JRDBDataset,
        }
    else:
        from spubert.datasets.ethucy import ETHUCYDataset
        from spubert.datasets.ethucy_sbert import ETHUCYSBertDataset
        from spubert.datasets.ethucy_star import ETHUCYSTARDataset
        from spubert.datasets.ethucy_tpp import ETHUCYTPPDataset
        from spubert.datasets.sdd_sbert import SDDSBertDataset

        mapping = {
            "ethucy": ETHUCYDataset,
            "ethucy_tpp": ETHUCYTPPDataset,
            "ethucy_star": ETHUCYSTARDataset,
            "ethucy_sbert": ETHUCYSBertDataset,
            "sdd_sbert": SDDSBertDataset,
        }

    if dataset_name not in mapping:
        supported = ", ".join(sorted(mapping))
        raise ValueError(
            f"Unsupported dataset_name='{dataset_name}' for framework='{framework}'. Supported: {supported}"
        )
    return mapping[dataset_name]


def _build_dataset(args, split: str):
    ds_args = copy.copy(args)
    if ds_args.dataset_name == "sdd_sbert":
        ds_args.dataset_split = "default"

    ds_cls = dataset_class(ds_args.framework, ds_args.dataset_name)
    return ds_cls(split=split, args=ds_args)


def build_loaders(args, for_test: bool = False):
    if for_test:
        test_dataset = _build_dataset(args, split="test")
        test_batch_size = args.test_batch_size if args.test_batch_size > 0 else args.batch_size
        test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            num_workers=args.num_worker,
            shuffle=args.shuffle,
        )
        return None, None, test_loader

    train_dataset = _build_dataset(args, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_worker,
        shuffle=True,
    )

    val_loader = None
    if normalize_framework(args.framework) == "spubert" and args.dataset_name == "ethucy_tpp" and args.mode == "finetune":
        val_dataset = _build_dataset(args, split="val")
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_worker,
            shuffle=False,
        )

    test_loader = None
    if args.test:
        test_dataset = _build_dataset(args, split="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_worker,
            shuffle=False,
        )

    return train_loader, val_loader, test_loader

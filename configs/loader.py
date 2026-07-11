from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Any

from src.utils import load_yaml_config, normalize_framework, resolve_config_path, resolve_model_dir

DEFAULT_CONFIG: dict[str, Any] = {
    'experiment': {
        'name': '',
        'framework': None,
        'task': None,
        'seed': 20,
    },
    'data': {
        'dataset_name': None,
        'dataset_path': './data/processed',
        'dataset_split': 'univ',
        'obs_len': 8,
        'pred_len': 12,
        'min_obs_len': 2,
        'num_nbr': 4,
        'augment': False,
        'sampling': 1.0,
        'view_range': 20.0,
        'view_angle': 2.09,
        'social_range': 2.0,
        'input_dim': 2,
        'output_dim': 2,
        'goal_dim': 2,
        'subsample_stride': 3,
        'traj_scale': 1.0,
        'goal_extra_frames': 8,
    },
    'scene': {
        'enabled': False,
        'env_range': 10.0,
        'env_resol': 0.2,
        'patch_size': 16,
        'binary': False,
    },
    'model': {
        'backbone': {
            'type': 'bert',
            'hidden_size': 256,
            'num_hidden_layers': 4,
            'num_attention_heads': 4,
            'intermediate_size': 1024,
            'dropout_prob': 0.1,
            'act_fn': 'auto',
        },
        'trajectory_tokens': {
            'use_temporal_embedding': True,
            'use_segment_embedding': True,
            'use_modal_embedding': True,
        },
        'heads': {
            'social': {
                'sip': False,
            },
            'spubert': {
                'share_backbone': False,
                'goal_hidden': 64,
                'goal_latent': 32,
                'k_sample': 20,
                'd_sample': 400,
                'normal': False,
            },
        },
    },
    'loss': {
        'traj_weight': 1.0,
        'goal_weight': 1.0,
        'kld_weight': 1.0,
        'col_weight': 0.0,
        'cvae_sigma': 1.0,
        'kld_clamp': None,
    },
    'train': {
        'train_mode': 'fs',
        'batch_size': 32,
        'test_batch_size': 64,
        'epoch': 200,
        'eval_interval': 10,
        'checkpoint_interval': 20,
        'patience': -1,
        'lr_scheduler': 'auto',
        'clip_grads': False,
        'num_cycle': 0,
        'evaluate_on_test': False,
        'use_gt_goal': False,
        'optimizer': {
            'name': 'adamw',
            'lr': 1.0e-4,
            'weight_decay': 0.01,
            'eps': 1.0e-8,
        },
        'scheduler': {
            'warm_up': 0.0,
            'decay_step': 10,
            'decay_gamma': 0.5,
        },
    },
    'runtime': {
        'cuda': False,
        'num_worker': 4,
        'model_dir': './output',
        'checkpoint': '',
        'pretrain_checkpoint': '',
        'viz': False,
        'shuffle': False,
        'dry_run': False,
    },
}

SECTION_KEYS = tuple(DEFAULT_CONFIG)

def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _normalize_scene_enabled(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {'', 'none', 'false', '0', 'no', 'off'}:
            return False
        if lowered in {'true', '1', 'yes', 'on'}:
            return True
        return True
    return bool(value)


def _canonicalize_config(data: dict[str, Any], command: str) -> dict[str, Any]:
    unknown_top_level = set(data) - set(SECTION_KEYS)
    if unknown_top_level:
        joined = ', '.join(sorted(unknown_top_level))
        raise SystemExit(f'Unsupported top-level config sections: {joined}')

    cfg = _deep_merge(deepcopy(DEFAULT_CONFIG), data)
    if cfg['experiment']['task'] is None and command == 'test':
        cfg['experiment']['task'] = 'finetune'

    cfg['experiment']['framework'] = normalize_framework(cfg['experiment']['framework'])
    cfg['scene']['enabled'] = _normalize_scene_enabled(cfg['scene']['enabled'])
    cfg['scene']['binary'] = bool(cfg['scene']['binary'])
    cfg['runtime']['cuda'] = bool(cfg['runtime']['cuda'])
    cfg['runtime']['viz'] = bool(cfg['runtime']['viz'])
    cfg['runtime']['shuffle'] = bool(cfg['runtime']['shuffle'])
    cfg['runtime']['dry_run'] = bool(cfg['runtime']['dry_run'])
    cfg['train']['clip_grads'] = bool(cfg['train']['clip_grads'])
    cfg['train']['evaluate_on_test'] = bool(cfg['train']['evaluate_on_test'])
    cfg['data']['augment'] = bool(cfg['data']['augment'])
    cfg['model']['heads']['social']['sip'] = bool(cfg['model']['heads']['social']['sip'])
    cfg['model']['heads']['spubert']['share_backbone'] = bool(cfg['model']['heads']['spubert']['share_backbone'])
    cfg['model']['heads']['spubert']['normal'] = bool(cfg['model']['heads']['spubert']['normal'])

    if cfg['data']['dataset_name'] == 'sdd_sbert':
        cfg['data']['dataset_split'] = 'default'
    return cfg


def _validate_config(cfg: dict[str, Any], config_path: str) -> None:
    missing: list[str] = []
    if not cfg['experiment']['framework']:
        missing.append('experiment.framework')
    if not cfg['experiment']['task']:
        missing.append('experiment.task')
    if not cfg['data']['dataset_name']:
        missing.append('data.dataset_name')
    if missing:
        raise SystemExit(f"Missing required config fields in {config_path}: {', '.join(missing)}")

    if cfg['experiment']['framework'] not in {'sbert', 'spubert'}:
        raise SystemExit(
            f"Unsupported framework '{cfg['experiment']['framework']}' in {config_path}. Supported: sbert, spubert"
        )
    if cfg['experiment']['task'] not in {'pretrain', 'finetune'}:
        raise SystemExit(
            f"Unsupported task '{cfg['experiment']['task']}' in {config_path}. Supported: pretrain, finetune"
        )
    if cfg['model']['backbone']['type'] != 'bert':
        raise SystemExit(f"Unsupported model.backbone.type in {config_path}: {cfg['model']['backbone']['type']}")
    if cfg['experiment']['framework'] == 'sbert' and cfg['scene']['enabled']:
        raise SystemExit(f'Social-BERT does not use a scene branch: {config_path}')


def _resolved_act_fn(cfg: dict[str, Any]) -> str:
    act_fn = cfg['model']['backbone']['act_fn']
    if act_fn and act_fn != 'auto':
        return act_fn
    return 'gelu' if cfg['experiment']['framework'] == 'sbert' else 'relu'


def _resolved_lr_scheduler(cfg: dict[str, Any]) -> str:
    scheduler = cfg['train']['lr_scheduler']
    if scheduler and scheduler != 'auto':
        return scheduler
    return 'linear' if cfg['experiment']['framework'] == 'sbert' else 'it_linear'


def _to_namespace(cfg: dict[str, Any], config_path: str, cli_dry_run: bool) -> argparse.Namespace:
    exp = cfg['experiment']
    data = cfg['data']
    scene = cfg['scene']
    model = cfg['model']
    backbone = model['backbone']
    heads = model['heads']
    loss = cfg['loss']
    train = cfg['train']
    runtime = cfg['runtime']

    model_dir = resolve_model_dir(runtime['model_dir'], exp['framework'], data['dataset_name'], data['dataset_split'])
    return argparse.Namespace(
        config=str(resolve_config_path(config_path)),
        framework=exp['framework'],
        mode=exp['task'],
        experiment_name=exp['name'],
        seed=exp['seed'],
        dataset_name=data['dataset_name'],
        dataset_path=data['dataset_path'],
        dataset_split=data['dataset_split'],
        obs_len=data['obs_len'],
        pred_len=data['pred_len'],
        min_obs_len=data['min_obs_len'],
        num_nbr=data['num_nbr'],
        aug=data['augment'],
        sampling=data['sampling'],
        view_range=data['view_range'],
        view_angle=data['view_angle'],
        social_range=data['social_range'],
        input_dim=data['input_dim'],
        output_dim=data['output_dim'],
        goal_dim=data['goal_dim'],
        subsample_stride=data['subsample_stride'],
        traj_scale=data['traj_scale'],
        goal_extra_frames=data['goal_extra_frames'],
        scene=scene['enabled'],
        env_range=scene['env_range'],
        env_resol=scene['env_resol'],
        patch_size=scene['patch_size'],
        binary_scene=scene['binary'],
        backbone_type=backbone['type'],
        hidden=backbone['hidden_size'],
        layer=backbone['num_hidden_layers'],
        head=backbone['num_attention_heads'],
        intermediate_size=backbone['intermediate_size'],
        dropout_prob=backbone['dropout_prob'],
        act_fn=_resolved_act_fn(cfg),
        sip=heads['social']['sip'],
        share=heads['spubert']['share_backbone'],
        goal_hidden=heads['spubert']['goal_hidden'],
        goal_latent=heads['spubert']['goal_latent'],
        k_sample=heads['spubert']['k_sample'],
        d_sample=heads['spubert']['d_sample'],
        normal=heads['spubert']['normal'],
        traj_weight=loss['traj_weight'],
        goal_weight=loss['goal_weight'],
        kld_weight=loss['kld_weight'],
        col_weight=loss['col_weight'],
        cvae_sigma=loss['cvae_sigma'],
        kld_clamp=loss['kld_clamp'],
        train_mode=train['train_mode'],
        batch_size=train['batch_size'],
        test_batch_size=train['test_batch_size'],
        epoch=train['epoch'],
        eval_interval=train['eval_interval'],
        checkpoint_interval=train['checkpoint_interval'],
        patience=train['patience'],
        lr_scheduler=_resolved_lr_scheduler(cfg),
        clip_grads=train['clip_grads'],
        num_cycle=train['num_cycle'],
        lr=train['optimizer']['lr'],
        weight_decay=train['optimizer']['weight_decay'],
        optimizer_eps=train['optimizer']['eps'],
        warm_up=train['scheduler']['warm_up'],
        decay_step=train['scheduler']['decay_step'],
        decay_gamma=train['scheduler']['decay_gamma'],
        test=train['evaluate_on_test'],
        use_gt_goal=train['use_gt_goal'],
        cuda=runtime['cuda'],
        num_worker=runtime['num_worker'],
        model_dir=model_dir,
        output_path=model_dir,
        checkpoint=runtime['checkpoint'],
        pretrain_checkpoint=runtime['pretrain_checkpoint'],
        viz=runtime['viz'],
        shuffle=runtime['shuffle'],
        dry_run=runtime['dry_run'] or cli_dry_run,
        _config=cfg,
    )


def load_runtime_namespace(config_value: str, command: str, cli_dry_run: bool = False) -> argparse.Namespace:
    raw_config = load_yaml_config(config_value)
    cfg = _canonicalize_config(raw_config, command)
    _validate_config(cfg, config_value)
    return _to_namespace(cfg, config_value, cli_dry_run=cli_dry_run)

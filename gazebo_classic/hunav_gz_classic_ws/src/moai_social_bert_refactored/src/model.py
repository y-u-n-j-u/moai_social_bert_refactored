from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertConfig, BertModel
from transformers.activations import get_activation

from .utils import bootstrap_paths

bootstrap_paths()

def _activation(name: str):
    return get_activation(name)


class ModalEmbedding(nn.Embedding):
    def __init__(self, modal_size, embedding_dim=512):
        super().__init__(num_embeddings=modal_size, embedding_dim=embedding_dim)


class SpatialEmbedding(nn.Module):
    def __init__(self, input_dim=2, embedding_dim=512, act_fn="relu"):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, embedding_dim)
        self.act_fn = _activation(act_fn)

    def forward(self, x):
        return self.act_fn(self.linear1(x))


class SIPPooler(nn.Module):
    def __init__(self, obs_len=8, pred_len=12, num_nbr=4):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_input_len = (obs_len + 1) * (num_nbr + 1) + pred_len
        self.pool_ids = torch.zeros(self.seq_input_len, dtype=torch.bool)
        self.pool_ids[(self.obs_len + self.pred_len + 1) :: (self.obs_len + 1)] = True

    def forward(self, x):
        x = x[:, : self.seq_input_len, :]
        return x[:, self.pool_ids, :]


class MTPPooler(nn.Module):
    def __init__(self, obs_len=8, pred_len=12, num_nbr=4):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_input_len = (obs_len + 1) * (num_nbr + 1) + pred_len
        self.pool_ids = torch.ones(self.seq_input_len, dtype=torch.bool)
        self.pool_ids[0] = False
        self.pool_ids[(self.obs_len + self.pred_len + 1) :: (self.obs_len + 1)] = False

    def forward(self, x):
        x = x[:, : self.seq_input_len, :]
        return x[:, self.pool_ids, :]


class SBertFutureTrajPooler(nn.Module):
    def __init__(self, obs_len=8, pred_len=12):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len

    def forward(self, x):
        return x[:, (self.obs_len + 1) : (self.obs_len + self.pred_len + 1), :]


class PastTrajPooler(nn.Module):
    def __init__(self, hidden_size, obs_len=8, layer_norm_eps=1e-4, dropout_prob=0.25, act_fn="relu"):
        super().__init__()
        self.obs_len = obs_len
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.act_fn = _activation(act_fn)

    def forward(self, x):
        return self.act_fn(self.linear(x[:, 1 : self.obs_len + 1, :]))


class FutureTrajPooler(nn.Module):
    def __init__(self, hidden_size, obs_len=8, pred_len=12, layer_norm_eps=1e-4, dropout_prob=0.25, act_fn="relu"):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.act_fn = _activation(act_fn)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = x[:, 1 + self.obs_len : 1 + self.obs_len + self.pred_len, :]
        x = self.act_fn(self.linear(x))
        x = self.LayerNorm(x)
        return self.dropout(x)


class FullTrajPooler(nn.Module):
    def __init__(self, hidden_size, obs_len, pred_len, layer_norm_eps=1e-4, dropout_prob=0.25, act_fn="relu"):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.act_fn = _activation(act_fn)

    def forward(self, x):
        return self.act_fn(self.linear(x[:, 1 + self.obs_len : 1 + self.obs_len + self.pred_len, :]))


class GoalPooler(nn.Module):
    def __init__(
        self,
        hidden_size,
        obs_len,
        pred_len,
        layer_norm_eps=1e-4,
        dropout_prob=0.25,
        act_fn="relu",
        token_offset=1,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.token_offset = token_offset
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.act_fn = _activation(act_fn)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = x[:, self.obs_len + self.pred_len + self.token_offset, :]
        x = self.act_fn(self.linear(x))
        x = self.LayerNorm(x)
        return self.dropout(x)


class MTPDecoder(nn.Module):
    def __init__(self, hidden_size, out_dim=2, hidden_layers=(128, 64), layer_norm_eps=1e-4, act_fn="relu"):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_layers[0])
        self.act_fn1 = _activation(act_fn)
        self.LayerNorm = nn.LayerNorm(hidden_layers[0], eps=layer_norm_eps)
        self.decoder = nn.Linear(hidden_layers[0], out_dim)
        self.decoder.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        x = self.act_fn1(self.linear1(x))
        x = self.LayerNorm(x)
        return self.decoder(x)


class SIPDecoder(nn.Module):
    def __init__(self, hidden_size, num_class=3):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_class)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class GoalDecoder(nn.Module):
    def __init__(self, enc_hidden_size, goal_latent_size, out_dim=2, hidden_layers=(128, 64), layer_norm_eps=1e-4, act_fn="relu"):
        super().__init__()
        self.linear1 = nn.Linear(enc_hidden_size + goal_latent_size, hidden_layers[0])
        self.act_fn1 = _activation(act_fn)
        self.decoder = nn.Linear(hidden_layers[0], out_dim)
        self.LayerNorm = nn.LayerNorm(hidden_layers[0], eps=layer_norm_eps)
        self.decoder.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        x = self.act_fn1(self.linear1(x))
        x = self.LayerNorm(x)
        return self.decoder(x)


class SpatialDecoder(nn.Module):
    def __init__(self, enc_hidden_size, out_dim=2, hidden_layers=(128, 64), layer_norm_eps=1e-4, act_fn="relu"):
        super().__init__()
        self.linear1 = nn.Linear(enc_hidden_size, hidden_layers[0])
        self.act_fn1 = _activation(act_fn)
        self.decoder = nn.Linear(hidden_layers[0], out_dim)
        self.LayerNorm = nn.LayerNorm(hidden_layers[0], eps=layer_norm_eps)
        self.decoder.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        x = self.act_fn1(self.linear1(x))
        x = self.LayerNorm(x)
        return self.decoder(x)


class GoalPriorNet(nn.Module):
    def __init__(self, enc_hidden_size, goal_latent_size, hidden_layers=(128, 64), act_fn="relu"):
        super().__init__()
        self.linear1 = nn.Linear(enc_hidden_size, hidden_layers[0])
        self.act_fn1 = _activation(act_fn)
        self.linear2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.act_fn2 = _activation(act_fn)
        self.linear3 = nn.Linear(hidden_layers[1], goal_latent_size * 2)

    def forward(self, x):
        x = self.act_fn1(self.linear1(x))
        x = self.act_fn2(self.linear2(x))
        return self.linear3(x)


class GoalRecognitionNet(nn.Module):
    def __init__(self, enc_hidden_size, goal_hidden_size, goal_latent_size, hidden_layers=(128, 64), act_fn="relu"):
        super().__init__()
        self.linear1 = nn.Linear(enc_hidden_size + goal_hidden_size, hidden_layers[0])
        self.act_fn1 = _activation(act_fn)
        self.linear2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.act_fn2 = _activation(act_fn)
        self.linear3 = nn.Linear(hidden_layers[1], goal_latent_size * 2)

    def forward(self, x):
        x = self.act_fn1(self.linear1(x))
        x = self.act_fn2(self.linear2(x))
        return self.linear3(x)


class GoalEncoder(nn.Module):
    def __init__(self, spatial_dim=2, goal_hidden_size=64, layer_norm_eps=1e-4, dropout_prob=0.25, act_fn="relu"):
        super().__init__()
        self.linear1 = nn.Linear(spatial_dim, goal_hidden_size)
        self.act_fn1 = _activation(act_fn)
        self.LayerNorm = nn.LayerNorm(goal_hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.act_fn1(self.linear1(x))
        x = self.LayerNorm(x)
        return self.dropout(x)


@dataclass
class TrajectoryBertBackboneConfig:
    hidden_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    intermediate_size: int = 1024
    dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1.0e-12
    initializer_range: float = 0.02


class TrajectoryBertBackbone(nn.Module):
    def __init__(self, cfg: TrajectoryBertBackboneConfig):
        super().__init__()
        bert_cfg = BertConfig(
            hidden_size=cfg.hidden_size,
            num_hidden_layers=cfg.num_hidden_layers,
            num_attention_heads=cfg.num_attention_heads,
            intermediate_size=cfg.intermediate_size,
            hidden_dropout_prob=cfg.dropout_prob,
            attention_probs_dropout_prob=cfg.attention_dropout_prob,
            max_position_embeddings=cfg.max_position_embeddings,
            type_vocab_size=cfg.type_vocab_size,
            layer_norm_eps=cfg.layer_norm_eps,
            initializer_range=cfg.initializer_range,
        )
        self.model = BertModel(bert_cfg, add_pooling_layer=True)

    def forward(self, inputs_embeds, attention_mask=None, token_type_ids=None, position_ids=None, output_attentions=False):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=None if token_type_ids is None else token_type_ids.long(),
            position_ids=None if position_ids is None else position_ids.long(),
            output_attentions=output_attentions,
            return_dict=True,
        )
        return {
            "last_hidden_state": outputs.last_hidden_state,
            "pooler_output": outputs.pooler_output,
            "attentions": outputs.attentions,
        }


class TrajectoryEncoder(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.spatial_embeddings = SpatialEmbedding(cfgs.input_dim, cfgs.hidden_size, cfgs.act_fn)
        self.backbone = TrajectoryBertBackbone(
            TrajectoryBertBackboneConfig(
                hidden_size=cfgs.hidden_size,
                num_hidden_layers=cfgs.num_layer,
                num_attention_heads=cfgs.num_head,
                intermediate_size=cfgs.intermediate_size,
                dropout_prob=cfgs.dropout_prob,
                attention_dropout_prob=cfgs.dropout_prob,
                max_position_embeddings=max(512, cfgs.obs_len + cfgs.pred_len + 32),
                type_vocab_size=getattr(cfgs, "segment_vocab_size", cfgs.num_nbr + 2),
                layer_norm_eps=cfgs.layer_norm_eps,
                initializer_range=cfgs.initializer_range,
            )
        )

    def forward(self, spatial_ids, temporal_ids, segment_ids, attn_mask, output_attentions=False):
        inputs_embeds = self.spatial_embeddings(spatial_ids)
        return self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            token_type_ids=segment_ids,
            position_ids=temporal_ids,
            output_attentions=output_attentions,
        )


class ScenePatchEmbedding(nn.Module):
    def __init__(self, patch_size=16, embedding_dim=512, act_fn="relu"):
        super().__init__()
        self.linear1 = nn.Linear(patch_size * patch_size, embedding_dim)
        self.act_fn = _activation(act_fn)

    def forward(self, x):
        return self.act_fn(self.linear1(x.float()))


class SceneTrajectoryEncoder(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs
        self.scene_enabled = bool(getattr(cfgs, "scene", False))
        self.scene_token_count = int(getattr(cfgs, "num_patch", 0)) if self.scene_enabled else 0
        self.spatial_embeddings = SpatialEmbedding(cfgs.input_dim, cfgs.hidden_size, cfgs.act_fn)
        self.scene_patch_embeddings = (
            ScenePatchEmbedding(cfgs.patch_size, cfgs.hidden_size, cfgs.act_fn) if self.scene_enabled else None
        )
        self.backbone = TrajectoryBertBackbone(
            TrajectoryBertBackboneConfig(
                hidden_size=cfgs.hidden_size,
                num_hidden_layers=cfgs.num_layer,
                num_attention_heads=cfgs.num_head,
                intermediate_size=cfgs.intermediate_size,
                dropout_prob=cfgs.dropout_prob,
                attention_dropout_prob=cfgs.dropout_prob,
                max_position_embeddings=max(512, cfgs.obs_len + cfgs.pred_len + self.scene_token_count + 32),
                type_vocab_size=getattr(cfgs, "segment_vocab_size", cfgs.num_nbr + self.scene_token_count + 2),
                layer_norm_eps=cfgs.layer_norm_eps,
                initializer_range=cfgs.initializer_range,
            )
        )

    def _trajectory_embeds(self, spatial_ids):
        return self.spatial_embeddings(spatial_ids)

    def _scene_embeds(self, env_spatial_ids, env_temporal_ids, env_segment_ids, env_attn_mask):
        if env_spatial_ids is None:
            raise ValueError("env_spatial_ids is required for scene encoding")
        if env_temporal_ids is None or env_segment_ids is None or env_attn_mask is None:
            raise ValueError("env temporal, segment, and attention ids are required for scene encoding")

        scene_tokens = self.scene_patch_embeddings(env_spatial_ids)
        return (
            scene_tokens,
            env_segment_ids.long(),
            env_temporal_ids.long(),
            env_attn_mask.float(),
        )

    def forward(
        self,
        spatial_ids,
        temporal_ids,
        segment_ids,
        attn_mask,
        env_spatial_ids=None,
        env_temporal_ids=None,
        env_segment_ids=None,
        env_attn_mask=None,
        envs=None,
        output_attentions=False,
    ):
        inputs_embeds = self._trajectory_embeds(spatial_ids)
        token_type_ids = segment_ids
        position_ids = temporal_ids
        attention_mask = attn_mask
        scene_tokens = None

        if self.scene_enabled:
            scene_tokens, scene_segment_ids, scene_temporal_ids, scene_attn_mask = self._scene_embeds(
                env_spatial_ids,
                env_temporal_ids,
                env_segment_ids,
                env_attn_mask,
            )
            inputs_embeds = torch.cat([inputs_embeds, scene_tokens], dim=1)
            token_type_ids = torch.cat([token_type_ids, scene_segment_ids], dim=1)
            position_ids = torch.cat([position_ids, scene_temporal_ids], dim=1)
            attention_mask = torch.cat([attention_mask, scene_attn_mask], dim=1)

        enc_out = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
        )
        enc_out["scene_tokens"] = scene_tokens
        enc_out["scene_attentions"] = None
        return enc_out


_LAZY_EXPORTS = {
    "SBertConfig": ("sbert.model", "SBertConfig"),
    "SBertModel": ("sbert.model", "SBertModel"),
    "SBertPTModel": ("sbert.model", "SBertPTModel"),
    "SBertFTModel": ("sbert.model", "SBertFTModel"),
    "SBertPlusPTConfig": ("spubert.model", "SBertPlusPTConfig"),
    "SBertPlusTGPConfig": ("spubert.model", "SBertPlusTGPConfig"),
    "SBertPlusMGPConfig": ("spubert.model", "SBertPlusMGPConfig"),
    "SBertPlusFTConfig": ("spubert.model", "SBertPlusFTConfig"),
    "SBertPlusModel": ("spubert.model", "SBertPlusModel"),
    "SBertPlusPTModel": ("spubert.model", "SBertPlusPTModel"),
    "SBertPlusTGPModel": ("spubert.model", "SBertPlusTGPModel"),
    "SBertPlusMGPModel": ("spubert.model", "SBertPlusMGPModel"),
    "SBertPlusFTModel": ("spubert.model", "SBertPlusFTModel"),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "ModalEmbedding",
    "SpatialEmbedding",
    "SIPPooler",
    "MTPPooler",
    "SBertFutureTrajPooler",
    "PastTrajPooler",
    "FutureTrajPooler",
    "FullTrajPooler",
    "GoalPooler",
    "MTPDecoder",
    "SIPDecoder",
    "GoalDecoder",
    "SpatialDecoder",
    "GoalPriorNet",
    "GoalRecognitionNet",
    "GoalEncoder",
    "TrajectoryBertBackboneConfig",
    "TrajectoryBertBackbone",
    "TrajectoryEncoder",
    "SceneTrajectoryEncoder",
] + sorted(_LAZY_EXPORTS)

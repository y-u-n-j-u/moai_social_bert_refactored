from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor, dtype, nn

from src.model import MTPDecoder, MTPPooler, SBertFutureTrajPooler, SIPDecoder, SIPPooler, TrajectoryEncoder
from src.loss import ADELoss, FDELoss, MaskedADELoss


class SBertModelBase(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs
        self._skip_init_param_ids: set[int] = set()

    def init_weights(self):
        self._skip_init_param_ids = set()
        for module in self.modules():
            if getattr(module, '_skip_custom_init', False):
                for param in module.parameters():
                    self._skip_init_param_ids.add(id(param))
        self.apply(self._init_weights)
        self._skip_init_param_ids = set()

    def _init_weights(self, module):
        direct_params = list(module.parameters(recurse=False))
        if direct_params and any(id(param) in self._skip_init_param_ids for param in direct_params):
            return
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.cfgs.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @property
    def dtype(self) -> dtype:
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
                return [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype


class SBertConfig:
    def __init__(
        self,
        hidden_size=512,
        num_layer=4,
        num_head=4,
        act_fn='relu',
        dropout_prob=0.1,
        input_dim=2,
        output_dim=2,
        view_range=20.0,
        view_angle=1.0,
        social_range=2.0,
        obs_len=8,
        pred_len=12,
        num_nbr=5,
        pad_token_id=0,
        layer_norm_eps=1e-12,
        initializer_range=0.02,
        sip=False,
        backbone_type='bert',
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act_fn = act_fn
        self.dropout_prob = dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.hidden_size = hidden_size
        self.intermediate_size = hidden_size * 4
        self.num_layer = num_layer
        self.num_head = num_head
        self.view_range = view_range
        self.view_angle = view_angle
        self.social_range = social_range
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.num_nbr = num_nbr
        self.pad_token_id = pad_token_id
        self.chunk_size_feed_forward = 0
        self.initializer_range = initializer_range
        self.sip = sip
        self.backbone_type = backbone_type
        self.segment_vocab_size = num_nbr + 2


class SBertModel(SBertModelBase):
    def __init__(self, cfgs):
        super().__init__(cfgs)
        self.encoder = TrajectoryEncoder(cfgs)
        self.init_weights()

    def forward(self, spatial_ids, temporal_ids, segment_ids, attn_mask, output_attentions=False):
        return self.encoder(
            spatial_ids=spatial_ids,
            temporal_ids=temporal_ids,
            segment_ids=segment_ids,
            attn_mask=attn_mask,
            output_attentions=output_attentions,
        )


class SBertPTModel(SBertModelBase):
    def __init__(self, cfgs):
        super().__init__(cfgs)
        self.sbert = SBertModel(cfgs=cfgs)
        self.sip_pooler = SIPPooler(obs_len=cfgs.obs_len, pred_len=cfgs.pred_len, num_nbr=cfgs.num_nbr)
        self.sip_decoder = SIPDecoder(hidden_size=cfgs.hidden_size)
        self.mtp_pooler = MTPPooler(obs_len=cfgs.obs_len, pred_len=cfgs.pred_len, num_nbr=cfgs.num_nbr)
        self.mtp_decoder = MTPDecoder(
            hidden_size=cfgs.hidden_size,
            out_dim=cfgs.output_dim,
            layer_norm_eps=cfgs.layer_norm_eps,
            act_fn=cfgs.act_fn,
        )
        self.mtp_loss_fn = MaskedADELoss()
        self.sip_loss_fn = nn.NLLLoss(ignore_index=0, reduction="mean")
        self.init_weights()

    def forward(
        self,
        train,
        spatial_ids,
        temporal_ids,
        segment_ids,
        attn_mask,
        traj_mask=None,
        traj_lbl=None,
        near_lbl=None,
        output_attentions=False,
    ):
        enc_h = self.sbert(
            spatial_ids=spatial_ids,
            segment_ids=segment_ids,
            temporal_ids=temporal_ids,
            attn_mask=attn_mask,
            output_attentions=output_attentions,
        )
        mtp_h = self.mtp_pooler(enc_h['last_hidden_state'])
        sip_h = self.sip_pooler(enc_h['last_hidden_state'])
        mtp_out = self.mtp_decoder(mtp_h)
        sip_out = self.sip_decoder(sip_h)

        mtp_loss = self.mtp_loss_fn(mtp_out, traj_lbl, traj_mask)
        if self.cfgs.sip:
            sip_loss = self.sip_loss_fn(sip_out.view(-1, 3), near_lbl.view(-1))
            total_loss = mtp_loss + sip_loss
        else:
            sip_loss = None
            total_loss = mtp_loss

        return {
            'total_loss': total_loss,
            'mtp_loss': mtp_loss,
            'sip_loss': sip_loss,
            'mtp_output': mtp_out,
            'sip_output': sip_out,
            'attentions': enc_h['attentions'],
        }


class SBertFTModel(SBertModelBase):
    def __init__(self, cfgs):
        super().__init__(cfgs)
        self.sbert = SBertModel(cfgs=cfgs)
        self.traj_pooler = SBertFutureTrajPooler(obs_len=cfgs.obs_len, pred_len=cfgs.pred_len)
        self.sbert_decoder = MTPDecoder(
            hidden_size=cfgs.hidden_size,
            out_dim=cfgs.output_dim,
            layer_norm_eps=cfgs.layer_norm_eps,
            act_fn=cfgs.act_fn,
        )
        self.ade_loss_fn = ADELoss()
        self.fde_loss_fn = FDELoss()
        self.init_weights()

    def forward(
        self,
        spatial_ids,
        temporal_ids,
        segment_ids,
        attn_mask,
        traj_lbl=None,
        train=False,
        output_attentions=False,
    ):
        enc_h = self.sbert(
            spatial_ids=spatial_ids,
            segment_ids=segment_ids,
            temporal_ids=temporal_ids,
            attn_mask=attn_mask,
            output_attentions=output_attentions,
        )
        traj_h = self.traj_pooler(enc_h['last_hidden_state'])
        pred_traj = self.sbert_decoder(traj_h)
        if not train:
            return {'pred_traj': pred_traj, 'attentions': enc_h['attentions']}

        ade_loss = self.ade_loss_fn(pred_traj, traj_lbl)
        fde_loss = self.fde_loss_fn(pred_traj, traj_lbl)
        return {
            'total_loss': ade_loss,
            'ade_loss': ade_loss,
            'fde_loss': fde_loss,
            'pred_traj': pred_traj,
            'attentions': enc_h['attentions'],
        }

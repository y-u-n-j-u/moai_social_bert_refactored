from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor, device, dtype, nn

from src.model import (
    FutureTrajPooler,
    GoalDecoder,
    GoalEncoder,
    GoalPooler,
    GoalPriorNet,
    GoalRecognitionNet,
    MTPDecoder,
    MTPPooler,
    SBertFutureTrajPooler,
    SIPDecoder,
    SIPPooler,
    SpatialDecoder,
)
from src.loss import (
    ADELoss,
    FDELoss,
    MGPCVAELoss,
    MaskedADELoss,
    cal_idx_from_pos,
    goal_collision_loss,
    pos_collision_loss,
)
from src.utils import MultiKMeans
from src.model import SceneTrajectoryEncoder


class SBertModelBase(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs
        self._skip_init_param_ids: set[int] = set()

    def init_weights(self):
        self._skip_init_param_ids = set()
        for module in self.modules():
            if getattr(module, "_skip_custom_init", False):
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

    def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: Tuple[int], dev: device) -> Tensor:
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input ids {input_shape} or attention mask {attention_mask.shape}"
            )

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        return (1.0 - extended_attention_mask) * -10000.0



def _apply_runtime_fields(cfg, *, backbone_type="bert", binary_scene=False):
    cfg.backbone_type = backbone_type
    cfg.binary_scene = binary_scene
    cfg.segment_vocab_size = cfg.num_nbr + cfg.num_patch + 2


class SBertPlusPTConfig:
    def __init__(
        self,
        hidden_size=512,
        num_layer=4,
        num_head=4,
        act_fn="relu",
        dropout_prob=0.1,
        input_dim=2,
        output_dim=2,
        goal_dim=2,
        view_range=20.0,
        view_angle=1.0,
        social_range=2.0,
        obs_len=8,
        pred_len=12,
        num_nbr=5,
        scene=False,
        num_patch=32,
        patch_size=32,
        col_weight=10,
        traj_weight=1.0,
        pad_token_id=0,
        layer_norm_eps=1e-12,
        initializer_range=0.02,
        sip=False,
        backbone_type="bert",
        binary_scene=False,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.goal_dim = goal_dim
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
        self.col_weight = col_weight
        self.traj_weight = traj_weight
        self.scene = scene
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.pad_token_id = pad_token_id
        self.chunk_size_feed_forward = 0
        self.initializer_range = initializer_range
        self.sip = sip
        _apply_runtime_fields(self, backbone_type=backbone_type, binary_scene=binary_scene)


class SBertPlusTGPConfig:
    def __init__(
        self,
        hidden_size=512,
        num_layer=4,
        num_head=4,
        act_fn="relu",
        dropout_prob=0.1,
        input_dim=2,
        output_dim=2,
        goal_dim=2,
        view_range=20.0,
        view_angle=1.0,
        social_range=2.0,
        obs_len=8,
        pred_len=12,
        num_nbr=5,
        scene=False,
        num_patch=32,
        patch_size=32,
        col_weight=10,
        traj_weight=1.0,
        pad_token_id=0,
        layer_norm_eps=1e-12,
        initializer_range=0.02,
        backbone_type="bert",
        binary_scene=False,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.goal_dim = goal_dim
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
        self.col_weight = col_weight
        self.traj_weight = traj_weight
        self.scene = scene
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.pad_token_id = pad_token_id
        self.chunk_size_feed_forward = 0
        self.initializer_range = initializer_range
        _apply_runtime_fields(self, backbone_type=backbone_type, binary_scene=binary_scene)


class SBertPlusMGPConfig:
    def __init__(
        self,
        hidden_size=512,
        num_layer=4,
        num_head=4,
        act_fn="relu",
        dropout_prob=0.1,
        input_dim=2,
        output_dim=2,
        goal_dim=2,
        view_range=20.0,
        view_angle=1.0,
        social_range=2.0,
        obs_len=8,
        pred_len=12,
        num_nbr=5,
        scene=False,
        num_patch=32,
        patch_size=32,
        pad_token_id=0,
        layer_norm_eps=1e-12,
        initializer_range=0.02,
        k_sample=20,
        goal_hidden_size=64,
        goal_latent_size=64,
        kld_weight=10,
        col_weight=10,
        goal_weight=1.0,
        cvae_sigma=1.0,
        kld_clamp=None,
        share=False,
        normal=False,
        backbone_type="bert",
        binary_scene=False,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.goal_dim = goal_dim
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
        self.scene = scene
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.pad_token_id = pad_token_id
        self.chunk_size_feed_forward = 0
        self.initializer_range = initializer_range
        self.k_sample = k_sample
        self.goal_hidden_size = goal_hidden_size
        self.goal_latent_size = goal_latent_size
        self.kld_weight = kld_weight
        self.col_weight = col_weight
        self.goal_weight = goal_weight
        self.share = share
        self.cvae_sigma = cvae_sigma
        self.kld_clamp = kld_clamp
        self.normal = normal
        _apply_runtime_fields(self, backbone_type=backbone_type, binary_scene=binary_scene)


class SBertPlusFTConfig:
    def __init__(self, traj_cfgs, goal_cfgs, share=False):
        self.input_dim = traj_cfgs.input_dim
        self.output_dim = traj_cfgs.output_dim
        self.goal_dim = goal_cfgs.goal_dim
        self.act_fn = traj_cfgs.act_fn
        self.dropout_prob = traj_cfgs.dropout_prob
        self.layer_norm_eps = traj_cfgs.layer_norm_eps
        self.hidden_size = traj_cfgs.hidden_size
        self.intermediate_size = traj_cfgs.intermediate_size
        self.view_range = traj_cfgs.view_range
        self.view_angle = traj_cfgs.view_angle
        self.social_range = traj_cfgs.social_range
        self.obs_len = traj_cfgs.obs_len
        self.pred_len = traj_cfgs.pred_len
        self.num_nbr = traj_cfgs.num_nbr
        self.scene = traj_cfgs.scene
        self.num_patch = traj_cfgs.num_patch
        self.patch_size = traj_cfgs.patch_size
        self.pad_token_id = traj_cfgs.pad_token_id
        self.chunk_size_feed_forward = traj_cfgs.chunk_size_feed_forward
        self.initializer_range = traj_cfgs.initializer_range
        self.col_weight = traj_cfgs.col_weight
        self.k_sample = goal_cfgs.k_sample
        self.goal_hidden_size = goal_cfgs.goal_hidden_size
        self.goal_latent_size = goal_cfgs.goal_latent_size
        self.kld_weight = goal_cfgs.kld_weight
        self.cvae_sigma = goal_cfgs.cvae_sigma
        self.kld_clamp = goal_cfgs.kld_clamp
        self.traj_weight = traj_cfgs.traj_weight
        self.goal_weight = goal_cfgs.goal_weight
        self.num_traj_layer = traj_cfgs.num_layer
        self.num_traj_head = traj_cfgs.num_head
        self.num_goal_layer = goal_cfgs.num_layer
        self.num_goal_head = goal_cfgs.num_head
        self.normal = goal_cfgs.normal
        self.share = share
        self.backbone_type = traj_cfgs.backbone_type
        self.binary_scene = traj_cfgs.binary_scene
        self.segment_vocab_size = traj_cfgs.segment_vocab_size
        if share:
            self.num_layer = self.num_traj_layer
            self.num_head = self.num_traj_head



class SBertPlusModel(SBertModelBase):
    def __init__(self, cfgs):
        super().__init__(cfgs)
        self.hf_encoder = SceneTrajectoryEncoder(cfgs)
        self.init_weights()

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
        return self.hf_encoder(
            spatial_ids=spatial_ids,
            temporal_ids=temporal_ids,
            segment_ids=segment_ids,
            attn_mask=attn_mask,
            env_spatial_ids=env_spatial_ids,
            env_temporal_ids=env_temporal_ids,
            env_segment_ids=env_segment_ids,
            env_attn_mask=env_attn_mask,
            envs=envs,
            output_attentions=output_attentions,
        )


class SBertPlusPTModel(SBertModelBase):

    def __init__(self, cfgs):
        super().__init__(cfgs)
        self.sbert = SBertPlusModel(cfgs=cfgs)
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
        spatial_ids,
        temporal_ids,
        segment_ids,
        attn_mask,
        traj_lbl=None,
        goal_lbl=None,
        traj_mask=None,
        near_lbl=None,
        env_spatial_ids=None,
        env_temporal_ids=None,
        env_segment_ids=None,
        env_attn_mask=None,
        envs=None,
        envs_params=None,
        output_attentions=False,
    ):
        enc_h = self.sbert(
            spatial_ids=spatial_ids,
            segment_ids=segment_ids,
            temporal_ids=temporal_ids,
            attn_mask=attn_mask,
            env_spatial_ids=env_spatial_ids,
            env_temporal_ids=env_temporal_ids,
            env_segment_ids=env_segment_ids,
            env_attn_mask=env_attn_mask,
            envs=envs,
            output_attentions=output_attentions,
        )
        mtp_h = self.mtp_pooler(enc_h["last_hidden_state"])
        sip_h = self.sip_pooler(enc_h["last_hidden_state"])
        mtp_out = self.mtp_decoder(mtp_h)
        sip_out = self.sip_decoder(sip_h)

        mtp_loss = self.mtp_loss_fn(mtp_out, traj_lbl, traj_mask)
        total_loss = mtp_loss
        if self.cfgs.sip:
            sip_loss = self.sip_loss_fn(sip_out.view(-1, 3), near_lbl.view(-1))
            total_loss = total_loss + sip_loss
        else:
            sip_loss = None

        return {
            "total_loss": total_loss,
            "mtp_loss": mtp_loss,
            "sip_loss": sip_loss,
            "mtp_output": mtp_out,
            "sip_output": sip_out,
            "attentions": enc_h["attentions"],
        }

# trajectory 예측 -> ade loss
class SBertPlusTGPModel(SBertModelBase):
    def __init__(self, cfgs):
        super().__init__(cfgs)
        self.sbert = SBertPlusModel(cfgs=cfgs)
        self.traj_pooler = FutureTrajPooler(
            hidden_size=cfgs.hidden_size,
            obs_len=cfgs.obs_len,
            pred_len=cfgs.pred_len,
        )
        self.sbert_decoder = SpatialDecoder(
            cfgs.hidden_size,
            cfgs.output_dim,
            layer_norm_eps=cfgs.layer_norm_eps,
            act_fn=cfgs.act_fn,
        )
        self.ade_loss_fn = ADELoss()
        self.fde_loss_fn = FDELoss()
        self.init_weights()

    def inference(
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
        enc_h = self.sbert(
            spatial_ids=spatial_ids,
            segment_ids=segment_ids,
            temporal_ids=temporal_ids,
            attn_mask=attn_mask,
            env_spatial_ids=env_spatial_ids,
            env_temporal_ids=env_temporal_ids,
            env_segment_ids=env_segment_ids,
            env_attn_mask=env_attn_mask,
            envs=envs,
            output_attentions=output_attentions,
        )
        traj_h = self.traj_pooler(enc_h["last_hidden_state"])
        pred_trajs = self.sbert_decoder(traj_h)
        return {"pred_trajs": pred_trajs, "attentions": enc_h["attentions"]}

    def forward(
        self,
        spatial_ids,
        temporal_ids,
        segment_ids,
        attn_mask,
        traj_lbl=None,
        goal_lbl=None,
        env_spatial_ids=None,
        env_temporal_ids=None,
        env_segment_ids=None,
        env_attn_mask=None,
        envs=None,
        envs_params=None,
        output_attentions=False,
    ):
        outputs = self.inference(
            spatial_ids=spatial_ids,
            temporal_ids=temporal_ids,
            segment_ids=segment_ids,
            attn_mask=attn_mask,
            env_spatial_ids=env_spatial_ids,
            env_temporal_ids=env_temporal_ids,
            env_segment_ids=env_segment_ids,
            env_attn_mask=env_attn_mask,
            envs=envs,
            output_attentions=output_attentions,
        )
        pred_trajs = outputs["pred_trajs"]
        ade_loss = self.ade_loss_fn(pred_trajs, traj_lbl) * self.cfgs.traj_weight
        fde_loss = self.fde_loss_fn(pred_trajs[:, -1], goal_lbl) * self.cfgs.traj_weight
        total_loss = ade_loss
        if self.cfgs.scene and self.cfgs.col_weight > 0:
            col_loss = self.cfgs.col_weight * pos_collision_loss(pred_trajs, envs, envs_params)
            total_loss = total_loss + col_loss
        else:
            col_loss = None

        return {
            "total_loss": total_loss,
            "ade_loss": ade_loss,
            "fde_loss": fde_loss,
            "col_loss": col_loss,
            "pred_trajs": pred_trajs,
            "attentions": outputs["attentions"],
        }

# goal 예측(CVAE) -> kld_loss + gde_loss
# kld_loss: CVAE의 KL divergence
# gde_loss
# col_loss: 충돌 loss
class SBertPlusMGPModel(SBertModelBase):
    def __init__(self, goal_cfgs, share_enc=None):
        super().__init__(goal_cfgs)
        self.sbert = share_enc if share_enc is not None else SBertPlusModel(goal_cfgs)
        self.enc_hidden_pooler = GoalPooler(
            hidden_size=goal_cfgs.hidden_size,
            obs_len=goal_cfgs.obs_len,
            pred_len=goal_cfgs.pred_len,
            act_fn=goal_cfgs.act_fn,
        )
        self.gt_goal_encoder = GoalEncoder(
            goal_cfgs.goal_dim,
            goal_cfgs.goal_hidden_size,
            layer_norm_eps=goal_cfgs.layer_norm_eps,
            dropout_prob=goal_cfgs.dropout_prob,
            act_fn=goal_cfgs.act_fn,
        )
        self.goal_recog_net = GoalRecognitionNet(
            enc_hidden_size=goal_cfgs.hidden_size,
            goal_hidden_size=goal_cfgs.goal_hidden_size,
            goal_latent_size=goal_cfgs.goal_latent_size,
            act_fn=goal_cfgs.act_fn,
        )
        self.goal_prior_net = GoalPriorNet(
            enc_hidden_size=goal_cfgs.hidden_size,
            goal_latent_size=goal_cfgs.goal_latent_size,
            act_fn=goal_cfgs.act_fn,
        )
        self.sbert_decoder = GoalDecoder(
            goal_cfgs.hidden_size,
            goal_cfgs.goal_latent_size,
            goal_cfgs.goal_dim,
            act_fn=goal_cfgs.act_fn,
        )
        self.gde_loss_fn = MGPCVAELoss()
        self.init_weights()

    def goal_predictor(self, pred_goal_h, k_sample, d_sample=0):
        if d_sample < k_sample:
            d_sample = k_sample

        if self.cfgs.normal:
            p_goal_mu = torch.zeros(pred_goal_h.size(0), d_sample, self.cfgs.goal_latent_size).to(pred_goal_h.device)
            p_goal_std = torch.ones(pred_goal_h.size(0), d_sample, self.cfgs.goal_latent_size).mul(self.cfgs.cvae_sigma).to(pred_goal_h.device)
            k_pred_goal_h = pred_goal_h.unsqueeze(1).repeat(1, d_sample, 1)
            eps = torch.randn_like(p_goal_std)
            k_goal_latent = eps.mul(p_goal_std).add(p_goal_mu)
        else:
            p_goal_out = self.goal_prior_net(pred_goal_h)
            p_goal_mu = p_goal_out[:, : self.cfgs.goal_latent_size]
            p_goal_logvar = p_goal_out[:, self.cfgs.goal_latent_size :]
            p_goal_std = p_goal_logvar.mul(0.5).exp()
            p_goal_mu = p_goal_mu.unsqueeze(1).repeat(1, d_sample, 1)
            p_goal_std = p_goal_std.unsqueeze(1).repeat(1, d_sample, 1)
            eps = torch.randn_like(p_goal_std)
            k_goal_latent = eps.mul(p_goal_std).add(p_goal_mu)
            k_pred_goal_h = pred_goal_h.unsqueeze(1).repeat(1, d_sample, 1)

        k_pred_goal_h = torch.cat([k_goal_latent, k_pred_goal_h], dim=-1)
        bk_pred_goal_h = k_pred_goal_h.reshape(-1, self.cfgs.goal_latent_size + self.cfgs.hidden_size)
        bk_pred_goals = self.sbert_decoder(bk_pred_goal_h)
        pred_goals = bk_pred_goals.reshape(len(pred_goal_h), d_sample, self.cfgs.goal_dim)

        if d_sample > k_sample:
            pred_goals = MultiKMeans(n_clusters=k_sample, n_kmeans=pred_goals.shape[0], max_iter=10, verbose=False).fit_predict(pred_goals)

        return pred_goals

    def goal_trainer(self, pred_goal_h, goal_lbl, k_sample):
        gt_goal_h = self.gt_goal_encoder(goal_lbl)
        r_goal_out = self.goal_recog_net(torch.cat([pred_goal_h, gt_goal_h], dim=-1))
        r_goal_mu = r_goal_out[:, : self.cfgs.goal_latent_size]
        r_goal_logvar = r_goal_out[:, self.cfgs.goal_latent_size :]
        alpha = 1.0

        if self.cfgs.normal:
            kld_loss = alpha * (-0.5 * torch.sum(1 + r_goal_logvar - r_goal_logvar.exp() - r_goal_mu.pow(2), dim=1).mean())
            r_goal_std = r_goal_logvar.mul(0.5).exp()
            k_r_goal_mu = r_goal_mu.unsqueeze(1).repeat(1, k_sample, 1)
            k_r_goal_std = r_goal_std.unsqueeze(1).repeat(1, k_sample, 1)
            k_pred_goal_h = pred_goal_h.unsqueeze(1).repeat(1, k_sample, 1)
            eps = torch.randn_like(k_r_goal_std)
            k_goal_latent = eps.mul(k_r_goal_std).add(k_r_goal_mu)
        else:
            p_goal_out = self.goal_prior_net(pred_goal_h)
            p_goal_mu = p_goal_out[:, : self.cfgs.goal_latent_size]
            p_goal_logvar = p_goal_out[:, self.cfgs.goal_latent_size :]
            kld_loss = alpha * (
                0.5
                * torch.sum(
                    (r_goal_logvar.exp() / p_goal_logvar.exp())
                    + (p_goal_mu - r_goal_mu).pow(2) / p_goal_logvar.exp()
                    - 1
                    + (p_goal_logvar - r_goal_logvar),
                    dim=-1,
                ).mean()
            )
            r_goal_std = r_goal_logvar.mul(0.5).exp()
            k_r_goal_mu = r_goal_mu.unsqueeze(1).repeat(1, k_sample, 1)
            k_r_goal_std = r_goal_std.unsqueeze(1).repeat(1, k_sample, 1)
            k_pred_goal_h = pred_goal_h.unsqueeze(1).repeat(1, k_sample, 1)
            eps = torch.randn_like(k_r_goal_std)
            k_goal_latent = eps.mul(k_r_goal_std).add(k_r_goal_mu)

        k_pred_goal_h = torch.cat([k_goal_latent, k_pred_goal_h], dim=-1)
        bk_pred_goal_h = k_pred_goal_h.reshape(-1, self.cfgs.goal_latent_size + self.cfgs.hidden_size)
        bk_pred_goals = self.sbert_decoder(bk_pred_goal_h)
        pred_goals = bk_pred_goals.reshape(len(pred_goal_h), k_sample, self.cfgs.goal_dim)
        return pred_goals, kld_loss

    def inference(
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
        d_sample=0,
    ):
        goal_enc_out = self.sbert(
            spatial_ids=spatial_ids,
            segment_ids=segment_ids,
            temporal_ids=temporal_ids,
            attn_mask=attn_mask,
            env_spatial_ids=env_spatial_ids,
            env_temporal_ids=env_temporal_ids,
            env_segment_ids=env_segment_ids,
            env_attn_mask=env_attn_mask,
            envs=envs,
            output_attentions=output_attentions,
        )
        goal_h = self.enc_hidden_pooler(goal_enc_out["last_hidden_state"])
        pred_goals = self.goal_predictor(goal_h, k_sample=self.cfgs.k_sample, d_sample=d_sample)
        return {"pred_goals": pred_goals, "attentions": goal_enc_out["attentions"]}

    def forward(
        self,
        spatial_ids,
        temporal_ids,
        segment_ids,
        attn_mask,
        traj_lbl=None,
        goal_lbl=None,
        env_spatial_ids=None,
        env_temporal_ids=None,
        env_segment_ids=None,
        env_attn_mask=None,
        envs=None,
        envs_params=None,
        output_attentions=False,
        kld_weight=None,
    ):
        goal_enc_out = self.sbert(
            spatial_ids=spatial_ids,
            segment_ids=segment_ids,
            temporal_ids=temporal_ids,
            attn_mask=attn_mask,
            env_spatial_ids=env_spatial_ids,
            env_temporal_ids=env_temporal_ids,
            env_segment_ids=env_segment_ids,
            env_attn_mask=env_attn_mask,
            envs=envs,
            output_attentions=output_attentions,
        )
        goal_h = self.enc_hidden_pooler(goal_enc_out["last_hidden_state"])
        pred_goals, kld_loss = self.goal_trainer(goal_h, goal_lbl, k_sample=self.cfgs.k_sample)
        gde_loss, best_idx = self.gde_loss_fn(pred_goals, goal_lbl, self.cfgs.k_sample, output_dim=self.cfgs.output_dim, best=True)
        kld_loss = kld_weight * kld_loss
        gde_loss = self.cfgs.goal_weight * gde_loss
        total_loss = kld_loss + gde_loss
        if self.cfgs.scene and self.cfgs.col_weight > 0:
            col_loss = self.cfgs.col_weight * goal_collision_loss(pred_goals, envs, envs_params)
            total_loss = total_loss + col_loss
        else:
            col_loss = None

        return {
            "total_loss": total_loss,
            "gde_loss": gde_loss,
            "col_loss": col_loss,
            "kld_loss": kld_loss,
            "pred_goals": pred_goals,
            "best_idx": best_idx,
            "attentions": goal_enc_out["attentions"],
        }


class SBertPlusFTModel(SBertModelBase):
    def __init__(self, tgp_cfgs, mgp_cfgs, cfgs):
        super().__init__(cfgs)
        if cfgs.share:
            self.tgp_model = SBertPlusTGPModel(cfgs)
            self.mgp_model = SBertPlusMGPModel(cfgs, share_enc=self.tgp_model.sbert)
        else:
            self.tgp_model = SBertPlusTGPModel(tgp_cfgs)
            self.mgp_model = SBertPlusMGPModel(mgp_cfgs)
        self.init_weights()

    # ── k개 복제용 (MGP 흐름) ──────────────────────────────────────────────
    # MGP가 예측한 k개 goal을 spatial_ids에 삽입, TGP 입력용 준비
    def add_goals(self, spatial_ids, goals, mask_val, pad_val):
        spatial_ids = spatial_ids.unsqueeze(1).repeat(1, self.cfgs.k_sample, 1, 1)
        spatial_ids[:, :, 1 + self.cfgs.obs_len : 1 + self.cfgs.obs_len + self.cfgs.pred_len, :] = mask_val
        spatial_ids[:, :, 1 + self.cfgs.obs_len + self.cfgs.pred_len, : self.cfgs.goal_dim] = goals
        spatial_ids[:, :, 1 + self.cfgs.obs_len + self.cfgs.pred_len, self.cfgs.goal_dim :] = pad_val
        return spatial_ids

    # ── goal 1개용 (GT goal 흐름) ──────────────────────────────────────────
    # GT goal 하나를 spatial_ids에 삽입 (k배 복제 없음)
    # 학습 때 dataset.py가 만드는 tgp_spatial_ids와 동일한 형태
    def add_single_goal(self, spatial_ids, goal, mask_val, pad_val):
        # spatial_ids: (batch, seq_len, input_dim) → 그대로 유지 (복제 없음)
        spatial_ids = spatial_ids.clone()
        # 중간 pred 구간 [MSK]로 채움 (1+obs_len ~ obs_len+pred_len)
        spatial_ids[:, 1 + self.cfgs.obs_len : 1 + self.cfgs.obs_len + self.cfgs.pred_len, :] = mask_val
        spatial_ids[:, 1 + self.cfgs.obs_len + self.cfgs.pred_len, : self.cfgs.goal_dim] = goal
        spatial_ids[:, 1 + self.cfgs.obs_len + self.cfgs.pred_len, self.cfgs.goal_dim :] = pad_val
        return spatial_ids

    def inference(
        self,
        mgp_spatial_ids,
        mgp_temporal_ids,
        mgp_segment_ids,
        mgp_attn_mask,
        tgp_temporal_ids,
        tgp_segment_ids,
        tgp_attn_mask,
        env_spatial_ids=None,
        env_temporal_ids=None,
        env_segment_ids=None,
        env_attn_mask=None,
        envs=None,
        output_attentions=False,
        d_sample=0,
    ):
        traj_batch_size, traj_seq_len, traj_spatial_dim = mgp_spatial_ids.size()

        # MGP prior에서 goal k개 샘플링
        mgp_out = self.mgp_model.inference(
            spatial_ids=mgp_spatial_ids,
            segment_ids=mgp_segment_ids,
            temporal_ids=mgp_temporal_ids,
            attn_mask=mgp_attn_mask,
            env_spatial_ids=env_spatial_ids,
            env_temporal_ids=env_temporal_ids,
            env_segment_ids=env_segment_ids,
            env_attn_mask=env_attn_mask,
            envs=envs,
            output_attentions=output_attentions,
            d_sample=d_sample,
        )
        pred_goals = mgp_out["pred_goals"]       # (batch, k, goal_dim)

        k_goal_spatial_ids = self.add_goals(
            mgp_spatial_ids,
            pred_goals,
            mask_val=self.cfgs.view_range,
            pad_val=-self.cfgs.view_range,
        ).view(-1, traj_seq_len, self.cfgs.input_dim)
        k_goal_segment_ids = tgp_segment_ids.unsqueeze(1).repeat(1, self.cfgs.k_sample, 1).view(-1, traj_seq_len)
        k_goal_temporal_ids = tgp_temporal_ids.unsqueeze(1).repeat(1, self.cfgs.k_sample, 1).view(-1, traj_seq_len)
        k_goal_attn_mask = tgp_attn_mask.unsqueeze(1).repeat(1, self.cfgs.k_sample, 1).view(-1, traj_seq_len)

        if self.cfgs.scene:
            _, env_seq_len, env_spatial_dim = env_spatial_ids.size()
            k_goal_env_spatial_ids = env_spatial_ids.unsqueeze(1).repeat(1, self.cfgs.k_sample, 1, 1).view(-1, env_seq_len, env_spatial_dim)
            k_goal_env_segment_ids = env_segment_ids.unsqueeze(1).repeat(1, self.cfgs.k_sample, 1).view(-1, env_seq_len)
            k_goal_env_temporal_ids = env_temporal_ids.unsqueeze(1).repeat(1, self.cfgs.k_sample, 1).view(-1, env_seq_len)
            k_goal_env_attn_mask = env_attn_mask.unsqueeze(1).repeat(1, self.cfgs.k_sample, 1).view(-1, env_seq_len)
        else:
            k_goal_env_spatial_ids = None
            k_goal_env_segment_ids = None
            k_goal_env_temporal_ids = None
            k_goal_env_attn_mask = None

        tgp_out = self.tgp_model.inference(
            spatial_ids=k_goal_spatial_ids,
            segment_ids=k_goal_segment_ids,
            temporal_ids=k_goal_temporal_ids,
            attn_mask=k_goal_attn_mask,
            env_spatial_ids=k_goal_env_spatial_ids,
            env_segment_ids=k_goal_env_segment_ids,
            env_temporal_ids=k_goal_env_temporal_ids,
            env_attn_mask=k_goal_env_attn_mask,
            envs=None if envs is None else envs.unsqueeze(1).repeat(1, self.cfgs.k_sample, 1, 1).view(-1, envs.size(-2), envs.size(-1)),
            output_attentions=output_attentions,
        )

        pred_trajs = tgp_out["pred_trajs"].reshape(traj_batch_size, self.cfgs.k_sample, self.cfgs.pred_len, self.cfgs.output_dim)
        pred_goals = pred_goals.reshape(traj_batch_size, self.cfgs.k_sample, self.cfgs.goal_dim)
        return {
            "pred_trajs": pred_trajs,          # (batch, k, pred_len, 2)
            "pred_goals": pred_goals,          # (batch, k, 2)
            "goal_attentions": mgp_out["attentions"],
            "traj_attentions": tgp_out["attentions"],
        }

    # ── 새 함수: GT goal → TGP 단독 호출 → trajectory 1개 ─────────────────
    # 학습 때 TGP가 본 입력 형태와 완전히 동일
    # 실 환경에서 GPS goal이 확정되었을 때 사용
    def inference_with_gt_goal(
        self,
        mgp_spatial_ids,       # obs + 이웃 시퀀스 (add_single_goal에서 goal 삽입)
        tgp_temporal_ids,
        tgp_segment_ids,
        tgp_attn_mask,
        gt_goals,              # (batch, goal_dim) — local 좌표 GT goal
        env_spatial_ids=None,
        env_temporal_ids=None,
        env_segment_ids=None,
        env_attn_mask=None,
        envs=None,
        output_attentions=False,
    ):
        # GT goal 하나를 시퀀스에 삽입 (k배 복제 없음)
        # 학습 때 tgp_spatial_ids = [SOT]+obs×8+[MSK]×11+goal_lbl 과 동일한 형태
        goal_spatial_ids = self.add_single_goal(
            mgp_spatial_ids,
            gt_goals,
            mask_val=self.cfgs.view_range,
            pad_val=-self.cfgs.view_range,
        )  # (batch, seq_len, input_dim)

        if self.cfgs.scene:
            tgp_env_spatial_ids  = env_spatial_ids
            tgp_env_segment_ids  = env_segment_ids
            tgp_env_temporal_ids = env_temporal_ids
            tgp_env_attn_mask    = env_attn_mask
        else:
            tgp_env_spatial_ids  = None
            tgp_env_segment_ids  = None
            tgp_env_temporal_ids = None
            tgp_env_attn_mask    = None

        # TGP 단독 호출 — trajectory 1개만 생성 (MGP 호출 없음)
        tgp_out = self.tgp_model.inference(
            spatial_ids=goal_spatial_ids,
            segment_ids=tgp_segment_ids,
            temporal_ids=tgp_temporal_ids,
            attn_mask=tgp_attn_mask,
            env_spatial_ids=tgp_env_spatial_ids,
            env_segment_ids=tgp_env_segment_ids,
            env_temporal_ids=tgp_env_temporal_ids,
            env_attn_mask=tgp_env_attn_mask,
            envs=envs,
            output_attentions=output_attentions,
        )

        return {
            "pred_trajs": tgp_out["pred_trajs"],   # (batch, pred_len, 2)
            "pred_goals": gt_goals,                # (batch, 2)
            "traj_attentions": tgp_out["attentions"],
        }

    def inference_with_goal_sampling(
        self,
        mgp_spatial_ids,
        tgp_temporal_ids,
        tgp_segment_ids,
        tgp_attn_mask,
        primary_goal,           # (batch, goal_dim) — external module이 준 기준 goal
        k: int = 10,
        sigma: float = 1.0,
        env_spatial_ids=None,
        env_temporal_ids=None,
        env_segment_ids=None,
        env_attn_mask=None,
        envs=None,              # (batch, H, W) 점유 격자
        envs_params=None,       # (batch, 6): [min_x, min_y, W, H, res, thresh]
        output_attentions=False,
    ):
        batch_size = primary_goal.shape[0]
        device = primary_goal.device

        # 1. Gaussian sampling: primary_goal 주변에서 k개 후보 goal 샘플링
        noise = torch.randn(batch_size, k, primary_goal.shape[-1], device=device) * sigma
        sampled_goals = primary_goal.unsqueeze(1).expand(-1, k, -1) + noise  # (batch, k, goal_dim)

        best_trajs = []
        best_goals_out = []

        for b in range(batch_size):
            # 2. 장애물 점유 필터: sampled_goals 중 free space에 있는 것만 유지
            if envs is not None and envs_params is not None:
                valid_indices = []
                ep = envs_params[b]  # (6,)
                grid = envs[b]       # (H, W)
                thresh = ep[5]
                for i in range(k):
                    g = sampled_goals[b, i]  # (goal_dim,)
                    x_id, x_v = cal_idx_from_pos(g[0:1], ep[0], ep[2], ep[4])
                    y_id, y_v = cal_idx_from_pos(g[1:2], ep[1], ep[3], ep[4])
                    if x_v.item() and y_v.item():
                        if grid[y_id.item(), x_id.item()] <= thresh:
                            valid_indices.append(i)
                    else:
                        valid_indices.append(i)  # 맵 범위 밖 → 일단 유지
                if not valid_indices:
                    valid_indices = []  # 모두 장애물이면 primary_goal만 사용
            else:
                valid_indices = list(range(k))

            # primary_goal도 후보에 포함 (항상 첫 번째)
            goal_candidates = (
                [primary_goal[b:b+1]]
                + [sampled_goals[b, i:i+1] for i in valid_indices]
            )

            # 단일 배치 슬라이스
            obs_b  = mgp_spatial_ids[b:b+1]
            temp_b = tgp_temporal_ids[b:b+1]
            seg_b  = tgp_segment_ids[b:b+1]
            mask_b = tgp_attn_mask[b:b+1]

            env_kw = {}
            if envs is not None:
                env_kw = dict(
                    env_spatial_ids=env_spatial_ids[b:b+1],
                    env_temporal_ids=env_temporal_ids[b:b+1],
                    env_segment_ids=env_segment_ids[b:b+1],
                    env_attn_mask=env_attn_mask[b:b+1],
                    envs=envs[b:b+1],
                )

            # 3. 각 후보 goal에 대해 TGP 실행 + collision rate 계산
            pg = primary_goal[b]  # (goal_dim,) — 선택 기준 기준점
            candidates = []
            for goal_c in goal_candidates:
                out = self.inference_with_gt_goal(
                    obs_b, temp_b, seg_b, mask_b,
                    goal_c, **env_kw,
                    output_attentions=output_attentions,
                )
                traj = out["pred_trajs"]  # (1, pred_len, 2)

                col = 0.0
                if envs is not None and envs_params is not None:
                    col_val = pos_collision_loss(traj, envs[b:b+1], envs_params[b:b+1])
                    if not torch.isnan(col_val):
                        col = col_val.item()

                # primary_goal까지의 goal 거리 (선택 tie-break용)
                dist = torch.norm(goal_c[0] - pg).item()
                candidates.append((col, dist, traj, goal_c))

            # 4. 선택 전략
            #    - 충돌 없는(col=0) 후보 존재 → 그 중 primary_goal과 가장 가까운 goal 선택
            #    - 모두 충돌 → collision rate 최소인 것 선택
            collision_free = [(col, dist, traj, gc) for col, dist, traj, gc in candidates if col == 0.0]
            if collision_free:
                collision_free.sort(key=lambda x: x[1])   # dist 기준 정렬
                _, _, best_traj, best_goal = collision_free[0]
            else:
                candidates.sort(key=lambda x: x[0])       # col 기준 정렬
                _, _, best_traj, best_goal = candidates[0]
            best_trajs.append(best_traj)
            best_goals_out.append(best_goal)

        return {
            "pred_trajs": torch.cat(best_trajs, dim=0),      # (batch, pred_len, 2)
            "pred_goals": torch.cat(best_goals_out, dim=0),  # (batch, goal_dim)
            "sampled_goals": sampled_goals,                   # (batch, k, goal_dim) — 시각화용
        }

    def forward(
        self,
        mgp_spatial_ids,
        mgp_temporal_ids,
        mgp_segment_ids,
        mgp_attn_mask,
        tgp_spatial_ids,
        tgp_temporal_ids,
        tgp_segment_ids,
        tgp_attn_mask,
        traj_lbl=None,
        goal_lbl=None,
        env_spatial_ids=None,
        env_temporal_ids=None,
        env_segment_ids=None,
        env_attn_mask=None,
        envs=None,
        envs_params=None,
        output_attentions=False,
        kld_weight=1.0,
        traj_weight=1.0,
        goal_weight=1.0,
    ):
        mgp_out = self.mgp_model(
            spatial_ids=mgp_spatial_ids,
            segment_ids=mgp_segment_ids,
            temporal_ids=mgp_temporal_ids,
            attn_mask=mgp_attn_mask,
            env_spatial_ids=env_spatial_ids,
            env_temporal_ids=env_temporal_ids,
            env_segment_ids=env_segment_ids,
            env_attn_mask=env_attn_mask,
            envs=envs,
            envs_params=envs_params,
            traj_lbl=traj_lbl,
            goal_lbl=goal_lbl,
            output_attentions=output_attentions,
            kld_weight=kld_weight,
        )
        tgp_out = self.tgp_model(
            spatial_ids=tgp_spatial_ids,
            segment_ids=tgp_segment_ids,
            temporal_ids=tgp_temporal_ids,
            attn_mask=tgp_attn_mask,
            env_spatial_ids=env_spatial_ids,
            env_segment_ids=env_segment_ids,
            env_temporal_ids=env_temporal_ids,
            env_attn_mask=env_attn_mask,
            envs=envs,
            envs_params=envs_params,
            traj_lbl=traj_lbl,
            goal_lbl=goal_lbl,
            output_attentions=output_attentions,
        )
        return {
            "mgp_loss": mgp_out["total_loss"],
            "tgp_loss": tgp_out["total_loss"],
            "ade_loss": tgp_out["ade_loss"],
            "fde_loss": tgp_out["fde_loss"],
            "gde_loss": mgp_out["gde_loss"],
            "tgp_col_loss": tgp_out["col_loss"],
            "mgp_col_loss": mgp_out["col_loss"],
            "kld_loss": mgp_out["kld_loss"],
            "pred_trajs": tgp_out["pred_trajs"],
            "pred_goals": mgp_out["pred_goals"],
            "goal_attentions": mgp_out["attentions"],
            "traj_attentions": tgp_out["attentions"],
        }

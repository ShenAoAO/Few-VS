# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.data import Dictionary
from unicore.models import (BaseUnicoreModel, register_model,
                            register_model_architecture)
from unicore.modules import LayerNorm
import unicore

from .transformer_encoder_with_pair import TransformerEncoderWithPair
from .unimol import NonLinearHead, UniMolModel, base_architecture
from torch.nn import Conv2d, Dropout
import math

logger = logging.getLogger(__name__)
from unicore.modules.multihead_attention import CrossMultiheadAttention


@register_model("fewshot")
class BindingAffinityModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--mol-pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pocket-pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pocket-encoder-layers",
            type=int,
            help="pocket encoder layers",
        )
        parser.add_argument(
            "--recycling",
            type=int,
            default=1,
            help="recycling nums of decoder",
        )

    def __init__(self, args, mol_dictionary, pocket_dictionary):
        super().__init__()
        drugclip_architecture(args)
        self.args = args
        self.mol_model = UniMolModel(args.mol, mol_dictionary)
        self.pocket_model = UniMolModel(args.pocket, pocket_dictionary)

        self.cross_distance_project = NonLinearHead(
            args.mol.encoder_embed_dim * 2 + args.mol.encoder_attention_heads, 1, "relu"
        )
        self.holo_distance_project = DistanceHead(
            args.mol.encoder_embed_dim + args.mol.encoder_attention_heads, "relu"
        )

        self.mol_project = NonLinearHead(
            args.mol.encoder_embed_dim, 128, "relu"
        )

        self.logit_scale = nn.Parameter(torch.ones([1], device="cuda") * np.log(14))

        self.pocket_project = NonLinearHead(
            args.pocket.encoder_embed_dim, 128, "relu"
        )

        self.fuse_project = NonLinearHead(
            256, 1, "relu"
        )
        self.classification_head = nn.Sequential(
            nn.Linear(args.pocket.encoder_embed_dim + args.pocket.encoder_embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # self.mol_model_ori = UniMolModel(args.mol, mol_dictionary)
        # self.pocket_model_ori = UniMolModel(args.pocket, pocket_dictionary)

        # self.learnable_embed_1_mol = nn.Parameter(
        #     self.mol.embed_tokens.weight[1].clone(), requires_grad=True
        # )
        # self.learnable_embed_1_pocket = nn.Parameter(
        #     self.pocket.embed_tokens.weight[1].clone(), requires_grad=True
        # )
        # self.mol_token_embed = nn.Parameter(
        #     self.args.mol.prompt_tokens, args.mol.encoder_embed_dim, 0
        # )
        # self.pocket_token_embed = nn.Embedding(
        #     self.args.pocket.prompt_tokens, args.pocket.encoder_embed_dim, 0
        # )
        # torch.manual_seed(42)
        # # self.prompt_tokens = prompt_tokens
        #
        # self.prompt_dropout_mol = Dropout(0.0)
        # # prompt_dim = self.prompt_config.PROJECT
        # self.prompt_proj_mol = nn.Linear(
        #     512, self.mol_model.encoder.embed_dim)
        # nn.init.kaiming_normal_(
        #     self.prompt_proj_mol.weight, a=0, mode='fan_out')
        # self.prompt_embeddings_mol = nn.Parameter(torch.zeros(
        #     1, self.args.mol.prompt_tokens, 512))
        # self.deep_layers = 5
        # self.prompt_embeddings = nn.Parameter(torch.zeros(
        #     self.deep_layers, self.prompt_tokens, 512))
        # total_d_layer = 16 - 1
        # self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
        #     total_d_layer, self.prompt_tokens, 512))
        # xavier_uniform initialization
        # val = math.sqrt(6. / (512 + 512))
        # nn.init.uniform_(self.prompt_embeddings_mol, -val, val)
        # self.prompt_dropout_pocket = Dropout(0.0)
        # # prompt_dim = self.prompt_config.PROJECT
        # self.prompt_proj_pocket = nn.Linear(
        #     512, self.pocket_model.encoder.embed_dim)
        # nn.init.kaiming_normal_(
        #     self.prompt_proj_pocket.weight, a=0, mode='fan_out')
        # self.prompt_embeddings_pocket = nn.Parameter(torch.zeros(
        #     1, self.args.pocket.prompt_tokens, 512))
        # self.deep_layers = 5
        # self.prompt_embeddings = nn.Parameter(torch.zeros(
        #     self.deep_layers, self.prompt_tokens, 512))
        # total_d_layer = 16 - 1
        # self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
        #     total_d_layer, self.prompt_tokens, 512))
        # xavier_uniform initialization
        # val = math.sqrt(6. / (512 + 512))
        # nn.init.uniform_(self.prompt_embeddings_pocket, -val, val)
        #
        # self.cross_layers = nn.ModuleList(
        #     [
        #         CrossMultiheadAttention(
        #         embed_dim = 512,
        #         num_heads=4,
        #         dropout=0.1)
        #         for _ in range(1)
        #     ])
        # self.cross_layers_enabled = [0,7,14]

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary, task.pocket_dictionary)

    def fill_attn_mask(self, attn_mask, padding_mask, x, fill_val=float("-inf")):
        if attn_mask is not None and padding_mask is not None:
            # merge key_padding_mask and attn_mask
            attn_mask = attn_mask.view(x.size(0), -1, x.size(1), x.size(1))
            attn_mask.masked_fill_(
                padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                fill_val,
            )
            attn_mask = attn_mask.view(-1, x.size(1), x.size(1))
            padding_mask = None
        return attn_mask, padding_mask

    def forward(
            self,
            mol_src_tokens_ori=None,
            mol_src_distance_ori=None,
            mol_src_edge_type_ori=None,
            pocket_src_tokens_ori=None,
            pocket_src_distance_ori=None,
            pocket_src_edge_type_ori=None,
            mol_src_tokens=None,
            mol_src_distance=None,
            mol_src_edge_type=None,
            pocket_src_tokens=None,
            pocket_src_distance=None,
            pocket_src_edge_type=None,
            train=True,
            inference=False,
            mol_rep_ori = None,
            pocket_rep_ori = None,
            **kwargs
    ):
        def get_dist_features(dist, et, flag):
            if flag == "mol":
                n_node = dist.size(-1)
                gbf_feature = self.mol_model.gbf(dist, et)
                gbf_result = self.mol_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                return graph_attn_bias
            else:
                n_node = dist.size(-1)
                gbf_feature = self.pocket_model.gbf(dist, et)
                gbf_result = self.pocket_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                return graph_attn_bias

        def get_dist_features_ori(dist, et, flag):
            if flag == "mol":
                n_node = dist.size(-1)
                gbf_feature = self.mol_model_ori.gbf(dist, et)
                gbf_result = self.mol_model_ori.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                return graph_attn_bias
            else:
                n_node = dist.size(-1)
                gbf_feature = self.pocket_model_ori.gbf(dist, et)
                gbf_result = self.pocket_model_ori.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                return graph_attn_bias

        # 获取mol和pocket的数量
        B_mol = mol_src_tokens.size(0)  # mol的batch size
        B_pocket = pocket_src_tokens.size(0)  # pocket的batch size

        # 处理原始模型的mol特征
        # mol_padding_mask_ori = mol_src_tokens_ori.eq(self.mol_model_ori.padding_idx)
        # mol_x_ori = self.mol_model_ori.embed_tokens(mol_src_tokens_ori)
        # mol_graph_attn_bias_ori = get_dist_features_ori(
        #     mol_src_distance_ori, mol_src_edge_type_ori, "mol"
        # )
        # mol_outputs_ori = self.mol_model_ori.encoder(
        #     mol_x_ori, padding_mask=mol_padding_mask_ori, attn_mask=mol_graph_attn_bias_ori
        # )
        # mol_rep_ori = mol_outputs_ori[0][:, 1:, :]  # [B_mol, Lm, D]
        #
        # # 处理原始模型的pocket特征
        # pocket_padding_mask_ori = pocket_src_tokens_ori.eq(self.pocket_model_ori.padding_idx)
        # pocket_x_ori = self.pocket_model_ori.embed_tokens(pocket_src_tokens_ori)
        # pocket_graph_attn_bias_ori = get_dist_features_ori(
        #     pocket_src_distance_ori, pocket_src_edge_type_ori, "pocket"
        # )
        # pocket_outputs_ori = self.pocket_model_ori.encoder(
        #     pocket_x_ori, padding_mask=pocket_padding_mask_ori, attn_mask=pocket_graph_attn_bias_ori
        # )
        # pocket_rep_ori = pocket_outputs_ori[0][:, 1:, :]  # [B_pocket, Lp, D]

        # # 原版 5D：逐特征维做 token×token 交互，再对 Lm/Lp 求均值得到 pairwise 交互特征
        mol_tok = mol_rep_ori[:, 1:, :]            # [B_mol, Lm, D]
        poc_tok = pocket_rep_ori[:, 1:, :]         # [B_pocket, Lp, D]
        mol_expand = mol_tok.unsqueeze(1)          # [B_mol, 1, Lm, D]
        pocket_expand = poc_tok.unsqueeze(0)       # [1, B_pocket, Lp, D]
        interact = mol_expand.unsqueeze(3) * pocket_expand.unsqueeze(2)  # [B_mol, B_pocket, Lm, Lp, D]
        interact_feat = interact.mean(dim=(2, 3))  # [B_mol, B_pocket, D]
        # interact_feat = torch.ones(
        #     B_mol,            # B_mol
        #     B_pocket,            # B_pocket
        #     mol_rep_ori.size(-1),           # D
        #     device=mol_rep_ori.device,
        #     dtype=mol_rep_ori.dtype,
        # )


        # 处理新的mol模型
        mol_padding_mask = mol_src_tokens.eq(self.mol_model.padding_idx)
        mol_x = self.mol_model.embed_tokens(mol_src_tokens)
        mol_graph_attn_bias = get_dist_features(
            mol_src_distance, mol_src_edge_type, "mol"
        )
        # 为每个 mol 汇聚所有 pocket 的 pairwise 交互
        mol_interact_feat = interact_feat.mean(dim=1)  # [B_mol, D]

        mol_outputs = self.mol_model.encoder(
            mol_x, padding_mask=mol_padding_mask, attn_mask=mol_graph_attn_bias,fine_feat=mol_interact_feat
        )
        mol_encoder_rep = mol_outputs[0]

        # 处理新的pocket模型
        pocket_padding_mask = pocket_src_tokens.eq(self.pocket_model.padding_idx)
        pocket_x = self.pocket_model.embed_tokens(pocket_src_tokens)
        pocket_graph_attn_bias = get_dist_features(
            pocket_src_distance, pocket_src_edge_type, "pocket"
        )

        # 为每个 pocket 汇聚所有 mol 的 pairwise 交互
        pocket_interact_feat = interact_feat.mean(dim=0)  # [B_pocket, D]

        pocket_outputs = self.pocket_model.encoder(
            pocket_x, padding_mask=pocket_padding_mask, attn_mask=pocket_graph_attn_bias, fine_feat=pocket_interact_feat
        )
        pocket_encoder_rep = pocket_outputs[0]

        # 提取最终的表示
        mol_rep = mol_encoder_rep[:, 0, :]  # [B_mol, D]
        pocket_rep = pocket_encoder_rep[:, 0, :]  # [B_pocket, D]

        mol_emb = self.mol_project(mol_rep)  # [B_mol, D]
        pocket_emb = self.pocket_project(pocket_rep)  # [B_pocket, D]

        mol_emb = mol_emb / mol_emb.norm(dim=1, keepdim=True)
        pocket_emb = pocket_emb / pocket_emb.norm(dim=1, keepdim=True)

        # if train:
        #     # 计算所有mol和所有pocket之间的相似度矩阵
        #     # [B_mol, D] @ [D, B_pocket] = [B_mol, B_pocket]
        #     ba_predict = torch.matmul(mol_emb, torch.transpose(pocket_emb, 0, 1))
        #     ba_predict = ba_predict * self.logit_scale.exp().detach()
        #     return ba_predict, self.logit_scale.exp()
        #
        # if inference:
        return mol_emb, pocket_emb

        # mol_padding_mask_ori = mol_src_tokens_ori.eq(self.mol_model_ori.padding_idx)
        # mol_x_ori = self.mol_model_ori.embed_tokens(mol_src_tokens_ori)
        # mol_graph_attn_bias_ori = get_dist_features_ori(
        #     mol_src_distance_ori, mol_src_edge_type_ori, "mol"
        # )
        # mol_outputs_ori = self.mol_model_ori.encoder(
        #     mol_x_ori, padding_mask=mol_padding_mask_ori, attn_mask=mol_graph_attn_bias_ori
        # )
        # mol_rep_ori = mol_outputs_ori[0][:,1:,:]

        # pocket_padding_mask_ori = pocket_src_tokens_ori.eq(self.pocket_model_ori.padding_idx)
        # pocket_x_ori = self.pocket_model_ori.embed_tokens(pocket_src_tokens_ori)
        # pocket_graph_attn_bias_ori = get_dist_features_ori(
        #     pocket_src_distance_ori, pocket_src_edge_type_ori, "pocket"
        # )
        # pocket_outputs_ori = self.pocket_model_ori.encoder(
        #     pocket_x_ori, padding_mask=pocket_padding_mask_ori, attn_mask=pocket_graph_attn_bias_ori
        # )
        # pocket_rep_ori = pocket_outputs_ori[0][:,1:,:]

        # B = mol_rep_ori.size(0)
        # pocket_rep_ori = pocket_rep_ori.expand(B, -1, -1)

        # mol_expand = mol_rep_ori.unsqueeze(2)  # [B, Lm, 1, D]
        # pocket_expand = pocket_rep_ori.unsqueeze(1)  # [B, 1, Lp, D]

        # interact = mol_expand * pocket_expand  # [B, Lm, Lp, D]
        # interact_feat = interact.mean(dim=(1, 2))

        # # [B, D]

        # mol_padding_mask = mol_src_tokens.eq(self.mol_model.padding_idx)
        # mol_x = self.mol_model.embed_tokens(mol_src_tokens)
        # mol_graph_attn_bias = get_dist_features(
        #     mol_src_distance, mol_src_edge_type, "mol"
        # )
        # mol_outputs = self.mol_model.encoder(
        #     mol_x, padding_mask=mol_padding_mask, attn_mask=mol_graph_attn_bias,fine_feat = interact_feat
        # )
        # mol_encoder_rep = mol_outputs[0]

        # pocket_src_tokens = pocket_src_tokens.expand(B, -1)
        # pocket_src_distance = pocket_src_distance.expand(B, -1, -1)
        # pocket_src_edge_type = pocket_src_edge_type.expand(B,-1,-1)
        # pocket_padding_mask = pocket_src_tokens.eq(self.pocket_model.padding_idx)
        # pocket_x = self.pocket_model.embed_tokens(pocket_src_tokens)
        # pocket_graph_attn_bias = get_dist_features(
        #     pocket_src_distance, pocket_src_edge_type, "pocket"
        # )
        # pocket_outputs = self.pocket_model.encoder(
        #     pocket_x, padding_mask=pocket_padding_mask, attn_mask=pocket_graph_attn_bias,fine_feat = interact_feat
        # )
        # pocket_encoder_rep = pocket_outputs[0]

        # mol_rep = mol_encoder_rep[:, 0, :]
        # pocket_rep = pocket_encoder_rep[:, 0, :]
        # mol_emb = self.mol_project(mol_rep)
        # pocket_emb = self.pocket_project(pocket_rep)
        # mol_emb = mol_emb / mol_emb.norm(dim=1, keepdim=True)
        # pocket_emb = pocket_emb / pocket_emb.norm(dim=1, keepdim=True)
        # if train:
        #     ba_predict = torch.matmul(pocket_emb, torch.transpose(mol_emb, 0, 1))
        #     ba_predict = ba_predict * self.logit_scale.exp().detach()
        #     return ba_predict, self.logit_scale.exp()
        # if inference:
        #     return mol_emb, pocket_emb

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""

        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates


class DistanceHead(nn.Module):
    def __init__(
            self,
            heads,
            activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        x[x == float("-inf")] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x


@register_model_architecture("fewshot", "fewshot")
def drugclip_architecture(args):
    parser = argparse.ArgumentParser()
    args.mol = parser.parse_args([])
    args.pocket = parser.parse_args([])

    args.mol.encoder_layers = getattr(args, "mol_encoder_layers", 15)
    args.mol.encoder_embed_dim = getattr(args, "mol_encoder_embed_dim", 512)
    args.mol.encoder_ffn_embed_dim = getattr(args, "mol_encoder_ffn_embed_dim", 2048)
    args.mol.encoder_attention_heads = getattr(args, "mol_encoder_attention_heads", 64)
    args.mol.dropout = getattr(args, "mol_dropout", 0.1)
    args.mol.emb_dropout = getattr(args, "mol_emb_dropout", 0.1)
    args.mol.attention_dropout = getattr(args, "mol_attention_dropout", 0.1)
    args.mol.activation_dropout = getattr(args, "mol_activation_dropout", 0.0)
    args.mol.pooler_dropout = getattr(args, "mol_pooler_dropout", 0.0)
    args.mol.max_seq_len = getattr(args, "mol_max_seq_len", 512)
    args.mol.activation_fn = getattr(args, "mol_activation_fn", "gelu")
    args.mol.pooler_activation_fn = getattr(args, "mol_pooler_activation_fn", "tanh")
    args.mol.post_ln = getattr(args, "mol_post_ln", False)
    args.mol.masked_token_loss = -1.0
    args.mol.masked_coord_loss = -1.0
    args.mol.masked_dist_loss = -1.0
    args.mol.x_norm_loss = -1.0
    args.mol.delta_pair_repr_norm_loss = -1.0
    args.mol.delta_pair_repr_norm_loss = -1.0
    args.mol.prompt_tokens = args.mol_token

    args.pocket.encoder_layers = getattr(args, "pocket_encoder_layers", 15)
    args.pocket.encoder_embed_dim = getattr(args, "pocket_encoder_embed_dim", 512)
    args.pocket.encoder_ffn_embed_dim = getattr(
        args, "pocket_encoder_ffn_embed_dim", 2048
    )
    args.pocket.encoder_attention_heads = getattr(
        args, "pocket_encoder_attention_heads", 64
    )
    args.pocket.dropout = getattr(args, "pocket_dropout", 0.1)
    args.pocket.emb_dropout = getattr(args, "pocket_emb_dropout", 0.1)
    args.pocket.attention_dropout = getattr(args, "pocket_attention_dropout", 0.1)
    args.pocket.activation_dropout = getattr(args, "pocket_activation_dropout", 0.0)
    args.pocket.pooler_dropout = getattr(args, "pocket_pooler_dropout", 0.0)
    args.pocket.max_seq_len = getattr(args, "pocket_max_seq_len", 512)
    args.pocket.activation_fn = getattr(args, "pocket_activation_fn", "gelu")
    args.pocket.pooler_activation_fn = getattr(
        args, "pocket_pooler_activation_fn", "tanh"
    )
    args.pocket.post_ln = getattr(args, "pocket_post_ln", False)
    args.pocket.masked_token_loss = -1.0
    args.pocket.masked_coord_loss = -1.0
    args.pocket.masked_dist_loss = -1.0
    args.pocket.x_norm_loss = -1.0
    args.pocket.delta_pair_repr_norm_loss = -1.0
    args.pocket.prompt_tokens = args.pocket_token

    base_architecture(args)




# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore.modules import TransformerEncoderLayer, LayerNorm
from functools import reduce
from operator import mul
from torch.nn import Conv2d, Dropout
class TransformerEncoderWithPair(nn.Module):
    def __init__(
        self,
        encoder_layers: int = 6,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        emb_dropout: float = 0.1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        max_seq_len: int = 256,
        activation_fn: str = "gelu",
        post_ln: bool = False,
        no_final_head_layer_norm: bool = False,
        prompt_tokens: int = 5
    ) -> None:

        super().__init__()
        self.emb_dropout = emb_dropout
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.emb_layer_norm = LayerNorm(self.embed_dim)
        if not post_ln:
            self.final_layer_norm = LayerNorm(self.embed_dim)
        else:
            self.final_layer_norm = None

        if not no_final_head_layer_norm:
            self.final_head_layer_norm = LayerNorm(attention_heads)
        else:
            self.final_head_layer_norm = None

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=ffn_embed_dim,
                    attention_heads=attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    post_ln=post_ln,
                )
                for _ in range(encoder_layers)
            ]
        )
        torch.manual_seed(42)
        self.prompt_tokens = prompt_tokens

        # self.prompt_dropout = Dropout(0.0)
        # # prompt_dim = self.prompt_config.PROJECT
        # self.prompt_proj = nn.Linear(
        #     512, self.embed_dim)
        # nn.init.kaiming_normal_(
        #     self.prompt_proj.weight, a=0, mode='fan_out')
        # # self.prompt_embeddings = nn.Parameter(torch.zeros(
        # #     1, self.prompt_tokens, 512))
        # # self.deep_layers = 5
        # # self.prompt_embeddings = nn.Parameter(torch.zeros(
        # #     self.deep_layers, self.prompt_tokens, 512))
        # total_d_layer = len(self.layers)
        # self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
        #     total_d_layer, self.prompt_tokens, 512))
        # # xavier_uniform initialization
        # val = math.sqrt(6. / (512 + 512))
        # nn.init.uniform_(self.deep_prompt_embeddings, -val, val)

        total_d_layer = len(self.layers)
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embed_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.embed_dim)
            ) for _ in range(total_d_layer)
        ])
        # self.fixed_adapter_tokens = nn.ParameterList([
        #     nn.Parameter(torch.zeros(1, self.prompt_tokens, self.embed_dim))  # shape: [1, prompt_tokens, embed_dim]
        #     for _ in range(len(self.layers))
        # ])
        # for token in self.fixed_adapter_tokens:
        #     nn.init.xavier_uniform_(token)
        self.prompt_dropout = Dropout(0.0)
        self.prompt_proj = nn.Linear(
            512, self.embed_dim)

        nn.init.kaiming_normal_(
            self.prompt_proj.weight, a=0, mode='fan_out')
        self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
            total_d_layer, self.prompt_tokens, 512))
        val = math.sqrt(6. / (512 + 512))
        nn.init.uniform_(self.deep_prompt_embeddings, -val, val)
        # self.prompt_weight = nn.Parameter(torch.tensor(1.0))
        # self.gate_weight = nn.Parameter(torch.tensor(1.0))

        # ------------------------------------------------------------------
        # `tmi_gate_mode` (set by the `fewshot` model from `--tmi-gate-mode`)
        #
        # `legacy`: prompt = prompt_embed * adapters(fine_feat).
        #              With the default initialisations |prompt| ~ 2e-3 while the
        #              real atom/residue tokens have |x| ~ 22.6, i.e. the prompt
        #              positions are numerically zero vectors and the TMI feature
        #              has *no* influence on the output (verified with
        #              `script/tmi_sensitivity_check.py`: zeroing fine_feat leaves
        #              the embeddings bit-identical).  Kept for reproducibility.
        # `residual` : gate = adapters(LayerNorm(fine_feat)) used as a *modulation
        #              around 1*, and the prompt tokens are LayerNorm-ed so that
        #              their scale (~sqrt(D) = 22.6) matches the real tokens.
        #              This is what makes the TMI variants actually distinguishable.
        # ------------------------------------------------------------------
        self.tmi_gate_mode = "legacy"
        self.prompt_gate_norm = LayerNorm(self.embed_dim)
        self.prompt_out_norm = LayerNorm(self.embed_dim)
        # per-layer injection strength for `gated`; zero-init => identity w.r.t.
        # the published `legacy` configuration
        self.tmi_alpha = nn.Parameter(torch.zeros(total_d_layer))
        # ------------------------------------------------------------------
        # Bounding / restricting the `gated` injection.
        #
        # The raw `alpha_t * LayerNorm(fine_feat)` term is unbounded and lives at
        # token scale (~sqrt(D) = 22.6) while `prompt_embed * gate` is~2e-3.  A
        # few tens of SGD steps on 2-16 support molecules are enough for alpha_t
        # to overshoot, which replaces the (numerically zero) prompt positions by
        # full-scale artificial tokens in *every* layer and destroys the frozen
        # encoder's geometry -- empirically the loss w.r.t. `legacy` grows with
        # the number of support updates (-0.16 AUROC at 2-shot, -1.92 at 16-shot).
        #
        # `tmi_alpha_beta`      : hard bound, effective strength = beta*tanh(alpha)
        #                         so |strength| < beta regardless of overshoot.
        #                         beta=1 reproduces the unbounded behaviour for
        #                         small alpha (tanh(a) ~ a), hence it is the
        #                         backward-compatible default.
        # `tmi_inject_layers`   : inject only in the last k layers (-1 = all).
        #                         Shallow injection corrupts the whole geometric
        #                         encoding chain; deep-only injection edits the
        #                         semantics alone.
        # ------------------------------------------------------------------
        self.tmi_alpha_beta = 1.0
        self.tmi_inject_layers = -1

    def _tmi_injection_scale(self, i):
        """Effective (bounded) injection strength of layer `i`, or None if this
        layer must not receive any TMI injection."""
        n_layers = len(self.layers)
        k = self.tmi_inject_layers
        if k is not None and k >= 0 and i < n_layers - k:
            return None
        return self.tmi_alpha_beta * torch.tanh(self.tmi_alpha[i])

    def tmi_injection_strengths(self):
        """Per-layer effective injection strengths (detached, for diagnostics)."""
        with torch.no_grad():
            n_layers = len(self.layers)
            k = self.tmi_inject_layers
            s = self.tmi_alpha_beta * torch.tanh(self.tmi_alpha.detach().float())
            if k is not None and k >= 0:
                mask = torch.zeros_like(s)
                if k > 0:
                    mask[max(n_layers - k, 0):] = 1.0
                s = s * mask
            return s.cpu()

    def forward(
        self,
        emb: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        fine_feat = None
    ) -> torch.Tensor:
        if fine_feat is not None:

            x_list = []

            bsz = emb.size(0)
            # seq_len = emb.size(1)+self.prompt_tokens
            seq_len = emb.size(1)
            x = self.emb_layer_norm(emb)
            # x = torch.cat((
            #     x[:, :1, :],
            #     self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(bsz, -1, -1)),
            #     # self.prompt_dropout(self.prompt_proj(x[:, 1:6, :]).expand(bsz, -1, -1)),
            #     x[:, 1+self.prompt_tokens:, :]
            # ), dim=1)
            x = F.dropout(x, p=self.emb_dropout, training=self.training)





            # account for padding while computing the representation
            if padding_mask is not None:
                x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
            input_attn_mask = attn_mask
            input_padding_mask = padding_mask

            def fill_attn_mask(attn_mask, padding_mask, fill_val=float("-inf")):
                if attn_mask is not None and padding_mask is not None:
                    # merge key_padding_mask and attn_mask
                    attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
                    attn_mask.masked_fill_(
                        padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                        fill_val,
                    )
                    attn_mask = attn_mask.view(-1, seq_len, seq_len)
                    padding_mask = None
                return attn_mask, padding_mask

            assert attn_mask is not None
            attn_mask, padding_mask = fill_attn_mask(attn_mask, padding_mask)

            x_list.append(x)

            for i in range(len(self.layers)):
                # for i in range(self.deep_layers):

                # if self.training:
                #     gate = self.adapters[i](fine_feat)
                #     prompt_embed = self.prompt_proj(self.deep_prompt_embeddings[i]).expand(bsz, -1, -1)
                #     prompt_input= self.prompt_dropout(gate* prompt_embed)
                #     with torch.no_grad():
                #         self.fixed_adapter_tokens[i].data = (
                #                 0.99 * self.fixed_adapter_tokens[i].data +
                #                 0.01 * prompt_input.mean(dim=0, keepdim=True).data
                #         )
                # else:
                #     prompt_input = self.fixed_adapter_tokens[i].expand(bsz, -1, -1)
                # if i==0:
                if self.tmi_gate_mode == "none":
                    # ablation of the paper ("we remove the TMI module ... the model
                    # simply inserts randomly initialised prompt tokens"): the prompt
                    # tokens are used WITHOUT the gate, i.e. |prompt| ~ 1.4 instead of
                    # ~2e-3, which is why that ablation is noisy -- the gate was in
                    # fact switching the prompts OFF, not injecting interaction info.
                    prompt_embed = self.prompt_proj(
                        self.deep_prompt_embeddings[i]).expand(bsz, -1, -1)
                    prompt_input = self.prompt_dropout(prompt_embed)
                elif self.tmi_gate_mode == "legacy":
                    gate = self.adapters[i](fine_feat).unsqueeze(1)
                    prompt_embed = self.prompt_proj(
                        self.deep_prompt_embeddings[i]).expand(bsz, -1, -1)
                    prompt_input = self.prompt_dropout(prompt_embed * gate)
                elif self.tmi_gate_mode == "gated":
                    # Identity-initialised gated residual injection.
                    #prompt = prompt_embed * adapters(fine_feat)            <- legacy
                    #+ beta*tanh(alpha_t) * LayerNorm(fine_feat) <- new
                    # `alpha_t` starts at 0, so at initialisation this branch is
                    # BIT-IDENTICAL to `legacy` (the published configuration), while
                    # d L / d alpha_t = <d L / d prompt, LayerNorm(fine_feat)> is at
                    # token scale and is NOT damped by the ~3e-3 gate, so the support
                    # set can actually learn whether/how much to use the interaction.
                    # `beta*tanh(.)` bounds that strength and `tmi_inject_layers`
                    # restricts it to the deepest layers (see __init__).
                    gate = self.adapters[i](fine_feat).unsqueeze(1)
                    prompt_embed = self.prompt_proj(
                        self.deep_prompt_embeddings[i]).expand(bsz, -1, -1)
                    prompt_input = prompt_embed * gate
                    alpha = self._tmi_injection_scale(i)
                    if alpha is not None:
                        prompt_input = prompt_input + alpha * self.prompt_gate_norm(
                            fine_feat).unsqueeze(1)
                    prompt_input = self.prompt_dropout(prompt_input)
                else:
                    # normalise the TMI feature (it is an element-wise product of
                    # two hidden states, so its scale/mean are arbitrary), use the
                    # gate as a modulation around 1, and bring the prompt tokens to
                    # the same scale as the real tokens.
                    feat = self.prompt_gate_norm(fine_feat)
                    gate = self.adapters[i](feat).unsqueeze(1)
                    prompt_embed = self.prompt_proj(
                        self.deep_prompt_embeddings[i]).expand(bsz, -1, -1)
                    prompt_input = prompt_embed * (1.0 + gate)
                    if self.tmi_gate_mode == "inject":
                        # `adapters` are randomly initialised (they are absent from
                        # the DrugCLIP checkpoint) so `gate` is a small random
                        # perturbation and the TMI content would still not reach the
                        # encoder.  Add the normalised interaction feature directly
                        # to the prompt tokens instead: the injected content is
                        # TMI-dependent by construction, no pre-training needed.
                        prompt_input = prompt_input + feat.unsqueeze(1)
                    prompt_input = self.prompt_dropout(
                        self.prompt_out_norm(prompt_input)
                    )
                # [bsz, prompt_tokens, embed_dim]

                x = torch.cat((
                    x[:, :1, :],
                        prompt_input,
                    x[:, 1 + self.prompt_tokens:, :]
                ), dim=1)


                    # x = torch.cat((
                    #     x[:, :1, :],
                    #     self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[i]).expand(bsz, -1, -1))+fine_feat,
                    #     # self.prompt_dropout(self.prompt_proj(x[:, 1:6, :]).expand(bsz, -1, -1)),
                    #     x[:, 1+self.prompt_tokens:, :]
                    # ), dim=1)

                x, attn_mask, _ = self.layers[i](
                    x, padding_mask=padding_mask, attn_bias=attn_mask, return_attn=True
                )


                x_list.append(x)

            def norm_loss(x, eps=1e-10, tolerance=1.0):
                x = x.float()
                max_norm = x.shape[-1] ** 0.5
                norm = torch.sqrt(torch.sum(x**2, dim=-1) + eps)
                error = torch.nn.functional.relu((norm - max_norm).abs() - tolerance)
                return error

            def masked_mean(mask, value, dim=-1, eps=1e-10):
                return (
                    torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))
                ).mean()

            x_norm = norm_loss(x)
            if input_padding_mask is not None:
                token_mask = 1.0 - input_padding_mask.float()
            else:
                token_mask = torch.ones_like(x_norm, device=x_norm.device)
            x_norm = masked_mean(token_mask, x_norm)

            if self.final_layer_norm is not None:
                x = self.final_layer_norm(x)
                x_list.append(x)

            delta_pair_repr = attn_mask - input_attn_mask
            delta_pair_repr, _ = fill_attn_mask(delta_pair_repr, input_padding_mask, 0)
            attn_mask = (
                attn_mask.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
            )
            delta_pair_repr = (
                delta_pair_repr.view(bsz, -1, seq_len, seq_len)
                .permute(0, 2, 3, 1)
                .contiguous()
            )

            pair_mask = token_mask[..., None] * token_mask[..., None, :]
            delta_pair_repr_norm = norm_loss(delta_pair_repr)
            delta_pair_repr_norm = masked_mean(
                pair_mask, delta_pair_repr_norm, dim=(-1, -2)
            )

            if self.final_head_layer_norm is not None:
                delta_pair_repr = self.final_head_layer_norm(delta_pair_repr)

            return x, attn_mask, delta_pair_repr, x_norm, delta_pair_repr_norm,x_list
        else:
            bsz = emb.size(0)
            seq_len = emb.size(1)
            x = self.emb_layer_norm(emb)
            x = F.dropout(x, p=self.emb_dropout, training=self.training)

            # account for padding while computing the representation
            if padding_mask is not None:
                x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
            input_attn_mask = attn_mask
            input_padding_mask = padding_mask

            def fill_attn_mask(attn_mask, padding_mask, fill_val=float("-inf")):
                if attn_mask is not None and padding_mask is not None:
                    # merge key_padding_mask and attn_mask
                    attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
                    attn_mask.masked_fill_(
                        padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                        fill_val,
                    )
                    attn_mask = attn_mask.view(-1, seq_len, seq_len)
                    padding_mask = None
                return attn_mask, padding_mask

            assert attn_mask is not None
            # breakpoint()
            attn_mask, padding_mask = fill_attn_mask(attn_mask, padding_mask)
            # breakpoint()

            for i in range(len(self.layers)):
                x, attn_mask, _ = self.layers[i](
                    x, padding_mask=padding_mask, attn_bias=attn_mask, return_attn=True
                )

            def norm_loss(x, eps=1e-10, tolerance=1.0):
                x = x.float()
                max_norm = x.shape[-1] ** 0.5
                norm = torch.sqrt(torch.sum(x**2, dim=-1) + eps)
                error = torch.nn.functional.relu((norm - max_norm).abs() - tolerance)
                return error

            def masked_mean(mask, value, dim=-1, eps=1e-10):
                return (
                    torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))
                ).mean()

            x_norm = norm_loss(x)
            if input_padding_mask is not None:
                token_mask = 1.0 - input_padding_mask.float()
            else:
                token_mask = torch.ones_like(x_norm, device=x_norm.device)
            x_norm = masked_mean(token_mask, x_norm)

            if self.final_layer_norm is not None:
                x = self.final_layer_norm(x)

            delta_pair_repr = attn_mask - input_attn_mask
            delta_pair_repr, _ = fill_attn_mask(delta_pair_repr, input_padding_mask, 0)
            attn_mask = (
                attn_mask.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
            )
            delta_pair_repr = (
                delta_pair_repr.view(bsz, -1, seq_len, seq_len)
                .permute(0, 2, 3, 1)
                .contiguous()
            )

            pair_mask = token_mask[..., None] * token_mask[..., None, :]
            delta_pair_repr_norm = norm_loss(delta_pair_repr)
            delta_pair_repr_norm = masked_mean(
                pair_mask, delta_pair_repr_norm, dim=(-1, -2)
            )

            if self.final_head_layer_norm is not None:
                delta_pair_repr = self.final_head_layer_norm(delta_pair_repr)

            return x, attn_mask, delta_pair_repr, x_norm, delta_pair_repr_norm

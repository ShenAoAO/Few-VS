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
        parser.add_argument(
            "--tmi-mode",
            type=str,
            default="legacy",
            choices=["legacy", "hadamard", "pairwise"],
            help="Token-Level Mutual Interaction aggregation. "
                 "`legacy`: uniform average over all molecule-pocket token pairs "
                 "(Eq. 2-3 of the manuscript, provably identical to `hadamard`); "
                 "`hadamard`: explicit mean-pooling + Hadamard-product baseline "
                 "(padding-masked); "
                 "`pairwise`: affinity-weighted token-pair aggregation that cannot "
                 "be factorised into a product of mean-pooled representations.",
        )
        parser.add_argument(
            "--tmi-cond",
            type=str,
            default="batch",
            choices=["batch", "invariant", "pair"],
            help="How the interaction-conditioned features are aggregated. "
                 "`batch`: average over the opposite batch dimension (Eq. 4-5, "
                 "depends on mini-batch composition); "
                 "`invariant`: token-vs-cached-anchor interaction, batch invariant "
                 "but the cross-modality side is reduced to a single anchor vector; "
                 "`pair`: the molecule branch keeps the *exact* token-pair feature "
                 "of the (molecule, pocket) pair being scored (requires a pocket "
                 "batch size of 1, which is enforced by the task) while the pocket "
                 "branch uses the cached support-set anchor -- genuinely "
                 "pair-specific *and* batch invariant.",
        )
        parser.add_argument(
            "--tmi-affinity",
            type=str,
            default="cosine",
            choices=["cosine", "dot"],
            help="Similarity used by the token-pair affinity of `pairwise` mode. "
                 "`cosine`: L2-normalised tokens, logits = cos / tau, so the "
                 "sharpness is controlled by `--tmi-temperature` alone and does "
                 "not depend on the (large) token norms; "
                 "`dot`: legacy scaled dot product, logits = <m, p> / (tau*sqrt(D)) "
                 "-- with tau=1 the 1/sqrt(512) factor flattens the distribution to "
                 "a nearly uniform average over thousands of token pairs, which makes "
                 "`pairwise` numerically collapse onto `legacy` / `hadamard`.",
        )
        parser.add_argument(
            "--tmi-temperature",
            type=float,
            default=None,
            help="softmax temperature of the token-pair affinity in `pairwise` mode "
                 "(default: 0.05 for `--tmi-affinity cosine`, 1.0 for `dot`)",
        )
        parser.add_argument(
            "--tmi-anchor-momentum",
            type=float,
            default=0.1,
            help="EMA momentum used to accumulate the batch-invariant TMI anchors "
                 "during few-shot adaptation",
        )
        parser.add_argument(
            "--tmi-gate-mode",
            type=str,
            default="legacy",
            choices=["legacy", "residual", "inject", "gated", "none"],
            help="How the TMI feature gates the deep prompts. "
                 "`legacy`: prompt = prompt_embed * adapters(fine_feat); with the "
                 "default initialisations |prompt| ~ 2e-3 against |token| ~ 22.6, so "
                 "the prompt positions are numerically zero and the TMI feature has "
                 "NO effect on the output (see script/tmi_sensitivity_check.py) -- "
                 "this is why every TMI variant gave the same AUC; "
                 "`gated`: legacy + alpha_t * LayerNorm(fine_feat) with alpha_t "
                 "zero-initialised -- bit-identical to `legacy` at initialisation, but "
                 "alpha_t receives an undamped, token-scale gradient so the support set "
                 "can learn whether to use the interaction  << recommended; "
                 "`residual`: LayerNorm(fine_feat) -> adapter -> modulation around 1, "
                 "prompt tokens LayerNorm-ed to the token scale (fixes the scale, but "
                 "`adapters` are absent from the DrugCLIP checkpoint, so the gate is "
                 "still a small random perturbation unless it is pre-trained); "
                 "`inject`: same, plus the normalised interaction feature is added "
                 "directly to the prompt tokens, so the TMI content reaches the "
                 "encoder without requiring pre-trained adapters; "
                 "`none`: prompt tokens without any gate -- this is the "
                 "'we remove the TMI module' ablation of the paper.",
        )
        parser.add_argument(
            "--tmi-diag",
            action="store_true",
            default=False,
            help="log how far the `pairwise` TMI feature actually is from the "
                 "mean-pool + Hadamard baseline (cosine / relative L2) together with "
                 "the affinity statistics (logit std, effective number of token pairs)",
        )
        parser.add_argument(
            "--tmi-alpha-beta",
            type=float,
            default=1.0,
            help="hard bound on the `gated` injection strength: the effective "
                 "per-layer strength is beta*tanh(alpha_t), so |strength| < beta "
                 "no matter how far alpha_t overshoots on the support set. "
                 "beta=1.0 reproduces the previous unbounded behaviour for small "
                 "alpha;0.01-0.1 keeps the injected term one to two orders of "
                 "magnitude below the real atom/residue tokens.",
        )
        parser.add_argument(
            "--tmi-inject-layers",
            type=int,
            default=-1,
            help="restrict the `gated` TMI injection to the last N encoder layers "
                 "(-1 = all layers, the previous behaviour; 0 = disable the "
                 "injection). Shallow injection corrupts the geometric encoding "
                 "chain of the frozen encoder, deep-only injection edits semantics.",
        )
        parser.add_argument(
            "--tmi-alpha-lr-scale",
            type=float,
            default=1.0,
            help="learning-rate multiplier applied to `tmi_alpha` only. Its "
                 "gradient is at token scale (~22.6) instead of the ~3e-3 gate "
                 "scale, so with the shared lr it overshoots within a few support "
                 "steps; 0.02-0.005 puts it back on the same effective scale as "
                 "the other adapted parameters.",
        )
        parser.add_argument(
            "--tmi-diag-interval",
            type=int,
            default=200,
            help="log the TMI diagnostics every N calls (the first call is always "
                 "logged)",
        )

    def __init__(self, args, mol_dictionary, pocket_dictionary):
        super().__init__()
        drugclip_architecture(args)
        self.args = args
        self.mol_model = UniMolModel(args.mol, mol_dictionary)
        self.pocket_model = UniMolModel(args.pocket, pocket_dictionary)
        self.mol_eos_idx = mol_dictionary.eos()
        self.pocket_eos_idx = pocket_dictionary.eos()

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
        # ------------------------------------------------------------------
        # Batch-invariant TMI anchors.
        # They are accumulated (EMA) over the *support* molecules and over the
        # *target* pockets during few-shot adaptation and kept frozen at
        # inference time, so that the interaction-conditioned representation of
        # a molecule (resp. a pocket) no longer depends on the other candidates
        # that happen to share its evaluation mini-batch.
        # ------------------------------------------------------------------
        self.register_buffer(
            "tmi_mol_anchor", torch.zeros(args.mol.encoder_embed_dim)
        )
        self.register_buffer(
            "tmi_pocket_anchor", torch.zeros(args.pocket.encoder_embed_dim)
        )
        self.register_buffer("tmi_anchor_count", torch.zeros(2))
        # how the TMI feature is injected into the deep prompts (see
        # `TransformerEncoderWithPair.tmi_gate_mode`)
        gate_mode = getattr(args, "tmi_gate_mode", "legacy")
        self.mol_model.encoder.tmi_gate_mode = gate_mode
        self.pocket_model.encoder.tmi_gate_mode = gate_mode
        # bound / restrict the `gated` injection (see TransformerEncoderWithPair)
        alpha_beta = float(getattr(args, "tmi_alpha_beta", 1.0))
        inject_layers = int(getattr(args, "tmi_inject_layers", -1))
        for enc in (self.mol_model.encoder, self.pocket_model.encoder):
            enc.tmi_alpha_beta = alpha_beta
            enc.tmi_inject_layers = inject_layers
        # diagnostics state (see `_tmi_diag_enabled` / `_log_tmi_diag`)
        self._tmi_diag_step = 0
        self._tmi_last_affinity_stats = None
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

    # ==================================================================
    # Token-Level Mutual Interaction (TMI)
    # ==================================================================
    def _graph_attn_bias(self, dist, et, flag, model=None):
        if model is None:
            model = self.mol_model if flag == "mol" else self.pocket_model
        n_node = dist.size(-1)
        gbf_feature = model.gbf(dist, et)
        gbf_result = model.gbf_proj(gbf_feature)
        graph_attn_bias = gbf_result
        graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
        graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
        return graph_attn_bias

    def freeze_tmi_encoders(self):
        """Snapshot the *current* (pre-adaptation) encoders and use them for TMI.

        The legacy pipeline fed the TMI with token representations that had been
        pre-computed offline with the frozen pre-trained encoders (Eq. 1).  Since
        the few-shot adaptation also tunes the LayerNorms of `mol_model` /
        `pocket_model`, encoding the TMI input with the live modules would make
        it drift.  Calling this method once per screening target -- before the
        adaptation starts -- restores the legacy semantics without requiring the
        offline embedding files.
        """
        import copy

        self.mol_model_ori = copy.deepcopy(self.mol_model)
        self.pocket_model_ori = copy.deepcopy(self.pocket_model)
        for m in (self.mol_model_ori, self.pocket_model_ori):
            m.eval()
            m.requires_grad_(False)
        return self

    def _encode_frozen(self, src_tokens, src_distance, src_edge_type, flag):
        """Prompt-free, dropout-free, detached encoding used as TMI input.

        This reproduces the frozen pre-trained token representations of
        Eq. (1) and makes the TMI features deterministic.  If
        :meth:`freeze_tmi_encoders` has been called, the frozen *snapshot* is
        used, so the TMI input does not change during few-shot adaptation.
        """
        if flag == "mol":
            model = getattr(self, "mol_model_ori", None)
            model = self.mol_model if model is None else model
        else:
            model = getattr(self, "pocket_model_ori", None)
            model = self.pocket_model if model is None else model
        was_training = model.training
        model.eval()
        with torch.no_grad():
            padding_mask = src_tokens.eq(model.padding_idx)
            x = model.embed_tokens(src_tokens)
            attn_bias = self._graph_attn_bias(
                src_distance, src_edge_type, flag, model=model
            )
            outputs = model.encoder(
                x, padding_mask=padding_mask, attn_mask=attn_bias
            )
        if was_training:
            model.train()
        return outputs[0].detach()

    def reset_tmi_anchors(self):
        """Clear the cached anchors. Call once per screening target, before
        few-shot adaptation."""
        self.tmi_mol_anchor.zero_()
        self.tmi_pocket_anchor.zero_()
        self.tmi_anchor_count.zero_()

    def tmi_alpha_report(self):
        """Per-encoder statistics of the learnt `gated` injection strengths.

        Used to check the main failure mode of `--tmi-gate-mode gated`: if
        ``max_abs`` approaches the token scale the prompt positions have stopped
        being near-zero placeholders and the frozen encoder is being distorted.
        """
        out = {}
        for tag, enc in (
            ("mol", self.mol_model.encoder),
            ("pocket", self.pocket_model.encoder),
        ):
            if not hasattr(enc, "tmi_injection_strengths"):
                continue
            s = enc.tmi_injection_strengths()
            out[tag] = {
                "max_abs": float(s.abs().max()),
                "mean_abs": float(s.abs().mean()),
                "l2": float(s.norm()),
                "per_layer": [round(float(v), 5) for v in s.tolist()],
            }
        return out

    @staticmethod
    def _valid_token_mask(rep, src_tokens, padding_idx, eos_idx=None):
        """[B, L] mask of atom tokens in ``rep``.

        ``rep`` has the BOS token removed.  Padding and EOS are excluded so the
        pairwise TMI is genuinely atom-token based rather than partly driven by
        special tokens.
        """
        if (
            src_tokens is not None
            and src_tokens.dim() == 2
            and src_tokens.size(0) == rep.size(0)
            and src_tokens.size(1) == rep.size(1) + 1
        ):
            tokens = src_tokens[:, 1:]
            valid = ~tokens.eq(padding_idx)
            if eos_idx is not None:
                valid = valid & ~tokens.eq(eos_idx)
            return valid
        return torch.ones(
            rep.shape[:2], dtype=torch.bool, device=rep.device
        )

    @staticmethod
    def _masked_mean(rep, valid):
        w = valid.to(rep.dtype).unsqueeze(-1)
        return (rep * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)

    # ------------------------------------------------------------------
    # token-pair affinity
    # ------------------------------------------------------------------
    def _tmi_affinity(self):
        return getattr(self.args, "tmi_affinity", "cosine")

    def _tmi_tau(self):
        tau = getattr(self.args, "tmi_temperature", None)
        if tau is None:
            tau = 0.05 if self._tmi_affinity() == "cosine" else 1.0
        return max(float(tau), 1e-6)

    @staticmethod
    def _l2_normalise(x):
        return F.normalize(x.float(), dim=-1, eps=1e-6).to(x.dtype)

    def _affinity_scale(self, d):
        """Return the (keys, queries) transform and the logit scale.

        `cosine` removes the token norms from the affinity, so the sharpness is
        set by tau only.  `dot` reproduces the legacy scaled dot product, whose
        1/sqrt(D) factor makes the softmax almost uniform for tau=1.
        """
        if self._tmi_affinity() == "cosine":
            return True, 1.0 / self._tmi_tau()
        return False, 1.0 / (self._tmi_tau() * math.sqrt(d))

    def _tmi_pairwise(self, mol_tok, poc_tok, mol_valid, poc_valid):
        """Affinity-weighted token-pair interaction -> [B_mol, B_pocket, D].

            feat[i, j] = sum_{a, b} w[i, j, a, b] * (mol[i, a] * poc[j, b])
            w[i, j, :, :] = softmax_{(a, b)} ( affinity(mol[i, a], poc[j, b]) )

        with ``affinity = cos(m_a, p_b) / tau`` (``--tmi-affinity cosine``, the
        default) or the legacy ``<m_a, p_b> / (tau*sqrt(D))``.  Only the affinity
        uses normalised tokens; the aggregated values are the raw tokens, so the
        feature scale is unchanged.

        Because the weights depend on the individual atom pair, this quantity is
        *not* factorisable into (mean-pooled mol) * (mean-pooled pocket).
        The [B_mol, B_pocket, Lm, Lp, D] tensor is never materialised:
        sum_{a,b} w_{ab} m_a * p_b = sum_a m_a * (W p)_a.
        """
        d = mol_tok.size(-1)
        normalise, scale = self._affinity_scale(d)
        mol_key = self._l2_normalise(mol_tok) if normalise else mol_tok
        poc_key = self._l2_normalise(poc_tok) if normalise else poc_tok
        logits = torch.einsum("iad,jbd->ijab", mol_key, poc_key) * scale
        pair_valid = mol_valid[:, None, :, None] & poc_valid[None, :, None, :]
        logits = logits.masked_fill(~pair_valid, float("-inf"))
        shp = logits.shape
        w = torch.softmax(
            logits.reshape(shp[0], shp[1], -1).float(), dim=-1
        ).reshape(shp).to(mol_tok.dtype)
        if self._tmi_diag_enabled():
            self._tmi_last_affinity_stats = self._affinity_stats(
                logits, w, pair_valid
            )
        ctx = torch.einsum("ijab,jbd->ijad", w, poc_tok)
        return torch.einsum("ijad,iad->ijd", ctx, mol_tok)

    def _tmi_anchor_feat(self, tok, valid, anchor, mode):
        """Interaction between every token of one entity and a single cached
        anchor vector of the opposite modality -> [B, D].

        Depends only on the entity itself and on the (frozen) anchor, hence it is
        invariant to the composition/size/ordering of the evaluation batch.
        """
        if mode == "pairwise":
            d = tok.size(-1)
            normalise, scale = self._affinity_scale(d)
            tok_key = self._l2_normalise(tok) if normalise else tok
            anchor_key = (
                self._l2_normalise(anchor.unsqueeze(0)).squeeze(0)
                if normalise
                else anchor
            )
            logits = torch.einsum("bad,d->ba", tok_key, anchor_key) * scale
            logits = logits.masked_fill(~valid, float("-inf"))
            w = torch.softmax(logits.float(), dim=-1).to(tok.dtype).unsqueeze(-1)
            pooled = (w * tok).sum(dim=1)
        elif mode == "legacy":
            pooled = tok.mean(dim=1)
        else:
            pooled = self._masked_mean(tok, valid)
        return pooled * anchor.unsqueeze(0)

    # ------------------------------------------------------------------
    # diagnostics: is the `pairwise` TMI really different from meanpool+Hadamard?
    # ------------------------------------------------------------------
    def _tmi_diag_enabled(self):
        return bool(getattr(self.args, "tmi_diag", False))

    @staticmethod
    def _affinity_stats(logits, w, pair_valid):
        """Scalar statistics of the token-pair softmax (no big tensor kept)."""
        with torch.no_grad():
            lf = logits.float()
            lf = torch.where(pair_valid, lf, torch.zeros_like(lf))
            cnt = pair_valid.sum().clamp(min=1).float()
            mean = lf.sum() / cnt
            var = (lf.pow(2).sum() / cnt - mean.pow(2)).clamp(min=0.0)
            wf = w.float().reshape(w.size(0), w.size(1), -1)
            n_pairs = pair_valid.reshape(w.size(0), w.size(1), -1).sum(-1).float()
            eff = 1.0 / wf.pow(2).sum(-1).clamp(min=1e-12)
            return {
                "logit_mean": float(mean),
                "logit_std": float(var.sqrt()),
                "n_pairs": float(n_pairs.mean()),
                "eff_pairs": float(eff.mean()),
                "max_w": float(wf.max(dim=-1)[0].mean()),
            }

    def _log_tmi_diag(self, cur, ref, mode, cond):
        """Compare the produced TMI feature ``cur`` with its mean-pool +
        Hadamard counterpart ``ref`` (same shape).  ``cos -> 1`` / ``rel_l2 -> 0``
        means the affinity weighting is (numerically) equivalent to the uniform
        average of Eq. (2)-(3), i.e. the `pairwise` variant has collapsed onto
        `legacy` / `hadamard`."""
        self._tmi_diag_step += 1
        interval = max(int(getattr(self.args, "tmi_diag_interval", 200)), 1)
        if self._tmi_diag_step != 1 and self._tmi_diag_step % interval != 0:
            return
        with torch.no_grad():
            cur = cur.float()
            ref = ref.float()
            num = (cur * ref).sum(dim=-1)
            den = cur.norm(dim=-1).clamp(min=1e-12) * ref.norm(dim=-1).clamp(min=1e-12)
            cos = float((num / den).mean())
            rel = float((cur - ref).norm() / ref.norm().clamp(min=1e-12))
            stats = self._tmi_last_affinity_stats or {}
            logger.info(
                "[TMI diag] step=%d mode=%s cond=%s affinity=%s tau=%.4g | "
                "cos(feat, meanpool*Hadamard)=%.6f rel_l2=%.4f | "
                "logit mean=%.3f std=%.3f | pairs=%.0f eff_pairs=%.1f max_w=%.3e",
                self._tmi_diag_step, mode, cond, self._tmi_affinity(),
                self._tmi_tau(), cos, rel,
                stats.get("logit_mean", float("nan")),
                stats.get("logit_std", float("nan")),
                stats.get("n_pairs", float("nan")),
                stats.get("eff_pairs", float("nan")),
                stats.get("max_w", float("nan")),
            )

    def _update_anchor(self, buf, idx, pooled):
        with torch.no_grad():
            cur = pooled.detach().float().mean(dim=0).to(buf.dtype)
            if float(self.tmi_anchor_count[idx]) == 0.0:
                buf.copy_(cur)
            else:
                m = float(getattr(self.args, "tmi_anchor_momentum", 0.1))
                buf.mul_(1.0 - m).add_(cur, alpha=m)
            self.tmi_anchor_count[idx] += 1

    def compute_tmi(
        self, mol_rep_ori, pocket_rep_ori, mol_src_tokens, pocket_src_tokens
    ):
        """Return the interaction-conditioned features fed to the prompt gates.

        Returns
        -------
        (mol_interact_feat [B_mol, D], pocket_interact_feat [B_pocket, D])
        """
        mode = getattr(self.args, "tmi_mode", "legacy")
        cond = getattr(self.args, "tmi_cond", "batch")

        mol_tok = mol_rep_ori[:, 1:, :]      # [B_mol, Lm, D]
        poc_tok = pocket_rep_ori[:, 1:, :]   # [B_pocket, Lp, D]
        mol_valid = self._valid_token_mask(
            mol_tok,
            mol_src_tokens,
            self.mol_model.padding_idx,
            self.mol_eos_idx,
        )
        poc_valid = self._valid_token_mask(
            poc_tok,
            pocket_src_tokens,
            self.pocket_model.padding_idx,
            self.pocket_eos_idx,
        )

        if mode == "legacy":
            # exactly Eq. (2)-(3): uniform average over *all* token positions
            mol_pool = mol_tok.mean(dim=1)
            poc_pool = poc_tok.mean(dim=1)
        else:
            mol_pool = self._masked_mean(mol_tok, mol_valid)
            poc_pool = self._masked_mean(poc_tok, poc_valid)

        if self.training:
            self._update_anchor(self.tmi_mol_anchor, 0, mol_pool)
            self._update_anchor(self.tmi_pocket_anchor, 1, poc_pool)

        if cond == "invariant":
            if float(self.tmi_anchor_count[0]) > 0.0:
                mol_anchor = self.tmi_mol_anchor
            else:
                mol_anchor = mol_pool.detach().mean(dim=0)
            if float(self.tmi_anchor_count[1]) > 0.0:
                poc_anchor = self.tmi_pocket_anchor
            else:
                poc_anchor = poc_pool.detach().mean(dim=0)
            mol_anchor = mol_anchor.to(mol_tok.dtype)
            poc_anchor = poc_anchor.to(poc_tok.dtype)
            mol_interact_feat = self._tmi_anchor_feat(
                mol_tok, mol_valid, poc_anchor, mode
            )
            pocket_interact_feat = self._tmi_anchor_feat(
                poc_tok, poc_valid, mol_anchor, mode
            )
            if self._tmi_diag_enabled():
                self._log_tmi_diag(
                    mol_interact_feat,
                    mol_pool * poc_anchor.unsqueeze(0),
                    mode,
                    cond,
                )
            return mol_interact_feat, pocket_interact_feat

        # ---- token-pair interaction ----
        if mode == "pairwise":
            interact_feat = self._tmi_pairwise(
                mol_tok, poc_tok, mol_valid, poc_valid
            )
        else:
            # `legacy` / `hadamard`: the token-pair average collapses to the
            # Hadamard product of the two mean-pooled representations, which is
            # computed directly here instead of materialising the 5-D tensor.
            interact_feat = mol_pool.unsqueeze(1) * poc_pool.unsqueeze(0)

        if self._tmi_diag_enabled():
            self._log_tmi_diag(
                interact_feat,
                mol_pool.unsqueeze(1) * poc_pool.unsqueeze(0),
                mode,
                cond,
            )

        if cond == "pair":
            # Molecule branch: no averaging over the opposite batch dimension.
            # The task feeds one pocket at a time (B_pocket == 1), therefore
            # `mean(dim=1)` returns exactly the token-pair feature of the pair
            # (i, j) that is being scored -- the atom/residue correspondence is
            # preserved and nothing depends on the other candidate molecules.
            mol_interact_feat = interact_feat.mean(dim=1)
            # Pocket branch: conditioned on the cached support-set anchor instead
            # of the current query batch -> batch invariant.
            if float(self.tmi_anchor_count[0]) > 0.0:
                mol_anchor = self.tmi_mol_anchor
            else:
                mol_anchor = mol_pool.detach().mean(dim=0)
            pocket_interact_feat = self._tmi_anchor_feat(
                poc_tok, poc_valid, mol_anchor.to(poc_tok.dtype), mode
            )
            return mol_interact_feat, pocket_interact_feat

        # ---- original, batch-conditioned formulation (Eq. 4-5) ----
        # [B_mol, D] and [B_pocket, D]
        return interact_feat.mean(dim=1), interact_feat.mean(dim=0)

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

        # # 原版5D：逐特征维做 token×token 交互，再对 Lm/Lp 求均值得到pairwise 交互特征
        # ------------------------------------------------------------------
        # Token-Level Mutual Interaction (TMI)
        # ------------------------------------------------------------------
        mol_tok_src = (
            mol_src_tokens_ori if mol_src_tokens_ori is not None else mol_src_tokens
        )
        poc_tok_src = (
            pocket_src_tokens_ori
            if pocket_src_tokens_ori is not None
            else pocket_src_tokens
        )
        if mol_rep_ori is None:
            mol_rep_ori = self._encode_frozen(
                mol_tok_src,
                mol_src_distance_ori
                if mol_src_distance_ori is not None
                else mol_src_distance,
                mol_src_edge_type_ori
                if mol_src_edge_type_ori is not None
                else mol_src_edge_type,
                "mol",
            )
        if pocket_rep_ori is None:
            pocket_rep_ori = self._encode_frozen(
                poc_tok_src,
                pocket_src_distance_ori
                if pocket_src_distance_ori is not None
                else pocket_src_distance,
                pocket_src_edge_type_ori
                if pocket_src_edge_type_ori is not None
                else pocket_src_edge_type,
                "pocket",
            )

        mol_interact_feat, pocket_interact_feat = self.compute_tmi(
            mol_rep_ori, pocket_rep_ori, mol_tok_src, poc_tok_src
        )
        # The TMI features gate the prompts of the tunable encoders, so their
        # batch dimension must match the corresponding encoder inputs.
        assert mol_interact_feat.size(0) == B_mol, (
            f"mol TMI feature batch {mol_interact_feat.size(0)} != mol batch {B_mol}; "
            "`mol_src_tokens_ori` must come from the same sample as `mol_src_tokens`"
        )
        assert pocket_interact_feat.size(0) == B_pocket, (
            f"pocket TMI feature batch {pocket_interact_feat.size(0)} != pocket batch "
            f"{B_pocket}; `pocket_src_tokens_ori` must come from the same sample as "
            "`pocket_src_tokens`"
        )
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

    # ---- Token-Level Mutual Interaction (TMI) options ----
    args.tmi_mode = getattr(args, "tmi_mode", "legacy")
    args.tmi_cond = getattr(args, "tmi_cond", "batch")
    args.tmi_affinity = getattr(args, "tmi_affinity", "cosine")
    # `cosine` logits live in [-1/tau, 1/tau]: tau=1 would be almost uniform, so
    # the default is a sharp temperature.  `dot` keeps the legacy tau=1.
    if getattr(args, "tmi_temperature", None) is None:
        args.tmi_temperature = 0.05 if args.tmi_affinity == "cosine" else 1.0
    args.tmi_anchor_momentum = getattr(args, "tmi_anchor_momentum", 0.1)
    args.tmi_gate_mode = getattr(args, "tmi_gate_mode", "legacy")
    args.tmi_alpha_beta = getattr(args, "tmi_alpha_beta", 1.0)
    args.tmi_inject_layers = getattr(args, "tmi_inject_layers", -1)
    args.tmi_alpha_lr_scale = getattr(args, "tmi_alpha_lr_scale", 1.0)
    args.tmi_diag = getattr(args, "tmi_diag", False)
    args.tmi_diag_interval = getattr(args, "tmi_diag_interval", 200)

    base_architecture(args)




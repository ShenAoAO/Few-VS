"""Numerical check of the algebraic claims about the TMI module.

(1) The original TMI formulation (Eq. 2-3 of the manuscript), i.e. the uniform
    average of the element-wise products over *all* molecule-pocket token pairs,
    is exactly equal to the Hadamard product of the two mean-pooled token
    representations.  The resulting [B_mol, B_pocket] matrix is therefore rank-1
    along every feature dimension, i.e. it carries no atom-pair correspondence.

(2) The revised TMI (``--tmi-mode pairwise``) replaces the uniform average by an
    affinity-based softmax weighting over token pairs.  The efficient einsum
    implementation used in the model matches the explicit 5-D reference exactly,
    and the resulting matrix is *not* rank-1, i.e. it cannot be factorised into
    a product of mean-pooled representations.

Run:  python script/tmi_equivalence_check.py
"""

import math

import torch


def uniform_tmi_reference(mol, poc):
    """Eq. (2)-(3): mean over (Lm, Lp) of the 5-D interaction tensor."""
    inter = mol.unsqueeze(1).unsqueeze(3) * poc.unsqueeze(0).unsqueeze(2)
    return inter.mean(dim=(2, 3))


def masked_mean(x, valid):
    w = valid.to(x.dtype).unsqueeze(-1)
    return (x * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)


def affinity_logits(mol, poc, affinity="cosine", tau=0.05):
    """Same affinity as the model (`--tmi-affinity`, `--tmi-temperature`)."""
    if affinity == "cosine":
        m = torch.nn.functional.normalize(mol, dim=-1, eps=1e-6)
        p = torch.nn.functional.normalize(poc, dim=-1, eps=1e-6)
        return torch.einsum("iad,jbd->ijab", m, p) / tau
    d = mol.size(-1)
    return torch.einsum("iad,jbd->ijab", mol, poc) / (tau * math.sqrt(d))


def pairwise_weights(mol, poc, mol_valid, poc_valid, affinity="cosine", tau=0.05):
    logits = affinity_logits(mol, poc, affinity, tau)
    pair_valid = mol_valid[:, None, :, None] & poc_valid[None, :, None, :]
    logits = logits.masked_fill(~pair_valid, float("-inf"))
    shp = logits.shape
    return torch.softmax(logits.reshape(shp[0], shp[1], -1), dim=-1).reshape(shp)


def pairwise_tmi_reference(mol, poc, mol_valid, poc_valid, affinity="cosine", tau=0.05):
    w = pairwise_weights(mol, poc, mol_valid, poc_valid, affinity, tau)
    inter = mol.unsqueeze(1).unsqueeze(3) * poc.unsqueeze(0).unsqueeze(2)
    return (w.unsqueeze(-1) * inter).sum(dim=(2, 3)), w


def pairwise_tmi_efficient(mol, poc, mol_valid, poc_valid, affinity="cosine", tau=0.05):
    """Implementation used in the model: never materialises the 5-D tensor."""
    w = pairwise_weights(mol, poc, mol_valid, poc_valid, affinity, tau)
    ctx = torch.einsum("ijab,jbd->ijad", w, poc)
    return torch.einsum("ijad,iad->ijd", ctx, mol)


def synthetic_tokens(b, l, d, shared_norm=5.95, spread_norm=7.53):
    """Tokens with the same first/second order statistics as the real UniMol
    hidden states used by the TMI (measured on `embeddings/dude/*.pockets`:
    token norm ~9.6, mean-pooled norm ~5.9, centred token norm ~7.5)."""
    shared = torch.randn(b, 1, d, dtype=torch.float64)
    shared = shared / shared.norm(dim=-1, keepdim=True) * shared_norm
    noise = torch.randn(b, l, d, dtype=torch.float64)
    noise = noise / noise.norm(dim=-1, keepdim=True) * spread_norm
    return shared + noise


def main():
    torch.manual_seed(0)
    b_mol, b_poc, l_mol, l_poc, d = 5, 4, 23, 61, 64
    mol = synthetic_tokens(b_mol, l_mol, d)
    poc = synthetic_tokens(b_poc, l_poc, d)
    mol_valid = torch.ones(b_mol, l_mol, dtype=torch.bool)
    poc_valid = torch.ones(b_poc, l_poc, dtype=torch.bool)
    mol_valid[0, 17:] = False           # padding
    poc_valid[2, 40:] = False

    # (1) uniform TMI == mean-pool + Hadamard
    ref = uniform_tmi_reference(mol, poc)
    had = mol.mean(dim=1).unsqueeze(1) * poc.mean(dim=1).unsqueeze(0)
    print("[1] max |TMI(Eq.2-3) - meanpool(x)meanpool| = %.3e" % (ref - had).abs().max())
    print("    per-dimension rank of the B_mol x B_pocket matrix : %d (rank-1 => no atom-pair information)"
          % torch.linalg.matrix_rank(ref[:, :, 0]).item())

    # (2) pairwise TMI: efficient == reference, and not rank-1
    had_masked = masked_mean(mol, mol_valid).unsqueeze(1) * masked_mean(poc, poc_valid).unsqueeze(0)
    ref_pw, w = pairwise_tmi_reference(mol, poc, mol_valid, poc_valid)
    eff_pw = pairwise_tmi_efficient(mol, poc, mol_valid, poc_valid)
    print("[2] max |pairwise(efficient) - pairwise(5-D reference)| = %.3e"
          % (ref_pw - eff_pw).abs().max())
    print("    relative distance to mean-pool + Hadamard = %.4f"
          % ((eff_pw - had_masked).norm() / had_masked.norm()).item())
    print("    per-dimension rank of the B_mol x B_pocket matrix : %d (> 1 => genuine token-pair information)"
          % torch.linalg.matrix_rank(eff_pw[:, :, 0]).item())

    # (3) the affinity must be sharp enough, otherwise the softmax degenerates
    #     into the uniform average and `pairwise` numerically collapses onto
    #     `legacy` / `hadamard` (this is what `--tmi-affinity`/`--tmi-temperature`
    #     control).
    print("[3] sharpness sweep (eff_pairs = 1/sum w^2, out of %d token pairs)"
          % (l_mol * l_poc))
    configs = [("dot", 1.0), ("dot", 0.1), ("cosine", 1.0), ("cosine", 0.1),
               ("cosine", 0.05), ("cosine", 0.02)]
    for affinity, tau in configs:
        feat = pairwise_tmi_efficient(mol, poc, mol_valid, poc_valid, affinity, tau)
        ww = pairwise_weights(mol, poc, mol_valid, poc_valid, affinity, tau)
        ww = ww.reshape(ww.size(0), ww.size(1), -1)
        eff = (1.0 / ww.pow(2).sum(-1)).mean().item()
        cos = torch.nn.functional.cosine_similarity(
            feat.reshape(-1, d), had_masked.reshape(-1, d), dim=-1
        ).mean().item()
        rel = ((feat - had_masked).norm() / had_masked.norm()).item()
        print("    affinity=%-6s tau=%-5.2f eff_pairs=%8.1f cos=%.6f rel_l2=%.4f"
              % (affinity, tau, eff, cos, rel))
    print("    (cos ~ 1 / rel_l2 ~ 0 means `pairwise` is indistinguishable from "
          "`legacy`)")


if __name__ == "__main__":
    main()

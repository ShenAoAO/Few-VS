"""Where does the TMI signal die?

Runs one real batch (DUDE / aa2ar) through the *whole* pipeline for several TMI
configurations and reports, stage by stage:

  1. fine_feat      : the TMI feature itself (raw cosine w.r.t. the legacy one and,
                      more importantly, the cosine *after removing the batch mean*
                      -- only the molecule-specific part can change a ranking)
  2. gate           : adapters[i](fine_feat) of the first prompted layer
                      (relative across-batch variation = how much the gate really
                      differentiates the molecules)
  3. mol_emb        : the projected embedding used for scoring
  4. scores/ranking : cosine similarity of the score vector and its Spearman
                      correlation with the legacy configuration

Usage (no training, evaluation-only, ~1 min):
    python script/tmi_sensitivity_check.py
"""

import os
import sys

import numpy as np
import torch
import unicore
from scipy.stats import spearmanr
from unicore import checkpoint_utils, options, tasks

ROOT = "/root/autodl-tmp/Few-VS"
TARGET = "aa2ar"
FT = 2
N_MOLS = 64

CONFIGS = [
    # mode, cond, affinity, tau, multiplier applied to the TMI feature
    ("legacy",   "batch", "dot",    1.0,   1.0),
    ("hadamard", "batch", "dot",    1.0,   1.0),
    ("pairwise", "batch", "dot",    1.0,   1.0),
    ("pairwise", "pair",  "dot",    1.0,   1.0),
    ("pairwise", "pair",  "cosine", 0.05,  1.0),
    ("pairwise", "pair",  "cosine", 0.02,  1.0),
    # sanity checks of the *gate path itself*: if killing / blowing up the TMI
    # feature does not move the embedding either, the prompt branch -- not the
    # TMI formulation -- is what makes the results insensitive.
    ("pairwise", "pair",  "cosine", 0.05,  0.0),
    ("pairwise", "pair",  "cosine", 0.05, 10.0),
    ("legacy",   "batch", "dot",    1.0,   0.0),
]


def build(argv):
    parser = options.get_validation_parser()
    parser.add_argument("--test-task", type=str, default="DUDE")
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser, input_args=argv)
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    model.load_state_dict(state["model"], strict=False)
    model = model.cuda().float().eval()
    model.freeze_tmi_encoders()
    model.reset_tmi_anchors()
    return args, task, model


def centred_cos(a, b):
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)
    return float(
        torch.nn.functional.cosine_similarity(a, b, dim=-1).mean()
    )


def main():
    argv = [
        os.path.join(ROOT, "data"),
        "--user-dir", os.path.join(ROOT, "unimol"),
        "--task", "drugclip", "--loss", "in_batch_softmax", "--arch", "fewshot",
        "--valid-subset", "test", "--path", os.path.join(ROOT, "checkpoint_best.pt"),
        "--max-pocket-atoms", "511", "--mol-token", "5", "--pocket-token", "3",
        "--ft", str(FT), "--sample-time", "1", "--batch-size", str(N_MOLS),
        "--no-tmi-emb-cache",          # encode with the frozen snapshot instead
    ]
    args, task, model = build(argv)

    mol_ds = task.load_mols_dataset_fewshot(
        f"{ROOT}/data/dude/raw/all/{TARGET}/mols_remain_{FT}pos_{FT}neg_1.lmdb",
        "atoms", "coordinates",
    )
    poc_ds = task.load_pockets_dataset_fewshot(
        f"{ROOT}/data/dude/raw/all/{TARGET}/pocket.lmdb"
    )
    sample = unicore.utils.move_to_cuda(
        mol_ds.collater([mol_ds[i] for i in range(N_MOLS)])
    )
    sample_pocket = unicore.utils.move_to_cuda(
        poc_ds.collater([poc_ds[i] for i in range(len(poc_ds))])
    )
    print(f"pockets in {TARGET}: {len(poc_ds)}  (cond=pair vs batch are identical "
          f"on the molecule branch when this is 1)")

    captured = {}
    orig_compute_tmi = model.compute_tmi
    scale = {"v": 1.0}

    def spy_compute_tmi(*a, **kw):
        out = orig_compute_tmi(*a, **kw)
        mol_feat = out[0] * scale["v"]
        captured["fine_feat"] = mol_feat.detach().float()
        return mol_feat, out[1] * scale["v"]

    model.compute_tmi = spy_compute_tmi
    model.mol_model.encoder.adapters[0].register_forward_hook(
        lambda m, i, o: captured.__setitem__("gate", o.detach().float())
    )
    # the prompt tokens sit at positions 1..P of the input of every layer
    model.mol_model.encoder.layers[0].register_forward_pre_hook(
        lambda m, i: captured.__setitem__("layer_in", i[0].detach().float())
    )

    ref = {}
    print("\n%-26s | %-17s | %-15s | %-17s | %s" % (
        "config", "fine_feat vs ref", "gate spread", "mol_emb vs ref", "scores vs ref"))
    print("%-26s | %-17s | %-15s | %-17s | %s" % (
        "mode/cond/affinity/tau", "cos   cos_centred", "std/|mean|", "cos   cos_centred",
        "cos    spearman"))
    print("-" * 108)

    for mode, cond, affinity, tau, mul in CONFIGS:
        model.args.tmi_mode = mode
        model.args.tmi_cond = cond
        model.args.tmi_affinity = affinity
        model.args.tmi_temperature = tau
        model.args.tmi_diag = False
        gate_mode = os.environ.get("TMI_GATE_MODE", "legacy")
        model.mol_model.encoder.tmi_gate_mode = gate_mode
        model.pocket_model.encoder.tmi_gate_mode = gate_mode
        scale["v"] = mul
        with torch.no_grad():
            mol_emb, pocket_emb = model(
                mol_src_tokens=sample["net_input"]["mol_src_tokens"],
                mol_src_distance=sample["net_input"]["mol_src_distance"],
                mol_src_edge_type=sample["net_input"]["mol_src_edge_type"],
                pocket_src_tokens=sample_pocket["net_input"]["pocket_src_tokens"],
                pocket_src_distance=sample_pocket["net_input"]["pocket_src_distance"],
                pocket_src_edge_type=sample_pocket["net_input"]["pocket_src_edge_type"],
                inference=True,
                **task._tmi_kwargs(sample, sample_pocket, None),
            )
            mol_emb = torch.nn.functional.normalize(mol_emb.float(), dim=-1)
            pocket_emb = torch.nn.functional.normalize(pocket_emb.float(), dim=-1)
            scores = (pocket_emb @ mol_emb.T).max(dim=0)[0]

        feat = captured["fine_feat"]
        gate = captured["gate"]
        li = captured["layer_in"]
        p = model.args.mol_token
        prompt_norm = float(li[:, 1:1 + p].norm(dim=-1).mean())
        atom_norm = float(li[:, 1 + p:].norm(dim=-1).mean())
        cls_norm = float(li[:, 0].norm(dim=-1).mean())
        gate_spread = float(
            gate.std(dim=0).mean() / gate.mean(dim=0).abs().mean().clamp(min=1e-12)
        )
        tag = f"{mode}/{cond}/{affinity}/{tau:g}" + (f" x{mul:g}" if mul != 1.0 else "")
        stage = ("      |gate|=%.3e  |prompt tok|=%.3e  vs |atom tok|=%.3e  |CLS|=%.3e"
                 % (float(gate.abs().mean()), prompt_norm, atom_norm, cls_norm))
        if not ref:
            ref = {"feat": feat, "emb": mol_emb, "score": scores}
            print("%-26s | %-17s | %-15.4f | %-17s | %s"
                  % (tag + "  (ref)", "-", gate_spread, "-", "-"))
            print(stage)
            continue
        fc = float(torch.nn.functional.cosine_similarity(feat, ref["feat"], dim=-1).mean())
        ec = float(torch.nn.functional.cosine_similarity(mol_emb, ref["emb"], dim=-1).mean())
        sc = float(torch.nn.functional.cosine_similarity(
            scores.unsqueeze(0), ref["score"].unsqueeze(0), dim=-1))
        rho = spearmanr(scores.cpu().numpy(), ref["score"].cpu().numpy()).correlation
        print("%-26s | %.4f  %8.4f | %-15.4f | %.4f  %8.4f | %.6f  %.6f"
              % (tag, fc, centred_cos(feat, ref["feat"]), gate_spread,
                 ec, centred_cos(mol_emb, ref["emb"]), sc, rho))
        print(stage)


if __name__ == "__main__":
    main()

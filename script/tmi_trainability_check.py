"""Does the few-shot adaptation actually train the prompt/adapter branch?

The prompt / adapter parameters are new (they are absent from the DrugCLIP
checkpoint by design) and are meant to be learned on the support set.  This
script checks whether that learning actually happens, by reproducing exactly the
few-shot loop of `test_dude_target` and reporting, per parameter group:

  * ||dtheta|| / ||theta||        : how much the support set moved the weights
  * ||grad||                     : the gradient magnitude that reaches them
  * |prompt token| vs |atom token| after adaptation
  * scores(fine_feat) vs scores(fine_feat = 0) after adaptation
    -> a Spearman of ~1.0 means the TMI has no influence on the ranking even
       after the support-set adaptation, i.e. the interaction branch is inert.

Run:python script/tmi_trainability_check.py           # gate=legacy (paper code)
      TMI_GATE_MODE=inject python script/tmi_trainability_check.py
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
import unicore
from scipy.stats import spearmanr
from unicore import checkpoint_utils, options, tasks
from unicore.modules import LayerNorm

ROOT = "/root/autodl-tmp/Few-VS"
TARGET = "aa2ar"
FT = 2
EPOCHS = 10
LR = 0.01
N_EVAL = 64

GROUPS = ["adapters", "deep_prompt_embeddings", "prompt_proj",
          "prompt_gate_norm", "prompt_out_norm", "tmi_alpha"]


def group_of(name):
    for g in GROUPS:
        if g in name:
            return g
    return "layer_norm/project"


def main():
    gate_mode = os.environ.get("TMI_GATE_MODE", "legacy")
    argv = [
        os.path.join(ROOT, "data"),
        "--user-dir", os.path.join(ROOT, "unimol"),
        "--task", "drugclip", "--loss", "in_batch_softmax", "--arch", "fewshot",
        "--valid-subset", "test", "--path", os.path.join(ROOT, "checkpoint_best.pt"),
        "--max-pocket-atoms", "511", "--mol-token", "5", "--pocket-token", "3",
        "--ft", str(FT), "--sample-time", "1", "--batch-size", str(N_EVAL),
        "--lr", str(LR), "--epoch-train", str(EPOCHS),
        "--tmi-mode", "pairwise", "--tmi-cond", "pair",
        "--tmi-affinity", "cosine", "--tmi-temperature", "0.05",
        "--tmi-gate-mode", gate_mode,
        "--no-tmi-emb-cache",
    ]
    parser = options.get_validation_parser()
    parser.add_argument("--test-task", type=str, default="DUDE")
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser, input_args=argv)

    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    missing = model.load_state_dict(state["model"], strict=False)
    n_prompt_missing = sum(
        1 for k in missing.missing_keys if any(g in k for g in GROUPS)
    )
    model = model.cuda().float()
    model.freeze_tmi_encoders()
    model.reset_tmi_anchors()
    params, names = task.collect_params(model)
    print(f"gate_mode={gate_mode}  prompt/adapter keys missing from the ckpt: "
          f"{n_prompt_missing} (randomly initialised, to be learned on the support set)")
    print(f"trainable tensors: {len(params)}")

    theta0 = [p.detach().clone() for p in params]
    grad_acc = {}

    mol_train = task.load_mols_dataset_fewshot(
        f"{ROOT}/data/dude/raw/all/{TARGET}/mols_few_{FT}pos_{FT}neg_1.lmdb",
        "atoms", "coordinates")
    mol_test = task.load_mols_dataset_fewshot(
        f"{ROOT}/data/dude/raw/all/{TARGET}/mols_remain_{FT}pos_{FT}neg_1.lmdb",
        "atoms", "coordinates")
    poc_ds = task.load_pockets_dataset_fewshot(
        f"{ROOT}/data/dude/raw/all/{TARGET}/pocket.lmdb")

    train_loader = torch.utils.data.DataLoader(
        mol_train, batch_size=5, collate_fn=mol_train.collater, shuffle=True,
        num_workers=0)
    poc_loader = torch.utils.data.DataLoader(
        poc_ds, batch_size=task._pocket_batch_size(), collate_fn=poc_ds.collater,
        shuffle=False, num_workers=0)

    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9)
    model.train()
    for _ in range(EPOCHS):
        for sample_pocket in poc_loader:
            sample_pocket = unicore.utils.move_to_cuda(sample_pocket)
            for sample in train_loader:
                sample = unicore.utils.move_to_cuda(sample)
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
                loss = task.compute_classification_loss(
                    F.normalize(mol_emb, dim=-1),
                    F.normalize(pocket_emb, dim=-1),
                    sample["target"],
                )
                loss.backward()
                for n, p in zip(names, params):
                    if p.grad is not None:
                        g = group_of(n)
                        grad_acc[g] = grad_acc.get(g, 0.0) + float(p.grad.norm())
                optimizer.step()
                optimizer.zero_grad()
    model.eval()

    print("\n%-26s | %8s | %-12s | %s" % (
        "parameter group", "#tensors", "|dtheta|/|theta|", "sum |grad|"))
    print("-" * 72)
    agg = {}
    for n, p, p0 in zip(names, params, theta0):
        g = group_of(n)
        agg.setdefault(g, [0.0, 0.0, 0])
        agg[g][0] += float((p.detach() - p0).norm() ** 2)
        agg[g][1] += float(p0.norm() ** 2)
        agg[g][2] += 1
    for g, (dn, tn, c) in agg.items():
        rel = (dn ** 0.5) / max(tn ** 0.5, 1e-12)
        print("%-26s | %8d | %-16.3e | %.3e"
              % (g, c, rel, grad_acc.get(g, 0.0)))

    # ---- post-adaptation sensitivity: does the TMI still matter? ----
    sample = unicore.utils.move_to_cuda(
        mol_test.collater([mol_test[i] for i in range(N_EVAL)]))
    sample_pocket = unicore.utils.move_to_cuda(
        poc_ds.collater([poc_ds[i] for i in range(len(poc_ds))]))
    captured = {}
    orig = model.compute_tmi
    scale = {"v": 1.0}

    def spy(*a, **kw):
        out = orig(*a, **kw)
        return out[0] * scale["v"], out[1] * scale["v"]

    model.compute_tmi = spy
    model.mol_model.encoder.layers[0].register_forward_pre_hook(
        lambda m, i: captured.__setitem__("layer_in", i[0].detach().float()))

    res = {}
    for tag, mul in [("fine_feat", 1.0), ("fine_feat=0", 0.0)]:
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
            m = F.normalize(mol_emb.float(), dim=-1)
            p = F.normalize(pocket_emb.float(), dim=-1)
            res[tag] = (p @ m.T).max(dim=0)[0].cpu().numpy()
        li = captured["layer_in"]
        pt = args.mol_token
        print("after adaptation (%-11s): |prompt tok|=%.3e  |atom tok|=%.3e"
              % (tag, float(li[:, 1:1 + pt].norm(dim=-1).mean()),
                 float(li[:, 1 + pt:].norm(dim=-1).mean())))

    rho = spearmanr(res["fine_feat"], res["fine_feat=0"]).correlation
    dmax = float(np.abs(res["fine_feat"] - res["fine_feat=0"]).max())
    print("\nTMI on vs off, AFTER support-set adaptation: spearman=%.6f  max|dscore|=%.3e"
          % (rho, dmax))
    print("-> spearman ~ 1.0 means the interaction branch is still inert, i.e. the "
          "support set did NOT learn to use the TMI.")
    for tag, enc in (("mol", model.mol_model.encoder),
                     ("pocket", model.pocket_model.encoder)):
        a = enc.tmi_alpha.detach().float().cpu().numpy()
        print("learned alpha (%s, %d layers): mean=%+.4f  max|.|=%.4f  %s"
              % (tag, len(a), a.mean(), np.abs(a).max(),
                 np.array2string(a[:6], precision=4, floatmode="fixed")))


if __name__ == "__main__":
    main()

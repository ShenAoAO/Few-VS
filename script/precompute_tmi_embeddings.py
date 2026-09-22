#!/usr/bin/env python3 -u
"""Pre-compute the frozen token representations used as TMI input.

This restores the legacy pipeline, which fed the Token-Level Mutual Interaction
with representations produced *offline* by the frozen pre-trained encoders
(Eq. 1 of the manuscript) instead of re-encoding them with the (partially
tuned) live encoders.

Output layout -- identical to the legacy one::

    <out-dir>/<dude|pcba>/<target>.npz.pkl.gz          {smi: np.ndarray[L, D]}
    <out-dir>/<dude|pcba>/<target>.pockets.npz.pkl.gz  {pocket_name: np.ndarray[L, D]}

`L` is the length of the *original* (prompt-free) token sequence, BOS/EOS
included, exactly what the task expects when it aligns the cached rows to the
current batch.

Examples
--------
# pockets + all few-shot support molecules (small, seconds/minutes)
python ./script/precompute_tmi_embeddings.py ./data --user-dir ./unimol \
       --task drugclip --loss in_batch_softmax --arch fewshot \
       --path checkpoint_best.pt --valid-subset test \
       --max-pocket-atoms 511 --mol-token 5 --pocket-token 3 \
       --test-task DUDE --subset few

# also cache the full screening library (WARNING: hundreds of GB, see --dry-run)
python ./script/precompute_tmi_embeddings.py ... --subset all --dry-run
"""

import argparse
import gzip
import logging
import os
import pickle
import sys

import numpy as np
import torch
from unicore import checkpoint_utils, distributed_utils, options, tasks
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.precompute_tmi")

DUDE_DIR = "/root/autodl-tmp/Few-VS/data/dude/raw/all"
PCBA_DIR = "/root/autodl-tmp/Few-VS/data/lit_pcba"


def _targets(task_name, args):
    base = DUDE_DIR if task_name == "dude" else PCBA_DIR
    if args.targets:
        names = [t for t in str(args.targets).split(",") if t.strip()]
    else:
        names = sorted(
            t for t in os.listdir(base) if os.path.isdir(os.path.join(base, t))
        )
    return base, names


def _mol_lmdbs(base, target, task_name, subset, fts, sample_times):
    """Molecule lmdb files that have to be encoded for one target.

    The cache is keyed by SMILES, so the different (ft, sample_time) splits of
    the *same* library would be encoded over and over again.  All `mols_few_*`
    files are tiny and are therefore all included, while a single `mols_remain_*`
    file is enough to cover the rest of the library.
    """
    paths = []
    for ft in fts:
        for st in sample_times:
            suffix = f"_{st}" if st in (1, 2) else ""
            few = os.path.join(
                base, target, f"mols_few_{ft}pos_{ft}neg{suffix}.lmdb"
            )
            if os.path.exists(few):
                paths.append(few)
    if subset == "all":
        for ft in fts:
            for st in sample_times:
                suffix = f"_{st}" if st in (1, 2) else ""
                remain = os.path.join(
                    base, target, f"mols_remain_{ft}pos_{ft}neg{suffix}.lmdb"
                )
                if os.path.exists(remain):
                    paths.append(remain)
                    return paths
    return paths


def _pocket_lmdb(base, target, task_name):
    name = "pocket.lmdb" if task_name == "dude" else "pockets.lmdb"
    path = os.path.join(base, target, name)
    return path if os.path.exists(path) else None


@torch.no_grad()
def _encode(model, dataset, flag, dtype, batch_size, desc):
    """Encode a dataset with the frozen encoder; returns {name: [L, D] array}."""
    import unicore

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collater,
        shuffle=False,
        num_workers=0,
    )
    if flag == "mol":
        tok_key, dist_key, et_key = (
            "mol_src_tokens_ori",
            "mol_src_distance_ori",
            "mol_src_edge_type_ori",
        )
        name_key = "smi_name"
        pad_idx = model.mol_model.padding_idx
    else:
        tok_key, dist_key, et_key = (
            "pocket_src_tokens_ori",
            "pocket_src_distance_ori",
            "pocket_src_edge_type_ori",
        )
        name_key = "pocket_name"
        pad_idx = model.pocket_model.padding_idx

    out = {}
    for sample in tqdm(loader, desc=desc, leave=False):
        sample = unicore.utils.move_to_cuda(sample)
        tokens = sample["net_input"][tok_key]
        rep = model._encode_frozen(
            tokens,
            sample["net_input"][dist_key],
            sample["net_input"][et_key],
            flag,
        )
        lengths = tokens.ne(pad_idx).sum(dim=1).tolist()
        rep = rep.float().cpu().numpy()
        for i, name in enumerate(sample[name_key]):
            out[name] = rep[i, : int(lengths[i])].astype(dtype)
    return out


def _dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wb", compresslevel=4) as f:
        pickle.dump(obj, f, protocol=4)
    os.replace(tmp, path)
    return os.path.getsize(path)


def main(args):
    torch.cuda.set_device(args.device_id)
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    model.load_state_dict(state["model"], strict=False)
    model = model.cuda().float().eval()
    # the snapshot *is* the pre-trained encoder here (no adaptation happened yet)
    if hasattr(model, "freeze_tmi_encoders"):
        model.freeze_tmi_encoders()

    dtype = np.float16 if args.dtype == "float16" else np.float32
    task_name = "dude" if args.test_task == "DUDE" else "pcba"
    base, targets = _targets(task_name, args)
    fts = [int(x) for x in str(args.fts).split(",") if x.strip()]
    sts = [int(x) for x in str(args.sample_times).split(",") if x.strip()]
    out_dir = os.path.join(args.out_dir, task_name)

    total_bytes = 0
    for target in tqdm(targets, desc=f"{task_name} targets"):
        mol_out = os.path.join(out_dir, f"{target}.npz.pkl.gz")
        poc_out = os.path.join(out_dir, f"{target}.pockets.npz.pkl.gz")

        # ---- pockets ----
        if not (args.skip_existing and os.path.exists(poc_out)):
            poc_path = _pocket_lmdb(base, target, task_name)
            if poc_path is not None:
                ds = task.load_pockets_dataset_fewshot(poc_path)
                if args.dry_run:
                    logger.info(f"[dry-run] {target}: {len(ds)} pockets")
                else:
                    reps = _encode(
                        model, ds, "pocket", dtype, args.batch, f"{target}/pockets"
                    )
                    total_bytes += _dump(reps, poc_out)

        # ---- molecules ----
        if args.skip_existing and os.path.exists(mol_out):
            continue
        paths = _mol_lmdbs(base, target, task_name, args.subset, fts, sts)
        if not paths:
            logger.warning(f"{target}: no molecule lmdb found, skipped")
            continue
        reps = {}
        n_mols = 0
        for path in paths:
            ds = task.load_mols_dataset_fewshot(path, "atoms", "coordinates")
            n_mols += len(ds)
            if args.dry_run:
                continue
            reps.update(
                _encode(
                    model, ds, "mol", dtype, args.batch,
                    f"{target}/{os.path.basename(path)}",
                )
            )
        if args.dry_run:
            per_mol = 26 * 512 * (2 if dtype is np.float16 else 4)
            logger.info(
                f"[dry-run] {target}: {n_mols} molecules "
                f"(~{n_mols * per_mol / 2**30:.2f} GiB uncompressed)"
            )
            continue
        total_bytes += _dump(reps, mol_out)

    if not args.dry_run:
        logger.info(
            f"done, wrote {total_bytes / 2**20:.1f} MiB to {out_dir}"
        )


def cli_main():
    parser = options.get_validation_parser()
    parser.add_argument("--test-task", type=str, default="DUDE",
                choices=["DUDE", "PCBA"])
    parser.add_argument("--out-dir", type=str,
                        default="/root/autodl-tmp/Few-VS/embeddings")
    parser.add_argument("--subset", type=str, default="few",
                        choices=["few", "all"],
                        help="`few`: support molecules only (tiny); "
                             "`all`: also the full screening library (huge)")
    parser.add_argument("--fts", type=str, default="2,4,8,16",
                        help="comma separated few-shot sizes to cover")
    parser.add_argument("--sample-times", type=str, default="1,2,3",
                        help="comma separated sampling repetitions to cover")
    parser.add_argument("--targets", type=str, default=None,
                        help="comma separated subset of targets (default: all)")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32"])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="only report the number of entries / disk estimate")
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()

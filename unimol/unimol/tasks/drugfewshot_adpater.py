# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time

from IPython import embed as debug_embedded
import logging
import os
import gzip
import random
import pandas as pd
from collections.abc import Iterable
from sklearn.metrics import roc_auc_score
from xmlrpc.client import Boolean
import numpy as np
import torch
import pickle
from tqdm import tqdm
from unicore import checkpoint_utils
import unicore
from unicore.data import (AppendTokenDataset, Dictionary, EpochShuffleDataset,
                          FromNumpyDataset, NestedDictionaryDataset,
                          PrependTokenDataset, RawArrayDataset, LMDBDataset, RawLabelDataset,
                          RightPadDataset, RightPadDataset2D, TokenizeDataset, SortDataset, data_utils)
from unicore.tasks import UnicoreTask, register_task
from unimol.data import (AffinityDataset, CroppingPocketDataset,
                         CrossDistanceDataset, DistanceDataset,
                         EdgeTypeDataset, KeyDataset, LengthDataset,
                         NormalizeDataset, NormalizeDockingPoseDataset,
                         PrependAndAppend2DDataset, RemoveHydrogenDataset,
                         RemoveHydrogenPocketDataset, RightPadDatasetCoord,
                         RightPadDatasetCross2D, TTADockingPoseDataset, AffinityTestDataset, AffinityValidDataset,
                         AffinityMolDataset, AffinityPocketDataset, ResamplingDataset,AffinityMolDataset_fewshot,AffinityPocketDataset_fewshot)
# from skchem.metrics import bedroc_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC, CalcEnrichment
from sklearn.metrics import roc_curve
from unicore.modules.layer_norm import LayerNorm
import torch.nn.functional as F
logger = logging.getLogger(__name__)
from torch import nn
import math
from datetime import datetime

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
os.environ.setdefault("PYTHONHASHSEED", "42")
try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except Exception:
    pass

GLOBAL_SEED = int(os.environ.get("SEED", "42"))
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

def _global_seed_worker(worker_id):
    worker_seed = GLOBAL_SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

GLOBAL_GENERATOR = torch.Generator()
GLOBAL_GENERATOR.manual_seed(GLOBAL_SEED)




def re_new(y_true, y_score, ratio):
    fp = 0
    tp = 0
    p = sum(y_true)
    n = len(y_true) - p
    num = ratio * n
    sort_index = np.argsort(y_score)[::-1]
    for i in range(len(sort_index)):
        index = sort_index[i]
        if y_true[index] == 1:
            tp += 1
        else:
            fp += 1
            if fp >= num:
                break
    if p*fp==0:
        print()
    return (tp * n) / (p * fp)


def calc_re(y_true, y_score, ratio_list):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    # print(fpr, tpr)
    res = {}
    res2 = {}
    total_active_compounds = sum(y_true)
    total_compounds = len(y_true)

    # for ratio in ratio_list:
    #     for i, t in enumerate(fpr):
    #         if t > ratio:
    #             #print(fpr[i], tpr[i])
    #             if fpr[i-1]==0:
    #                 res[str(ratio)]=tpr[i]/fpr[i]
    #             else:
    #                 res[str(ratio)]=tpr[i-1]/fpr[i-1]
    #             break
    # for ratio in ratio_list:
    #     for i, t in enumerate(fpr):
    #         if t > ratio:
    #             if i > 0 and fpr[i - 1] != 0:
    #                 res2[str(ratio)] = tpr[i - 1] / fpr[i - 1]  # 防止fpr[i-1]为0
    #             elif fpr[i - 1] == 0:
    #                 res2[str(ratio)] = tpr[i] / fpr[i]  # 如果之前fpr为零，可以直接用当前值
    #             break

    for ratio in ratio_list:
        res2[str(ratio)] = re_new(y_true, y_score, ratio)



    # print(res)
    # print(res2)
    return res2


def cal_metrics(y_true, y_score, alpha):
    """
    Calculate BEDROC score.

    Parameters:
    - y_true: true binary labels (0 or 1)
    - y_score: predicted scores or probabilities
    - alpha: parameter controlling the degree of early retrieval emphasis

    Returns:
    - BEDROC score
    """

    # concate res_single and labels
    scores = np.expand_dims(y_score, axis=1)
    y_true = np.expand_dims(y_true, axis=1)
    scores = np.concatenate((scores, y_true), axis=1)
    # inverse sort scores based on first column
    scores = scores[scores[:, 0].argsort()[::-1]]
    bedroc = CalcBEDROC(scores, 1, 80.5)
    count = 0
    # sort y_score, return index
    index = np.argsort(y_score)[::-1]
    for i in range(int(len(index) * 0.005)):
        if y_true[index[i]] == 1:
            count += 1
    auc = CalcAUC(scores, 1)
    ef_list = CalcEnrichment(scores, 1, [0.005, 0.01, 0.02, 0.05])
    ef = {
        "0.005": ef_list[0],
        "0.01": ef_list[1],
        "0.02": ef_list[2],
        "0.05": ef_list[3]
    }
    re_list = calc_re(y_true, y_score, [0.005, 0.01, 0.02, 0.05])
    return auc, bedroc, ef, re_list


@register_task("drugclip")
class DrugCLIP(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="downstream data path",
        )
        parser.add_argument(
            "--finetune-mol-model",
            default=None,
            type=str,
            help="pretrained molecular model path",
        )
        parser.add_argument(
            "--finetune-pocket-model",
            default=None,
            type=str,
            help="pretrained pocket model path",
        )
        parser.add_argument(
            "--dist-threshold",
            type=float,
            default=6.0,
            help="threshold for the distance between the molecule and the pocket",
        )
        parser.add_argument(
            "--max-pocket-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a pocket",
        )
        parser.add_argument(
            "--test-model",
            default=False,
            type=Boolean,
            help="whether test model",
        )
        parser.add_argument("--prompt-tokens", default=5, type=int, help="vpt")

        parser.add_argument("--reg", action="store_true", help="regression task")
        parser.add_argument("--ft",default=2, help="few shot pairs",type=int)
        parser.add_argument("--lr",default=0.001, help="learning rate",type=float)
        parser.add_argument("--epoch-train",default=30, help="epoch train",type=int)
        parser.add_argument("--zero-shot", action="store_true", default=False,
                            help="evaluate the frozen base encoders without few-shot adaptation")
        # parser.add_argument("--epoch-cls",default=20, help="epoch cls",type=int)
        parser.add_argument("--sample-time", default=1, type=int, help="sample_time")
        parser.add_argument("--pocket-token", default=5, type=int, help="pocket_token")
        parser.add_argument("--mol-token", default=5, type=int, help="mol_token")
        parser.add_argument("--bsz", default=32, type=int, help="test batch size")
        parser.add_argument(
            "--pocket-aggregate",
            default="max",
            choices=("max", "topk_mean", "mean", "logsumexp"),
            type=str,
            help="aggregation across PCBA pockets at screening time: max (legacy), "
                 "topk_mean (mean of the top --pocket-topk pockets), mean, or "
                 "logsumexp. DUD-E has one pocket, so all modes are equivalent.",
        )
        parser.add_argument(
            "--pocket-topk",
            default=2,
            type=int,
            help="number of highest-scoring pockets used by topk_mean; clipped to "
                 "the number of pockets",
        )
        parser.add_argument(
            "--pocket-score-norm",
            default="none",
            choices=("none", "zscore", "rank"),
            type=str,
            help="calibrate each pocket's scores over the whole screening "
                 "library before aggregation (label free). `max` over "
                 "uncalibrated pockets is dominated by pockets with a globally "
                 "higher score offset, which hurts early enrichment; `zscore` "
                 "removes the per-pocket mean/std and `rank` maps each pocket to "
                 "uniform [0,1] ranks. No effect with a single pocket (DUD-E).",
        )
        parser.add_argument(
            "--target-subset",
            default=None,
            type=str,
            help="restrict the evaluation to these targets: a comma separated "
                 "list of names, or a path to a text file with one name per "
                 "line. Use it to tune hyper-parameters on a development subset "
                 "instead of the reported benchmark.",
        )
        parser.add_argument(
            "--max-targets",
            default=0,
            type=int,
            help="evaluate at most N targets (0 = no limit); applied after "
                 "--target-subset",
        )
        parser.add_argument(
            "--keep-target-warm-start",
            action="store_true",
            help="keep adapted parameters and TMI anchors across targets; "
                 "this intentionally reproduces the historical target-order "
                 "dependent warm-start behavior",
        )
        # ---- batch-invariance diagnostic (reviewer comment #2) ----
        parser.add_argument(
            "--batch-invariance",
            action="store_true",
            help="after few-shot adaptation, re-score the same screening library "
                 "with several batch sizes / random batch partitions and report "
                 "Spearman correlation and top-ranked overlap",
        )
        parser.add_argument(
            "--bi-batch-sizes",
            default="8,16,32,64",
            type=str,
            help="comma separated evaluation batch sizes for the invariance test",
        )
        parser.add_argument(
            "--bi-repeats",
            default=2,
            type=int,
            help="number of random batch partitions per batch size",
        )
        parser.add_argument(
            "--bi-topk-frac",
            default=0.01,
            type=float,
            help="fraction of top-ranked candidates used for the overlap metric",
        )
        parser.add_argument(
            "--bi-max-mols",
            default=2000,
            type=int,
            help="sub-sample the screening library to at most this many molecules "
                 "for the invariance test (<=0 uses the full library)",
        )
        # ---- legacy behaviour: TMI is fed with pre-computed frozen embeddings ----
        parser.add_argument(
            "--tmi-emb-dir",
            default="/root/autodl-tmp/Few-VS/embeddings",
            type=str,
            help="directory holding the pre-computed frozen token representations "
                 "used as TMI input (`<dir>/<dude|pcba>/<target>.npz.pkl.gz` and "
                 "`<target>.pockets.npz.pkl.gz`, as in the legacy pipeline). "
                 "Molecules/pockets that are missing from the cache fall back to "
                 "an on-the-fly encoding with the frozen encoder snapshot.",
        )
        parser.add_argument(
            "--no-tmi-emb-cache",
            
            action="store_true",
            help="ignore the pre-computed embeddings and always encode the TMI "
                 "input on the fly with the frozen encoder snapshot",
        )



    def __init__(self, args, dictionary, pocket_dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.pocket_dictionary = pocket_dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        self.pocket_mask_idx = pocket_dictionary.add_symbol("[MASK]", is_special=True)
        self.mol_reps = None
        self.keys = None

    @classmethod
    def setup_task(cls, args, **kwargs):
        mol_dictionary = Dictionary.load(os.path.join(args.data, "dict_mol.txt"))
        pocket_dictionary = Dictionary.load(os.path.join(args.data, "dict_pkt.txt"))
        logger.info("ligand dictionary: {} types".format(len(mol_dictionary)))
        logger.info("pocket dictionary: {} types".format(len(pocket_dictionary)))
        return cls(args, mol_dictionary, pocket_dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        'smi','pocket','atoms','coordinates','pocket_atoms','pocket_coordinates'
        Args:
            split (str): name of the data scoure (e.g., bppp)
        """
        data_path = os.path.join(self.args.data, split + ".lmdb")
        dataset = LMDBDataset(data_path)
        if split.startswith("train"):
            smi_dataset = KeyDataset(dataset, "smi")
            poc_dataset = KeyDataset(dataset, "pocket")

            dataset = AffinityDataset(
                dataset,
                self.args.seed,
                "atoms",
                "coordinates",
                "pocket_atoms",
                "pocket_coordinates",
                "label",
                True,
            )
            tgt_dataset = KeyDataset(dataset, "affinity")

        else:

            dataset = AffinityDataset(
                dataset,
                self.args.seed,
                "atoms",
                "coordinates",
                "pocket_atoms",
                "pocket_coordinates",
                "label",
            )
            tgt_dataset = KeyDataset(dataset, "affinity")
            smi_dataset = KeyDataset(dataset, "smi")
            poc_dataset = KeyDataset(dataset, "pocket")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket_atoms",
            "pocket_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDataset(
            dataset,
            self.seed,
            "pocket_atoms",
            "pocket_coordinates",
            self.args.max_pocket_atoms,
        )

        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)

        apo_dataset = NormalizeDataset(dataset, "coordinates")
        apo_dataset = NormalizeDataset(apo_dataset, "pocket_coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        mol_len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)

        src_pocket_dataset = KeyDataset(apo_dataset, "pocket_atoms")
        pocket_len_dataset = LengthDataset(src_pocket_dataset)
        src_pocket_dataset = TokenizeDataset(
            src_pocket_dataset,
            self.pocket_dictionary,
            max_seq_len=self.args.max_seq_len,
        )
        coord_pocket_dataset = KeyDataset(apo_dataset, "pocket_coordinates")
        src_pocket_dataset = PrependAndAppend(
            src_pocket_dataset,
            self.pocket_dictionary.bos(),
            self.pocket_dictionary.eos(),
        )
        pocket_edge_type = EdgeTypeDataset(
            src_pocket_dataset, len(self.pocket_dictionary)
        )
        coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
        distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
        coord_pocket_dataset = PrependAndAppend(coord_pocket_dataset, 0.0, 0.0)
        distance_pocket_dataset = PrependAndAppend2DDataset(
            distance_pocket_dataset, 0.0
        )

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_tokens": RightPadDataset(
                        src_pocket_dataset,
                        pad_idx=self.pocket_dictionary.pad(),
                    ),
                    "pocket_src_distance": RightPadDataset2D(
                        distance_pocket_dataset,
                        pad_idx=0,
                    ),
                    "pocket_src_edge_type": RightPadDataset2D(
                        pocket_edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_coord": RightPadDatasetCoord(
                        coord_pocket_dataset,
                        pad_idx=0,
                    ),
                    "mol_len": RawArrayDataset(mol_len_dataset),
                    "pocket_len": RawArrayDataset(pocket_len_dataset)
                },
                "target": {
                    "finetune_target": RawLabelDataset(tgt_dataset),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                "pocket_name": RawArrayDataset(poc_dataset),
            },
        )
        if split == "train":
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(src_dataset))

            self.datasets[split] = SortDataset(
                nest_dataset,
                sort_order=[shuffle],
            )
            self.datasets[split] = ResamplingDataset(
                self.datasets[split]
            )
        else:
            self.datasets[split] = nest_dataset

    def load_mols_dataset(self, data_path, atoms, coords, **kwargs):

        dataset = LMDBDataset(data_path)
        label_dataset = KeyDataset(dataset, "label")
        dataset = AffinityMolDataset(
            dataset,
            self.args.seed,
            atoms,
            coords,
            False,
        )

        smi_dataset = KeyDataset(dataset, "smi")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)

        apo_dataset = NormalizeDataset(dataset, "coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                "target": RawArrayDataset(label_dataset),
                "mol_len": RawArrayDataset(len_dataset),
            },
        )
        return nest_dataset
    def load_mols_dataset_fewshot(self, data_path, atoms, coords, **kwargs):

        dataset = LMDBDataset(data_path)
        label_dataset = KeyDataset(dataset, "label")
        dataset = AffinityMolDataset_fewshot(
            dataset,
            self.args.seed,
            atoms,
            coords,
            False,
            token = self.args.mol_token
        )

        smi_dataset = KeyDataset(dataset, "smi")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)

        apo_dataset = NormalizeDataset(dataset, "coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)

        dataset = LMDBDataset(data_path)
        dataset_ori = AffinityMolDataset(
            dataset,
            self.args.seed,
            atoms,
            coords,
            False,
        )

        smi_dataset_ori = KeyDataset(dataset_ori, "smi")

        dataset_ori = RemoveHydrogenDataset(dataset_ori, "atoms", "coordinates", True, True)

        apo_dataset_ori = NormalizeDataset(dataset_ori, "coordinates")

        src_dataset_ori = KeyDataset(apo_dataset_ori, "atoms")
        len_dataset_ori = LengthDataset(src_dataset_ori)
        src_dataset_ori = TokenizeDataset(
            src_dataset_ori, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset_ori = KeyDataset(apo_dataset_ori, "coordinates")
        src_dataset_ori = PrependAndAppend(
            src_dataset_ori, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type_ori = EdgeTypeDataset(src_dataset_ori, len(self.dictionary))
        coord_dataset_ori = FromNumpyDataset(coord_dataset_ori)
        distance_dataset_ori = DistanceDataset(coord_dataset_ori)
        coord_dataset_ori = PrependAndAppend(coord_dataset_ori, 0.0, 0.0)
        distance_dataset_ori = PrependAndAppend2DDataset(distance_dataset_ori, 0.0)

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens_ori": RightPadDataset(
                        src_dataset_ori,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance_ori": RightPadDataset2D(
                        distance_dataset_ori,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type_ori": RightPadDataset2D(
                        edge_type_ori,
                        pad_idx=0,
                    ),
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                },
                "smi_name": RawArrayDataset(smi_dataset_ori),
                "target": RawArrayDataset(label_dataset),
                "mol_len": RawArrayDataset(len_dataset_ori),
            },
        )
        return nest_dataset

    def load_mols_dataset_fewshot_test(self, data_path, atoms, coords, **kwargs):

        dataset = LMDBDataset(data_path)
        label_dataset = KeyDataset(dataset, "label")
        dataset = AffinityMolDataset_fewshot(
            dataset,
            self.args.seed,
            atoms,
            coords,
            False,
        )

        smi_dataset = KeyDataset(dataset, "smi")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)

        apo_dataset = NormalizeDataset(dataset, "coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                "target": RawArrayDataset(label_dataset),
                "mol_len": RawArrayDataset(len_dataset),
            },
        )
        return nest_dataset

    def load_retrieval_mols_dataset(self, data_path, atoms, coords, **kwargs):

        dataset = LMDBDataset(data_path)
        dataset = AffinityMolDataset(
            dataset,
            self.args.seed,
            atoms,
            coords,
            False,
        )

        smi_dataset = KeyDataset(dataset, "smi")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)

        apo_dataset = NormalizeDataset(dataset, "coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                "mol_len": RawArrayDataset(len_dataset),
            },
        )
        return nest_dataset

    def load_pockets_dataset(self, data_path, **kwargs):

        dataset = LMDBDataset(data_path)

        dataset = AffinityPocketDataset(
            dataset,
            self.args.seed,
            "pocket_atoms",
            "pocket_coordinates",
            False,
            "pocket"
        )
        poc_dataset = KeyDataset(dataset, "pocket")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket_atoms",
            "pocket_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDataset(
            dataset,
            self.seed,
            "pocket_atoms",
            "pocket_coordinates",
            self.args.max_pocket_atoms,
        )

        apo_dataset = NormalizeDataset(dataset, "pocket_coordinates")

        src_pocket_dataset = KeyDataset(apo_dataset, "pocket_atoms")
        len_dataset = LengthDataset(src_pocket_dataset)
        src_pocket_dataset = TokenizeDataset(
            src_pocket_dataset,
            self.pocket_dictionary,
            max_seq_len=self.args.max_seq_len,
        )
        coord_pocket_dataset = KeyDataset(apo_dataset, "pocket_coordinates")
        src_pocket_dataset = PrependAndAppend(
            src_pocket_dataset,
            self.pocket_dictionary.bos(),
            self.pocket_dictionary.eos(),
        )
        pocket_edge_type = EdgeTypeDataset(
            src_pocket_dataset, len(self.pocket_dictionary)
        )
        coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
        distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
        coord_pocket_dataset = PrependAndAppend(coord_pocket_dataset, 0.0, 0.0)
        distance_pocket_dataset = PrependAndAppend2DDataset(
            distance_pocket_dataset, 0.0
        )

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "pocket_src_tokens": RightPadDataset(
                        src_pocket_dataset,
                        pad_idx=self.pocket_dictionary.pad(),
                    ),
                    "pocket_src_distance": RightPadDataset2D(
                        distance_pocket_dataset,
                        pad_idx=0,
                    ),
                    "pocket_src_edge_type": RightPadDataset2D(
                        pocket_edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_coord": RightPadDatasetCoord(
                        coord_pocket_dataset,
                        pad_idx=0,
                    ),
                },
                "pocket_name": RawArrayDataset(poc_dataset),
                "pocket_len": RawArrayDataset(len_dataset),
            },
        )
        return nest_dataset
    def load_pockets_dataset_fewshot(self, data_path, **kwargs):

        dataset = LMDBDataset(data_path)

        dataset = AffinityPocketDataset_fewshot(
            dataset,
            self.args.seed,
            "pocket_atoms",
            "pocket_coordinates",
            False,
            "pocket",
            token = self.args.pocket_token
        )
        poc_dataset = KeyDataset(dataset, "pocket")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket_atoms",
            "pocket_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDataset(
            dataset,
            self.seed,
            "pocket_atoms",
            "pocket_coordinates",
            self.args.max_pocket_atoms,
        )

        apo_dataset = NormalizeDataset(dataset, "pocket_coordinates")

        src_pocket_dataset = KeyDataset(apo_dataset, "pocket_atoms")
        len_dataset = LengthDataset(src_pocket_dataset)
        src_pocket_dataset = TokenizeDataset(
            src_pocket_dataset,
            self.pocket_dictionary,
            max_seq_len=self.args.max_seq_len,
        )
        coord_pocket_dataset = KeyDataset(apo_dataset, "pocket_coordinates")
        src_pocket_dataset = PrependAndAppend(
            src_pocket_dataset,
            self.pocket_dictionary.bos(),
            self.pocket_dictionary.eos(),
        )
        pocket_edge_type = EdgeTypeDataset(
            src_pocket_dataset, len(self.pocket_dictionary)
        )
        coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
        distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
        coord_pocket_dataset = PrependAndAppend(coord_pocket_dataset, 0.0, 0.0)
        distance_pocket_dataset = PrependAndAppend2DDataset(
            distance_pocket_dataset, 0.0
        )

        dataset = LMDBDataset(data_path)
        dataset_ori = AffinityPocketDataset(
            dataset,
            self.args.seed,
            "pocket_atoms",
            "pocket_coordinates",
            False,
            "pocket"
        )
        dataset_ori = RemoveHydrogenPocketDataset(
            dataset_ori,
            "pocket_atoms",
            "pocket_coordinates",
            True,
            True,
        )
        dataset_ori = CroppingPocketDataset(
            dataset_ori,
            self.seed,
            "pocket_atoms",
            "pocket_coordinates",
            self.args.max_pocket_atoms,
        )

        apo_dataset_ori = NormalizeDataset(dataset_ori, "pocket_coordinates")

        src_pocket_dataset_ori = KeyDataset(apo_dataset_ori, "pocket_atoms")
        len_dataset_ori = LengthDataset(src_pocket_dataset_ori)
        src_pocket_dataset_ori = TokenizeDataset(
            src_pocket_dataset_ori,
            self.pocket_dictionary,
            max_seq_len=self.args.max_seq_len,
        )
        coord_pocket_dataset_ori = KeyDataset(apo_dataset_ori, "pocket_coordinates")
        src_pocket_dataset_ori = PrependAndAppend(
            src_pocket_dataset_ori,
            self.pocket_dictionary.bos(),
            self.pocket_dictionary.eos(),
        )
        pocket_edge_type_ori = EdgeTypeDataset(
            src_pocket_dataset_ori, len(self.pocket_dictionary)
        )
        coord_pocket_dataset_ori = FromNumpyDataset(coord_pocket_dataset_ori)
        distance_pocket_dataset_ori = DistanceDataset(coord_pocket_dataset_ori)
        coord_pocket_dataset_ori = PrependAndAppend(coord_pocket_dataset_ori, 0.0, 0.0)
        distance_pocket_dataset_ori = PrependAndAppend2DDataset(
            distance_pocket_dataset_ori, 0.0
        )

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "pocket_src_tokens_ori": RightPadDataset(
                        src_pocket_dataset_ori,
                        pad_idx=self.pocket_dictionary.pad(),
                    ),
                    "pocket_src_distance_ori": RightPadDataset2D(
                        distance_pocket_dataset_ori,
                        pad_idx=0,
                    ),
                    "pocket_src_edge_type_ori": RightPadDataset2D(
                        pocket_edge_type_ori,
                        pad_idx=0,
                    ),
                    "pocket_src_coord_ori": RightPadDatasetCoord(
                        coord_pocket_dataset_ori,
                        pad_idx=0,
                    ),
                    "pocket_src_tokens": RightPadDataset(
                        src_pocket_dataset,
                        pad_idx=self.pocket_dictionary.pad(),
                    ),
                    "pocket_src_distance": RightPadDataset2D(
                        distance_pocket_dataset,
                        pad_idx=0,
                    ),
                    "pocket_src_edge_type": RightPadDataset2D(
                        pocket_edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_coord": RightPadDatasetCoord(
                        coord_pocket_dataset,
                        pad_idx=0,
                    ),
                },
                "pocket_name": RawArrayDataset(poc_dataset),
                "pocket_len": RawArrayDataset(len_dataset_ori),
            },
        )
        return nest_dataset

    def load_pockets_dataset_fewshot_test(self, data_path, **kwargs):

        dataset = LMDBDataset(data_path)

        dataset = AffinityPocketDataset_fewshot(
            dataset,
            self.args.seed,
            "pocket_atoms",
            "pocket_coordinates",
            False,
            "pocket"
        )
        poc_dataset = KeyDataset(dataset, "pocket")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket_atoms",
            "pocket_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDataset(
            dataset,
            self.seed,
            "pocket_atoms",
            "pocket_coordinates",
            self.args.max_pocket_atoms,
        )

        apo_dataset = NormalizeDataset(dataset, "pocket_coordinates")

        src_pocket_dataset = KeyDataset(apo_dataset, "pocket_atoms")
        len_dataset = LengthDataset(src_pocket_dataset)
        src_pocket_dataset = TokenizeDataset(
            src_pocket_dataset,
            self.pocket_dictionary,
            max_seq_len=self.args.max_seq_len,
        )
        coord_pocket_dataset = KeyDataset(apo_dataset, "pocket_coordinates")
        src_pocket_dataset = PrependAndAppend(
            src_pocket_dataset,
            self.pocket_dictionary.bos(),
            self.pocket_dictionary.eos(),
        )
        pocket_edge_type = EdgeTypeDataset(
            src_pocket_dataset, len(self.pocket_dictionary)
        )
        coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
        distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
        coord_pocket_dataset = PrependAndAppend(coord_pocket_dataset, 0.0, 0.0)
        distance_pocket_dataset = PrependAndAppend2DDataset(
            distance_pocket_dataset, 0.0
        )

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "pocket_src_tokens": RightPadDataset(
                        src_pocket_dataset,
                        pad_idx=self.pocket_dictionary.pad(),
                    ),
                    "pocket_src_distance": RightPadDataset2D(
                        distance_pocket_dataset,
                        pad_idx=0,
                    ),
                    "pocket_src_edge_type": RightPadDataset2D(
                        pocket_edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_coord": RightPadDatasetCoord(
                        coord_pocket_dataset,
                        pad_idx=0,
                    ),
                },
                "pocket_name": RawArrayDataset(poc_dataset),
                "pocket_len": RawArrayDataset(len_dataset),
            },
        )
        return nest_dataset


    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)

        state = checkpoint_utils.load_checkpoint_to_cpu('/root/autodl-tmp/Few-VS/checkpoint_best.pt')
        model.load_state_dict(state, strict=False)


        return model

    def train_step(
            self, sample, model, loss, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *loss*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~unicore.data.UnicoreDataset`.
            model (~unicore.models.BaseUnicoreModel): the model
            loss (~unicore.losses.UnicoreLoss): the loss
            optimizer (~unicore.optim.UnicoreOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """

        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = loss(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, loss, test=False):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = loss(model, sample)
        return loss, sample_size, logging_output

    def compute_classification_loss(self,mol_emb, pocket_emb, mol_targets):
        """计算正常分类loss"""

        sim_matrix = (pocket_emb @ mol_emb.T).T  # [B_mol, B_pocket]

        B_mol = mol_targets.size(0)
        B_pocket = pocket_emb.size(0)
        labels_matrix = torch.zeros(B_mol, B_pocket, device=mol_emb.device)

        for mol_idx in range(B_mol):
            if mol_targets[mol_idx] == 1:
                labels_matrix[mol_idx, :] = 1.0
            else:
                labels_matrix[mol_idx, :] = 0.0

        return F.binary_cross_entropy_with_logits(sim_matrix, labels_matrix)

    # ------------------------------------------------------------------
    # Batch-invariance diagnostic
    # ------------------------------------------------------------------
    # Legacy TMI input: pre-computed frozen token representations
    # ------------------------------------------------------------------
    def _load_rep_cache(self, tag, target):
        """Load `<emb_dir>/<tag>/<target>{,.pockets}.npz.pkl.gz`.

        Returns ``(mol_rep_map, pocket_rep_map)``; a missing/unreadable file
        yields ``None`` for the corresponding map, in which case the caller
        falls back to encoding with the frozen encoder snapshot.
        """
        if getattr(self.args, "no_tmi_emb_cache", False):
            return None, None
        base = os.path.join(getattr(self.args, "tmi_emb_dir", ""), tag)
        maps = []
        for name in (f"{target}.npz.pkl.gz", f"{target}.pockets.npz.pkl.gz"):
            path = os.path.join(base, name)
            try:
                with gzip.open(path, "rb") as f:
                    maps.append(pickle.load(f))
            except Exception:
                maps.append(None)
        if maps[0] is None and maps[1] is None:
            logger.info(f"[TMI] no embedding cache for {tag}/{target}, "
                        f"encoding on the fly")
        return maps[0], maps[1]

    @staticmethod
    def _align_rep(rep_map, names, target_length, device):
        """Legacy alignment of cached representations to the current batch.

        Rows are truncated to``target_length`` or padded by repeating the last
        row (exactly as in the original implementation), missing entries are
        filled with zeros.  Returns ``None`` when nothing can be resolved.
        """
        if rep_map is None:
            return None
        feats = []
        for name in names:
            arr = rep_map.get(name) if hasattr(rep_map, "get") else None
            if arr is None:
                if len(feats) > 0:
                    feats.append(torch.zeros_like(feats[0]))
                    continue
                return None
            arr_t = torch.as_tensor(np.asarray(arr), dtype=torch.float32)
            actual_length = arr_t.size(0)
            feature_dim = arr_t.size(1) if arr_t.dim() > 1 else 0
            if actual_length == 0:
                feats.append(
                    torch.zeros((target_length, feature_dim), dtype=torch.float32)
                )
            elif actual_length >= target_length:
                feats.append(arr_t[:target_length])
            else:
                last_row = arr_t[actual_length - 1: actual_length]
                pad_rows = last_row.expand(target_length - actual_length, -1).clone()
                feats.append(torch.cat([arr_t, pad_rows], dim=0))
        try:
            return torch.stack(feats, dim=0).to(device)
        except Exception:
            return None

    def _tmi_kwargs(self, sample, sample_pocket, rep_maps):
        """Build the TMI-related keyword arguments of ``model.forward``.

        Cached (pre-computed, frozen) representations are used when available --
        this is the legacy behaviour.  Only the sides that are missing from the
        cache fall back to the `*_ori` tensors, which the model then encodes with
        the frozen encoder snapshot.
        """
        mol_rep_map, pocket_rep_map = rep_maps if rep_maps else (None, None)
        ni_m = sample["net_input"]
        ni_p = sample_pocket["net_input"]
        mol_st = ni_m["mol_src_tokens"]
        poc_st = ni_p["pocket_src_tokens"]
        mol_rep = self._align_rep(
            mol_rep_map, sample["smi_name"], mol_st.size(1), mol_st.device
        )
        poc_rep = self._align_rep(
            pocket_rep_map,
            sample_pocket["pocket_name"],
            poc_st.size(1),
            poc_st.device,
        )
        kw = {"mol_rep_ori": mol_rep, "pocket_rep_ori": poc_rep}
        if mol_rep is None:
            kw["mol_src_tokens_ori"] = ni_m["mol_src_tokens_ori"]
            kw["mol_src_distance_ori"] = ni_m["mol_src_distance_ori"]
            kw["mol_src_edge_type_ori"] = ni_m["mol_src_edge_type_ori"]
        if poc_rep is None:
            kw["pocket_src_tokens_ori"] = ni_p["pocket_src_tokens_ori"]
            kw["pocket_src_distance_ori"] = ni_p["pocket_src_distance_ori"]
            kw["pocket_src_edge_type_ori"] = ni_p["pocket_src_edge_type_ori"]
        return kw

    # ------------------------------------------------------------------
    def _pocket_batch_size(self):
        """`--tmi-cond pair` scores one pocket at a time so that the molecule
        branch receives the exact token-pair feature of the pair being scored."""
        if getattr(self.args, "tmi_cond", "batch") == "pair":
            return 1
        return 5

    def _normalize_pocket_scores(self, scores):
        """Calibrate every pocket over the *whole* library before aggregation.

        With raw cosine/dot scores different pockets live on different offsets,
        so ``max`` over pockets systematically returns the score of the pocket
        with the highest offset instead of the best molecule-pocket match: a
        single mis-calibrated pocket can flood the top of the ranking, which is
        exactly what BEDROC / EF@0.5% measure.  Calibration uses no labels, only
        the score distribution of the screening library.

        ``scores`` is ``[num_pockets, num_molecules]``.
        """
        mode = getattr(self.args, "pocket_score_norm", "none")
        if mode == "none" or scores.shape[0] <= 1:
            return scores
        if mode == "zscore":
            mu = scores.mean(axis=1, keepdims=True)
            sd = scores.std(axis=1, keepdims=True)
            sd = np.where(sd < 1e-6, 1.0, sd)
            return (scores - mu) / sd
        if mode == "rank":
            n = scores.shape[1]
            out = np.empty_like(scores)
            for p in range(scores.shape[0]):
                order = np.argsort(scores[p], kind="stable")
                ranks = np.empty(n, dtype=np.float32)
                ranks[order] = np.arange(n, dtype=np.float32)
                out[p] = ranks / max(n - 1, 1)
            return out
        raise ValueError(f"unknown pocket score normalisation: {mode}")

    def _reset_adapted_state(self, model):
        """Restore the pre-adaptation state before every screening target.

        `checkpoint_best.pt` holds none of the adapted tensors (no `adapters`,
        `deep_prompt_embeddings`, `prompt_*` or `tmi_alpha` key), so the
        `load_state_dict(..., strict=False)` performed at the start of each
        target left them at the values reached on the *previous* target: the
        few-shot adaptation was silently warm-started from an unrelated target,
        and `freeze_tmi_encoders` then snapshotted those contaminated encoders
        as the "frozen" TMI input.  Results therefore depended on the evaluation
        order -- PPARG scored EF@0.5%=23.3 when evaluated first and 0.0 when
        evaluated 13th under identical hyper-parameters.

        The pristine state is cached on the first target and restored for every
        subsequent one, which makes each target independent and the run
        order-invariant.  `mol_model_ori` / `pocket_model_ori` are skipped
        because `freeze_tmi_encoders` rebuilds them right after this call.
        """
        skip = ("mol_model_ori.", "pocket_model_ori.")
        state = {
            k: v for k, v in model.state_dict().items()
            if not k.startswith(skip)
        }
        if getattr(self, "_pristine_state", None) is None:
            self._pristine_state = {
                k: v.detach().to("cpu", copy=True) for k, v in state.items()
            }
            logger.info(
                "[reset] cached the pre-adaptation state of %d tensor(s)",
                len(self._pristine_state),
            )
            return
        unknown = [k for k in state if k not in self._pristine_state]
        if unknown:
            logger.warning(
                "[reset] %d tensor(s) missing from the snapshot and left "
                "untouched, e.g. %s", len(unknown), unknown[:5],
            )
        model.load_state_dict(self._pristine_state, strict=False)
        logger.info("[reset] restored the pre-adaptation state")

    def _log_pocket_score_stats(self, scores, target, tag):
        """Per-pocket score statistics: a pocket whose mean is far above the
        others is the one `max` aggregation will keep, and the prime suspect
        when AUROC improves while BEDROC / EF@0.5% degrade."""
        try:
            scores = np.asarray(scores, dtype=np.float32)
            if scores.ndim != 2:
                return
            means = np.round(scores.mean(axis=1), 4).tolist()
            stds = np.round(scores.std(axis=1), 4).tolist()
            maxes = np.round(scores.max(axis=1), 4).tolist()
            logger.info(
                "[pocket scores] dataset=%s target=%s pockets=%d agg=%s norm=%s "
                "mean=%s std=%s max=%s",
                tag, target, scores.shape[0],
                getattr(self.args, "pocket_aggregate", "max"),
                getattr(self.args, "pocket_score_norm", "none"),
                means, stds, maxes,
            )
        except Exception as exc:  # diagnostics must never break a run
            logger.warning("[pocket scores] %s failed: %s", target, exc)

    def _pocket_agg_tag(self):
        """Result files must state how the pockets were aggregated, otherwise a
        `max` and a `topk_mean` sweep are indistinguishable afterwards."""
        mode = getattr(self.args, "pocket_aggregate", "max")
        norm = getattr(self.args, "pocket_score_norm", "none")
        if mode == "max" and norm == "none":
            return ""
        tag = f"_pagg-{mode}"
        if mode == "topk_mean":
            tag += str(int(getattr(self.args, "pocket_topk", 2)))
        if norm != "none":
            tag += f"-{norm}"
        return tag

    def _aggregate_pocket_scores(self, pocket_scores):
        """Aggregate per-pocket molecule scores for screening.

        ``pocket_scores`` is ``[num_pockets, num_molecules]``.  The historical
        PCBA path used a hard max, which can promote a molecule because of one
        anomalously high pocket.  ``topk_mean`` keeps strong pocket evidence but
        requires support from up to k pockets; with one pocket it is identical
        to max, so DUD-E behavior is unchanged.
        """
        scores = np.asarray(pocket_scores, dtype=np.float32)
        if scores.ndim != 2 or scores.shape[0] == 0:
            raise ValueError(
                f"expected [num_pockets, num_molecules], got {scores.shape}"
            )
        scores = self._normalize_pocket_scores(scores)
        mode = getattr(self.args, "pocket_aggregate", "max")
        if mode == "max":
            return scores.max(axis=0)
        if mode == "mean":
            return scores.mean(axis=0)
        if mode == "logsumexp":
            m = scores.max(axis=0)
            return m + np.log(np.exp(scores - m[None, :]).mean(axis=0))
        if mode == "topk_mean":
            k = max(1, min(int(getattr(self.args, "pocket_topk", 2)), scores.shape[0]))
            if k == 1:
                return scores.max(axis=0)
            return np.sort(scores, axis=0)[-k:, :].mean(axis=0)
        raise ValueError(f"unknown pocket aggregation: {mode}")

    def _score_library(self, model, mol_dataset, pocket_data, bsz, perm_seed, indices=None, rep_maps=None):
        """Score a molecule library with an explicit index-aware batching.

        The mini-batches are built from a permutation of ``indices`` so that
        every score can be traced back to its molecule, which makes it possible
        to compare the scores obtained under different batch sizes / partitions.
        """
        if indices is None:
            indices = np.arange(len(mol_dataset))
        order = np.asarray(indices, dtype=np.int64).copy()
        np.random.RandomState(perm_seed).shuffle(order)

        scores, labels = {}, {}
        model.eval()
        with torch.no_grad():
            for start in range(0, len(order), bsz):
                idx = order[start:start + bsz]
                sample = mol_dataset.collater([mol_dataset[int(i)] for i in idx])
                sample = unicore.utils.move_to_cuda(sample)
                mol_dist = sample["net_input"]["mol_src_distance"]
                mol_et = sample["net_input"]["mol_src_edge_type"]
                mol_st = sample["net_input"]["mol_src_tokens"]

                preds = []
                for _, sample_pocket in enumerate(pocket_data):
                    sample_pocket = unicore.utils.move_to_cuda(sample_pocket)
                    pocket_dist = sample_pocket["net_input"]["pocket_src_distance"]
                    pocket_et = sample_pocket["net_input"]["pocket_src_edge_type"]
                    pocket_st = sample_pocket["net_input"]["pocket_src_tokens"]

                    mol_emb, pocket_emb = model(
                        mol_src_tokens=mol_st,
                        mol_src_distance=mol_dist,
                        mol_src_edge_type=mol_et,
                        pocket_src_tokens=pocket_st,
                        pocket_src_distance=pocket_dist,
                        pocket_src_edge_type=pocket_et,
                        inference=True,
                        **self._tmi_kwargs(sample, sample_pocket, rep_maps),
                    )
                    res = pocket_emb @ mol_emb.T  # [B_pocket, B_mol]
                    preds.append(res.max(dim=0)[0].float().cpu().numpy())

                final = np.stack(preds).max(axis=0)
                tgt = sample["target"].detach().cpu().numpy()
                for k, i in enumerate(idx):
                    scores[int(i)] = float(final[k])
                    labels[int(i)] = int(tgt[k])
        return scores, labels

    def _batch_invariance_report(self, model, mol_dataset, pocket_data, target, tag,
                                 rep_maps=None):
        """Re-score the same library under different batch sizes / partitions."""
        from scipy.stats import spearmanr

        n = len(mol_dataset)
        indices = np.arange(n)
        max_mols = int(getattr(self.args, "bi_max_mols", 2000))
        if 0 < max_mols < n:
            indices = np.sort(
                np.random.RandomState(GLOBAL_SEED).choice(n, size=max_mols, replace=False)
            )

        bszs = [int(x) for x in str(self.args.bi_batch_sizes).split(",") if x.strip()]
        repeats = max(int(self.args.bi_repeats), 1)

        runs = []
        for bs in bszs:
            for rep in range(repeats):
                sc, lb = self._score_library(
                    model, mol_dataset, pocket_data, bs, 1000 * bs + rep, indices,
                    rep_maps=rep_maps,
                )
                runs.append((bs, rep, sc, lb))

        keys = sorted(runs[0][2].keys())
        ref_vec = np.array([runs[0][2][k] for k in keys], dtype=np.float64)
        y_true = np.array([runs[0][3][k] for k in keys], dtype=np.int32)
        topk = max(1, int(round(float(self.args.bi_topk_frac) * len(keys))))
        key_arr = np.asarray(keys)
        ref_top = set(key_arr[np.argsort(-ref_vec)[:topk]].tolist())

        rows = []
        for bs, rep, sc, _ in runs:
            vec = np.array([sc[k] for k in keys], dtype=np.float64)
            rho = 1.0
            if len(keys) > 2 and np.std(vec) > 0 and np.std(ref_vec) > 0:
                rho = float(spearmanr(ref_vec, vec).correlation)
            top = set(key_arr[np.argsort(-vec)[:topk]].tolist())
            auc, bedroc, ef, re = cal_metrics(y_true, vec, 80.5)
            rows.append({
                "target": target,
                "batch_size": bs,
                "partition": rep,
                "n_mols": len(keys),
                "spearman_vs_ref": rho,
                f"top{topk}_overlap": len(top & ref_top) / float(topk),
                "max_abs_score_diff": float(np.max(np.abs(vec - ref_vec))),
                "AUC": auc,
                "BEDROC": bedroc,
                "EF@0.01": ef["0.01"],
            })
            logger.info(
                "[batch-invariance] %s bs=%d part=%d rho=%.6f top%d-overlap=%.4f "
                "max|dscore|=%.3e AUC=%.4f",
                target, bs, rep, rho, topk, rows[-1][f"top{topk}_overlap"],
                rows[-1]["max_abs_score_diff"], auc,
            )

        out = (
            f"batch_invariance_{tag}_{target}_ft{self.args.ft}"
            f"_tmi{getattr(self.args, 'tmi_mode', 'legacy')}"
            f"_{getattr(self.args, 'tmi_cond', 'batch')}"
            f"_{self._tmi_affinity_tag()}"
            f"_sample{self.args.sample_time}.csv"
        )
        pd.DataFrame(rows).to_csv(out, index=False)
        return rows

    def _tmi_affinity_tag(self):
        """`<gate>-<affinity>tau<temperature>` -- part of the result file names so
        that runs with a different gate mode / affinity sharpness cannot be
        confused."""
        aff = getattr(self.args, "tmi_affinity", "cosine")
        tau = getattr(self.args, "tmi_temperature", None)
        if tau is None:
            tau = 0.05 if aff == "cosine" else 1.0
        gate = getattr(self.args, "tmi_gate_mode", "legacy")
        tag = f"gate{gate}-{aff}tau{float(tau):g}"
        if gate == "gated":
            beta = float(getattr(self.args, "tmi_alpha_beta", 1.0))
            k = int(getattr(self.args, "tmi_inject_layers", -1))
            als = float(getattr(self.args, "tmi_alpha_lr_scale", 1.0))
            tag += f"-beta{beta:g}-inj{k}-alr{als:g}"
        if getattr(self.args, "keep_target_warm_start", False):
            tag += "-warmstart"
        return tag

    def _log_tmi_alpha(self, model, target, tag):
        """Log the learnt injection strengths so that a per-target correlation
        with the metric change can be computed afterwards (grep `[TMI alpha]`)."""
        if getattr(self.args, "tmi_gate_mode", "legacy") != "gated":
            return
        if not hasattr(model, "tmi_alpha_report"):
            return
        try:
            report = model.tmi_alpha_report()
        except Exception as exc:  # diagnostics must never break a run
            logger.warning("[TMI alpha] %s failed: %s", target, exc)
            return
        for branch, stats in report.items():
            logger.info(
                "[TMI alpha] dataset=%s target=%s branch=%s max_abs=%.6f "
                "mean_abs=%.6f l2=%.6f per_layer=%s",
                tag, target, branch, stats["max_abs"], stats["mean_abs"],
                stats["l2"], stats["per_layer"],
            )

    def _select_targets(self, targets):
        """Restrict the evaluation to a development subset.

        Sweeping hyper-parameters on all 102 DUD-E targets costs ~2.5 h per
        configuration *and* tunes on the reported test set.  `--target-subset`
        (comma separated names, or a path to a text file with one name per line)
        and `--max-targets` keep the search cheap and honest.
        """
        subset = getattr(self.args, "target_subset", None)
        if subset:
            if os.path.isfile(subset):
                with open(subset) as fh:
                    wanted = [ln.strip() for ln in fh if ln.strip()]
            else:
                wanted = [t.strip() for t in subset.split(",") if t.strip()]
            available = set(targets)
            missing = [t for t in wanted if t not in available]
            if missing:
                logger.warning(
                    "[targets] %d requested target(s) not available and skipped: %s",
                    len(missing), ",".join(missing),
                )
            targets = [t for t in wanted if t in available]
        max_targets = int(getattr(self.args, "max_targets", 0) or 0)
        if max_targets > 0:
            targets = targets[:max_targets]
        logger.info("[targets] evaluating %d target(s)", len(targets))
        return targets

    def _target_subset_tag(self):
        """Marks result files produced on a development subset so they can never
        be confused with a full-benchmark run."""
        subset = getattr(self.args, "target_subset", None)
        max_targets = int(getattr(self.args, "max_targets", 0) or 0)
        if not subset and max_targets <= 0:
            return ""
        if subset and os.path.isfile(subset):
            name = os.path.splitext(os.path.basename(subset))[0]
        elif subset:
            name = f"n{len([t for t in subset.split(',') if t.strip()])}"
        else:
            name = f"first{max_targets}"
        return f"_dev-{name}"


    def _test_pcba_zero_shot_target(self, target, model):
        """Score the query library with the frozen base encoders only."""
        model.eval()
        if not hasattr(model, "mol_model_ori"):
            model.freeze_tmi_encoders()

        ft = self.args.ft
        suffix = f"_{self.args.sample_time}" if self.args.sample_time in [1, 2] else ""
        data_path_test = (
            f"/root/autodl-tmp/Few-VS/data/lit_pcba/{target}/"
            f"mols_remain_{ft}pos_{ft}neg{suffix}.lmdb"
        )
        mol_dataset_test = self.load_mols_dataset(
            data_path_test, "atoms", "coordinates"
        )
        pocket_dataset = self.load_pockets_dataset(
            f"/root/autodl-tmp/Few-VS/data/lit_pcba/{target}/pockets.lmdb"
        )
        mol_data_test = torch.utils.data.DataLoader(
            mol_dataset_test,
            batch_size=self.args.bsz,
            collate_fn=mol_dataset_test.collater,
            shuffle=False,
            num_workers=0,
        )
        pocket_data = torch.utils.data.DataLoader(
            pocket_dataset,
            batch_size=5,
            collate_fn=pocket_dataset.collater,
            shuffle=False,
            num_workers=0,
        )

        per_batch_pocket_scores = []
        labels = []
        with torch.no_grad():
            for sample in tqdm(mol_data_test):
                sample = unicore.utils.move_to_cuda(sample)
                mol_rep = model._encode_frozen(
                    sample["net_input"]["mol_src_tokens"],
                    sample["net_input"]["mol_src_distance"],
                    sample["net_input"]["mol_src_edge_type"],
                    "mol",
                )
                mol_emb = F.normalize(
                    model.mol_project(mol_rep[:, 0, :]), dim=-1
                )
                batch_scores = []
                for sample_pocket in pocket_data:
                    sample_pocket = unicore.utils.move_to_cuda(sample_pocket)
                    pocket_rep = model._encode_frozen(
                        sample_pocket["net_input"]["pocket_src_tokens"],
                        sample_pocket["net_input"]["pocket_src_distance"],
                        sample_pocket["net_input"]["pocket_src_edge_type"],
                        "pocket",
                    )
                    pocket_emb = F.normalize(
                        model.pocket_project(pocket_rep[:, 0, :]), dim=-1
                    )
                    batch_scores.append(
                        (pocket_emb @ mol_emb.T).float().cpu().numpy()
                    )
                per_batch_pocket_scores.append(
                    np.concatenate(batch_scores, axis=0)
                )
                labels.extend(sample["target"].detach().cpu().numpy())

        all_pocket_scores = np.concatenate(per_batch_pocket_scores, axis=1)
        scores = self._aggregate_pocket_scores(all_pocket_scores)
        labels = np.asarray(labels, dtype=np.int32)
        auc, bedroc, ef_list, re_list = cal_metrics(labels, scores, 80.5)
        print(target)
        print(np.sum(labels), len(labels) - np.sum(labels))
        print(f"target:{target}", f"auc:{auc}", f"bedroc:{bedroc}")
        torch.cuda.empty_cache()
        return auc, bedroc, ef_list, re_list, list(scores), labels

    def test_pcba_target(self, target, model, **kwargs):
        if getattr(self.args, "zero_shot", False):
            return self._test_pcba_zero_shot_target(target, model)
        ft = self.args.ft
        model_original_weight = unicore.utils.move_to_cuda(
            torch.load('/root/autodl-tmp/Few-VS/checkpoint_best.pt')["model"])
        model.load_state_dict(model_original_weight, strict=False)
        model = model.to('cuda').float()

        warm_start = getattr(self.args, "keep_target_warm_start", False)
        if warm_start:
            logger.warning(
                "[warm-start] retaining adapted parameters and TMI anchors for target %s",
                target,
            )
        else:
            # The checkpoint omits adapted tensors, so restore them before each
            # target to keep the benchmark independent of target order.
            self._reset_adapted_state(model)

        # Freeze a snapshot of the pre-adaptation encoders: it reproduces the
        # legacy offline embeddings for the entries missing from the cache.
        if hasattr(model, "freeze_tmi_encoders"):
            model.freeze_tmi_encoders()

        # Anchors are target-specific in the independent-reset mode. In the
        # warm-start mode they are intentionally retained as part of the
        # historical cross-target state.
        if not warm_start and hasattr(model, "reset_tmi_anchors"):
            model.reset_tmi_anchors()

        params, param_names = self.collect_params(model)

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        seed = getattr(self, 'seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

        seed_worker = _global_seed_worker
        g = torch.Generator()
        g.manual_seed(GLOBAL_SEED)

        suffix = f"_{self.args.sample_time}" if self.args.sample_time in [1, 2] else ""
        data_path_train = f"/root/autodl-tmp/Few-VS/data/lit_pcba/{target}/mols_few_{ft}pos_{ft}neg{suffix}.lmdb"
        data_path_test = f"/root/autodl-tmp/Few-VS/data/lit_pcba/{target}/mols_remain_{ft}pos_{ft}neg{suffix}.lmdb"


        # data_path_train = "/root/autodl-tmp/Few-VS/Drug-fewshot/data/lit_pcba/" + target + f"/mols_few_{ft}pos_{ft}neg_{self.args.sample_time}.lmdb"
        mol_dataset_train = self.load_mols_dataset_fewshot(data_path_train, "atoms", "coordinates")
        # data_path_test = "/root/autodl-tmp/Few-VS/Drug-fewshot/data/lit_pcba/" + target + f"/mols_remain_{ft}pos_{ft}neg_{self.args.sample_time}.lmdb"
        mol_dataset_test = self.load_mols_dataset_fewshot(data_path_test, "atoms", "coordinates")
        num_data = len(mol_dataset_train)
        bsz = self.args.bsz



        print(num_data // bsz)
        mol_reps = []
        mol_names = []
        labels = []

        mol_data_train = torch.utils.data.DataLoader(
            mol_dataset_train,
            batch_size=5,
            collate_fn=mol_dataset_train.collater,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            worker_init_fn=seed_worker,
            generator=g,
        )
        mol_data_test = torch.utils.data.DataLoader(
            mol_dataset_test,
            batch_size=bsz,
            collate_fn=mol_dataset_test.collater,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            worker_init_fn=seed_worker,
            generator=g
        )
        pocket_dataset = self.load_pockets_dataset_fewshot("/root/autodl-tmp/Few-VS/data/lit_pcba/" + target + "/pockets.lmdb")
        pocket_data = torch.utils.data.DataLoader(
            pocket_dataset,
            batch_size=self._pocket_batch_size(),
            collate_fn=pocket_dataset.collater,
            shuffle=False,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=g,
        )

        optimizer = torch.optim.SGD(params, lr=self.args.lr, momentum=0.9)
        epoch = self.args.epoch_train

        rep_maps = self._load_rep_cache("pcba", target)

        model.train()

        # Preserve the legacy optimizer cadence while keeping pairwise TMI:
        # the old loop processed one batch of up to five pockets, then stepped
        # once for *each* molecule batch.  Pair conditioning requires separate
        # forwards, so form the same five-pocket groups, average their losses
        # for one molecule batch, and then step.  Thus ft=2/4/8 retain the old
        # number and ordering of SGD-momentum updates.
        pocket_group_size = 5 if self._pocket_batch_size() == 1 else 1
        pocket_batches = list(pocket_data)
        for i in tqdm(range(epoch)):
            for start in range(0, len(pocket_batches), pocket_group_size):
                pocket_group = pocket_batches[start:start + pocket_group_size]
                group_size = len(pocket_group)
                for _, sample in enumerate(mol_data_train):
                    sample = unicore.utils.move_to_cuda(sample)
                    mol_dist = sample["net_input"]["mol_src_distance"]
                    mol_et = sample["net_input"]["mol_src_edge_type"]
                    mol_st = sample["net_input"]["mol_src_tokens"]

                    optimizer.zero_grad()
                    for sample_pocket in pocket_group:
                        sample_pocket = unicore.utils.move_to_cuda(sample_pocket)
                        pocket_dist = sample_pocket["net_input"]["pocket_src_distance"]
                        pocket_et = sample_pocket["net_input"]["pocket_src_edge_type"]
                        pocket_st = sample_pocket["net_input"]["pocket_src_tokens"]

                        mol_emb, pocket_emb = model(
                            mol_src_tokens=mol_st,
                            mol_src_distance=mol_dist,
                            mol_src_edge_type=mol_et,
                            pocket_src_tokens=pocket_st,
                            pocket_src_distance=pocket_dist,
                            pocket_src_edge_type=pocket_et,
                            inference=True,
                            **self._tmi_kwargs(sample, sample_pocket, rep_maps),
                        )
                        mol_emb = F.normalize(mol_emb, dim=-1)
                        pocket_emb = F.normalize(pocket_emb, dim=-1)
                        loss = self.compute_classification_loss(
                            mol_emb, pocket_emb, sample["target"]
                        )
                        (loss / group_size).backward()
                    optimizer.step()
        model.eval()
        self._log_tmi_alpha(model, target, "pcba")
        if getattr(self.args, "batch_invariance", False):
            self._batch_invariance_report(
                model, mol_dataset_test, pocket_data, target, "pcba",
                rep_maps=rep_maps,
            )
        pocket_reps = []
        res_single_list = []
        res_single_tmp_list = []
        # `[num_pockets, num_molecules]` for the *whole* library: the pockets can
        # only be calibrated / top-k averaged once every molecule has been
        # scored, so the aggregation is deferred to after the loop.
        per_batch_pocket_scores = []
        with torch.no_grad():
            for _, sample in enumerate(tqdm(mol_data_test)):
                # 为每个测试mol收集所有pocket的预测结果
                mol_predictions = []
                sample = unicore.utils.move_to_cuda(sample)
                for _, sample_pocket in enumerate(pocket_data):

                    sample_pocket = unicore.utils.move_to_cuda(sample_pocket)
                    mol_dist = sample["net_input"]["mol_src_distance"]
                    mol_et = sample["net_input"]["mol_src_edge_type"]
                    mol_st = sample["net_input"]["mol_src_tokens"]
                    pocket_dist = sample_pocket["net_input"]["pocket_src_distance"]
                    pocket_et = sample_pocket["net_input"]["pocket_src_edge_type"]
                    pocket_st = sample_pocket["net_input"]["pocket_src_tokens"]

                    mol_emb, pocket_emb = model(
                        mol_src_tokens=mol_st,
                        mol_src_distance=mol_dist,
                        mol_src_edge_type=mol_et,
                        pocket_src_tokens=pocket_st,
                        pocket_src_distance=pocket_dist,
                        pocket_src_edge_type=pocket_et,
                        inference=True,
                        **self._tmi_kwargs(sample, sample_pocket, rep_maps),
                    )

                    res = pocket_emb @ mol_emb.T  # [B_pocket, B_mol]

                    # keep every pocket row: collapsing the pocket batch with a
                    # max here made `mean` / `topk_mean` / calibration operate on
                    # already-maxed groups of 5 pockets
                    mol_predictions.append(res.float().cpu().numpy())

                # [num_pockets, B_mol]
                batch_scores = np.concatenate(mol_predictions, axis=0)
                per_batch_pocket_scores.append(batch_scores)
                mol_names.extend(sample["smi_name"])
                labels.extend(sample["target"].detach().cpu().numpy())

            # [num_pockets, num_molecules] over the full library
            all_pocket_scores = np.concatenate(per_batch_pocket_scores, axis=1)
            self._log_pocket_score_stats(all_pocket_scores, target, "pcba")
            res_single_list = list(self._aggregate_pocket_scores(all_pocket_scores))

            labels = np.array(labels, dtype=np.int32)
            auc, bedroc, ef_list, re_list = cal_metrics(labels, res_single_list, 80.5)

            print(target)
            print(np.sum(labels), len(labels) - np.sum(labels))
            print(f'target:{target}', f'auc:{auc}', f'bedroc:{bedroc}')

            del model
            torch.cuda.empty_cache()

        return auc, bedroc, ef_list, re_list, res_single_list, labels

    def test_dude_target(self, target, model, **kwargs):
        ft = self.args.ft
        model_original_weight = unicore.utils.move_to_cuda(
            torch.load('/root/autodl-tmp/Few-VS/checkpoint_best.pt')["model"])
        model.load_state_dict(model_original_weight, strict=False)
        model = model.to('cuda').float()

        # see test_pcba_target: reset the adapted tensors so that the targets
        # are independent of each other and of the evaluation order
        self._reset_adapted_state(model)

        # Freeze a snapshot of the pre-adaptation encoders (see test_pcba_target).
        if hasattr(model, "freeze_tmi_encoders"):
            model.freeze_tmi_encoders()

        # the batch-invariant TMI anchors are target specific -> reset them
        if hasattr(model, "reset_tmi_anchors"):
            model.reset_tmi_anchors()

        params, param_names = self.collect_params(model)

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        seed = getattr(self, 'seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

  
        seed_worker = _global_seed_worker
        g = torch.Generator()
        g.manual_seed(GLOBAL_SEED)
        suffix = f"_{self.args.sample_time}" if self.args.sample_time in [1, 2] else ""
        data_path_train = f"/root/autodl-tmp/Few-VS/data/dude/raw/all/{target}/mols_few_{ft}pos_{ft}neg{suffix}.lmdb"
        data_path_test = f"/root/autodl-tmp/Few-VS/data/dude/raw/all/{target}/mols_remain_{ft}pos_{ft}neg{suffix}.lmdb"


        mol_dataset_train = self.load_mols_dataset_fewshot(data_path_train, "atoms", "coordinates")
        mol_dataset_test = self.load_mols_dataset_fewshot(data_path_test, "atoms", "coordinates")
        num_data = len(mol_dataset_train)
        bsz = self.args.bsz

        print(num_data // bsz)
        mol_reps = []
        mol_names = []
        labels = []

        # generate mol data

        mol_data_train = torch.utils.data.DataLoader(
            mol_dataset_train,
            batch_size=5,
            collate_fn=mol_dataset_train.collater,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            worker_init_fn=seed_worker,
            generator=g,
        )
        mol_data_test = torch.utils.data.DataLoader(
            mol_dataset_test,
            batch_size=bsz,
            collate_fn=mol_dataset_test.collater,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            worker_init_fn=seed_worker,
            generator=g
        )
        pocket_dataset = self.load_pockets_dataset_fewshot("/root/autodl-tmp/Few-VS/data/dude/raw/all/" + target + "/pocket.lmdb")
        pocket_data = torch.utils.data.DataLoader(
            pocket_dataset,
            batch_size=self._pocket_batch_size(),
            collate_fn=pocket_dataset.collater,
            shuffle=False,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=g,
        )
        optimizer = torch.optim.SGD(params, lr=self.args.lr, momentum=0.9)

        epoch = self.args.epoch_train

        rep_maps = self._load_rep_cache("dude", target)

        model.train()

        for i in tqdm(range(epoch)):
            for _, sample_pocket in enumerate(pocket_data):
                sample_pocket = unicore.utils.move_to_cuda(sample_pocket)
                pocket_dist = sample_pocket["net_input"]["pocket_src_distance"]
                pocket_et = sample_pocket["net_input"]["pocket_src_edge_type"]
                pocket_st = sample_pocket["net_input"]["pocket_src_tokens"]
                for _, sample in enumerate(mol_data_train):

                    sample = unicore.utils.move_to_cuda(sample)

                    mol_dist = sample["net_input"]["mol_src_distance"]
                    mol_et = sample["net_input"]["mol_src_edge_type"]
                    mol_st = sample["net_input"]["mol_src_tokens"]

                    mol_emb, pocket_emb = model(
                        mol_src_tokens=mol_st,
                        mol_src_distance=mol_dist,
                        mol_src_edge_type=mol_et,
                        pocket_src_tokens=pocket_st,
                        pocket_src_distance=pocket_dist,
                        pocket_src_edge_type=pocket_et,
                        inference=True,
                        **self._tmi_kwargs(sample, sample_pocket, rep_maps),
                    )
                    mol_emb = F.normalize(mol_emb, dim=-1)
                    pocket_emb = F.normalize(pocket_emb, dim=-1)
                    loss = self.compute_classification_loss(mol_emb, pocket_emb, sample['target'])

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
        model.eval()
        self._log_tmi_alpha(model, target, "dude")
        if getattr(self.args, "batch_invariance", False):
            self._batch_invariance_report(
                model, mol_dataset_test, pocket_data, target, "dude",
                rep_maps=rep_maps,
            )
        pocket_reps = []
        res_single_list = []
        res_single_tmp_list = []
        with torch.no_grad():
            for _, sample in enumerate(tqdm(mol_data_test)):
                mol_predictions = []
                sample = unicore.utils.move_to_cuda(sample)

                for _, sample_pocket in enumerate(pocket_data):

                    sample_pocket = unicore.utils.move_to_cuda(sample_pocket)
                    mol_dist = sample["net_input"]["mol_src_distance"]
                    mol_et = sample["net_input"]["mol_src_edge_type"]
                    mol_st = sample["net_input"]["mol_src_tokens"]
                    pocket_dist = sample_pocket["net_input"]["pocket_src_distance"]
                    pocket_et = sample_pocket["net_input"]["pocket_src_edge_type"]
                    pocket_st = sample_pocket["net_input"]["pocket_src_tokens"]

                    mol_emb, pocket_emb = model(
                        mol_src_tokens=mol_st,
                        mol_src_distance=mol_dist,
                        mol_src_edge_type=mol_et,
                        pocket_src_tokens=pocket_st,
                        pocket_src_distance=pocket_dist,
                        pocket_src_edge_type=pocket_et,
                        inference=True,
                        **self._tmi_kwargs(sample, sample_pocket, rep_maps),
                    )

                    res = pocket_emb @ mol_emb.T  # [B_pocket, B_mol]

              
                    mol_max_similarities = res.max(dim=0)[0]  
                    mol_predictions.append(mol_max_similarities.cpu().numpy())


                mol_predictions = np.stack(mol_predictions)  # [num_pockets, B_mol]


                final_mol_scores = mol_predictions.max(axis=0)  # [B_mol]

                res_single_list.extend(final_mol_scores)
                mol_names.extend(sample["smi_name"])
                labels.extend(sample["target"].detach().cpu().numpy())

            labels = np.array(labels, dtype=np.int32)
            auc, bedroc, ef_list, re_list = cal_metrics(labels, res_single_list, 80.5)

            print(target)
            print(np.sum(labels), len(labels) - np.sum(labels))
            print(f'target:{target}', f'auc:{auc}', f'bedroc:{bedroc}')

            del model
            torch.cuda.empty_cache()

        return auc, bedroc, ef_list, re_list, res_single_list, labels

    def test_pcba(self, model, **kwargs):
        base_dir = "/root/autodl-tmp/Few-VS/data/lit_pcba/"
        targets = os.listdir(base_dir)
        filtered_targets = []
        for t in targets:
            lmdb_path = os.path.join(base_dir, t, f"mols_remain_{self.args.ft}pos_{self.args.ft}neg.lmdb")
            if os.path.exists(lmdb_path):
                filtered_targets.append(t)
        # save filtered target names as npz
        np.savez(os.path.join(base_dir, f"targets_with_{self.args.ft}pos_{self.args.ft}neg.npz"), targets=np.array(filtered_targets))
        # BUGFIX: PCBA used to ignore `--target-subset` / `--max-targets`, so a
        # run meant for the 6 degraded targets silently screened all 15.
        targets = self._select_targets(filtered_targets)

        # print(targets)
        auc_list = []
        ef_list = []
        bedroc_list = []

        re_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": []
        }
        ef_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": []
        }
        for target in targets:
            auc, bedroc, ef, re, res_single, labels = self.test_pcba_target(target, model)
            auc_list.append(auc)
            bedroc_list.append(bedroc)
            for key in ef:
                ef_list[key].append(ef[key])
            # print("re", re)
            # print("ef", ef)
            for key in re:
                re_list[key].append(re[key])
        print(auc_list)
        print(ef_list)
        print("auc 25%", np.percentile(auc_list, 25))
        print("auc 50%", np.percentile(auc_list, 50))
        print("auc 75%", np.percentile(auc_list, 75))
        print("auc mean", np.mean(auc_list))
        print("bedroc 25%", np.percentile(bedroc_list, 25))
        print("bedroc 50%", np.percentile(bedroc_list, 50))
        print("bedroc 75%", np.percentile(bedroc_list, 75))
        print("bedroc mean", np.mean(bedroc_list))

        for key in ef_list:
            print("ef", key, "25%", np.percentile(ef_list[key], 25))
            print("ef", key, "50%", np.percentile(ef_list[key], 50))
            print("ef", key, "75%", np.percentile(ef_list[key], 75))
            print("ef", key, "mean", np.mean(ef_list[key]))
        for key in re_list:
            print("re", key, "25%", np.percentile(re_list[key], 25))
            print("re", key, "50%", np.percentile(re_list[key], 50))
            print("re", key, "75%", np.percentile(re_list[key], 75))
            print("re", key, "mean", np.mean(re_list[key]))
        df = pd.DataFrame({
            'target': targets,
            'AUC': auc_list,
            'BEDROC': bedroc_list,
            'EF@0.005': ef_list["0.005"],
            'EF@0.01': ef_list["0.01"],
            'EF@0.02': ef_list["0.02"],
            'EF@0.05': ef_list["0.05"],
            'RE@0.005': re_list["0.005"],
            'RE@0.01': re_list["0.01"],
            'RE@0.02': re_list["0.02"],
            'RE@0.05': re_list["0.05"]
        })

        zero_shot_tag = "_zero-shot" if getattr(self.args, "zero_shot", False) else ""
        df.to_csv(f'pcba_target_metrics_{self.args.ft}_{self.args.lr}_{self.args.epoch_train}_fewshot_{self.args.sample_time}_moltoken{self.args.mol_token}_pockettoken{self.args.pocket_token}_tmi{getattr(self.args, "tmi_mode", "legacy")}-{getattr(self.args, "tmi_cond", "batch")}-{self._tmi_affinity_tag()}{self._pocket_agg_tag()}{self._target_subset_tag()}{zero_shot_tag}_{datetime.now():%Y%m%d-%H%M%S}_adapter.csv', index=False)

        
        return

    def collect_params(self, model):
        model.requires_grad_(False)  

        params, names = [], []
        # `prompt_gate_norm` / `prompt_out_norm` are themselves LayerNorm modules,
        # so the LayerNorm sweep below already collects them; matching them again
        # by keyword used to append the *same* tensors a second time, which made
        # SGD apply their update twice per step (hence the runtime warning
        # "optimizer contains a parameter group with duplicate parameters") and
        # silently doubled their effective learning rate.  Harmless for
        # `--tmi-gate-mode legacy` (where `prompt_gate_norm` is unused) but not
        # for `gated`.  `seen` makes the collection idempotent.
        seen = set()

        def _add(param, name):
            if not param.requires_grad:
                return
            if id(param) in seen:
                return
            seen.add(id(param))
            params.append(param)
            names.append(name)

        # Match the previous few-shot implementation. The encoder parameters
        # are named `adapters`, `deep_prompt_embeddings` and `prompt_proj`;
        # using only `mol_adapter` / `pocket_adapter` silently freezes them.
        target_keywords = ['adapters', 'deep_prompt_embeddings', 'prompt_proj', 'prompt_weight',
                           'prompt_gate_norm', 'prompt_out_norm', 'tmi_alpha']


        for submodel in [model.mol_model, model.pocket_model]:
            for name, module in submodel.named_modules():
                if isinstance(module, LayerNorm):
                    module.requires_grad_(True)
                    for pname, param in module.named_parameters():
                        if pname in ['weight', 'bias']:
                            param.requires_grad_(True)
                            _add(param, f"{name}.{pname}")


        for submodel in [model.mol_model, model.pocket_model]:
            for name, param in submodel.named_parameters():
                if any(key in name for key in target_keywords):
                    param.requires_grad_(True)
                    _add(param, name)


        for name, param in model.named_parameters():
            if name.startswith(('mol_project', 'pocket_project')):
                param.requires_grad_(True)
                _add(param, name)

        # `tmi_alpha` multiplies a token-scale (~sqrt(D) = 22.6) quantity while
        # every other adapted parameter only ever sees the ~3e-3 gated prompt, so
        # with a shared lr it moves orders of magnitude faster and overshoots
        # within the first few support updates.  Give it its own group.
        alpha_scale = float(getattr(self.args, "tmi_alpha_lr_scale", 1.0))
        base_lr = float(self.args.lr)
        if alpha_scale == 1.0:
            param_groups = [{"params": params}]
        else:
            alpha_params, alpha_names = [], []
            other_params, other_names = [], []
            for p, n in zip(params, names):
                if "tmi_alpha" in n:
                    alpha_params.append(p)
                    alpha_names.append(n)
                else:
                    other_params.append(p)
                    other_names.append(n)
            param_groups = [{"params": other_params, "lr": base_lr}]
            if alpha_params:
                param_groups.append(
                    {"params": alpha_params, "lr": base_lr * alpha_scale}
                )
            logger.info(
                "[TMI params] %d adapted tensors (%d tmi_alpha at lr=%.3g, "
                "%d others at lr=%.3g)",
                len(params), len(alpha_params), base_lr * alpha_scale,
                len(other_params), base_lr,
            )
            names = other_names + alpha_names

        return param_groups, names

    

    def test_dude(self, model, **kwargs):
        base_dir = "/root/autodl-tmp/Few-VS/data/dude/raw/all/"
        targets = os.listdir(base_dir)
        filtered_targets = []
        for t in targets:
            lmdb_path = os.path.join(base_dir, t, f"mols_remain_{self.args.ft}pos_{self.args.ft}neg.lmdb")
            if os.path.exists(lmdb_path):
                filtered_targets.append(t)
        np.savez(os.path.join(base_dir, f"targets_with_{self.args.ft}pos_{self.args.ft}neg.npz"),
                 targets=np.array(filtered_targets))
        targets = self._select_targets(filtered_targets)
        auc_list = []
        bedroc_list = []
        ef_list = []
        res_list = []
        labels_list = []
        re_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": [],
        }
        ef_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": [],
        }
        for i, target in enumerate(targets):
            auc, bedroc, ef, re, res_single, labels = self.test_dude_target(target, model)
            auc_list.append(auc)
            bedroc_list.append(bedroc)
            for key in ef:
                ef_list[key].append(ef[key])
            for key in re_list:
                re_list[key].append(re[key])
            res_list.append(res_single)
            labels_list.append(labels)
        print("auc mean", np.mean(auc_list))
        print("bedroc mean", np.mean(bedroc_list))

        for key in ef_list:
            print("ef", key, "mean", np.mean(ef_list[key]))

        for key in re_list:
            print("re", key, "mean", np.mean(re_list[key]))

        df = pd.DataFrame({
            'target': targets,
            'AUC': auc_list,
            'BEDROC': bedroc_list,
            'EF@0.005': ef_list["0.005"],
            'EF@0.01': ef_list["0.01"],
            'EF@0.02': ef_list["0.02"],
            'EF@0.05': ef_list["0.05"],
            'RE@0.005': re_list["0.005"],
            'RE@0.01': re_list["0.01"],
            'RE@0.02': re_list["0.02"],
            'RE@0.05': re_list["0.05"],

        })

        # 保存为 CSV 文件
        df.to_csv(f'dude_target_metrics_{self.args.ft}_{self.args.lr}_{self.args.epoch_train}_fewshot_{self.args.sample_time}_moltoken{self.args.mol_token}_pockettoken{self.args.pocket_token}_tmi{getattr(self.args, "tmi_mode", "legacy")}-{getattr(self.args, "tmi_cond", "batch")}-{self._tmi_affinity_tag()}{self._target_subset_tag()}_{datetime.now():%Y%m%d-%H%M%S}_adapter.csv', index=False)

        return
























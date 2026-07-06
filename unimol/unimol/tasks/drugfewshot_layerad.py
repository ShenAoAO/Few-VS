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
        # parser.add_argument("--epoch-cls",default=20, help="epoch cls",type=int)
        parser.add_argument("--sample-time", default=1, type=int, help="sample_time")
        parser.add_argument("--pocket-token", default=5, type=int, help="pocket_token")
        parser.add_argument("--mol-token", default=5, type=int, help="mol_token")



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

        state = checkpoint_utils.load_checkpoint_to_cpu('/root/autodl-tmp/Drug-fewshot/checkpoint_best.pt')
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


    def test_pcba_target(self, target, model, **kwargs):
        ft = self.args.ft
        model_original_weight = unicore.utils.move_to_cuda(
            torch.load('/root/autodl-tmp/Drug-fewshot/checkpoint_best.pt')["model"])
        model.load_state_dict(model_original_weight, strict=False)
        model = model.to('cuda').float()

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
        data_path_train = f"/root/autodl-tmp/Drug-fewshot/data/lit_pcba/{target}/mols_few_{ft}pos_{ft}neg{suffix}.lmdb"
        data_path_test = f"/root/autodl-tmp/Drug-fewshot/data/lit_pcba/{target}/mols_remain_{ft}pos_{ft}neg{suffix}.lmdb"


        # data_path_train = "/root/autodl-tmp/Drug-fewshot/data/lit_pcba/" + target + f"/mols_few_{ft}pos_{ft}neg_{self.args.sample_time}.lmdb"
        mol_dataset_train = self.load_mols_dataset(data_path_train, "atoms", "coordinates")
        # data_path_test = "/root/autodl-tmp/Drug-fewshot/data/lit_pcba/" + target + f"/mols_remain_{ft}pos_{ft}neg_{self.args.sample_time}.lmdb"
        mol_dataset_test = self.load_mols_dataset(data_path_test, "atoms", "coordinates")
        num_data = len(mol_dataset_train)
        bsz = 32



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
        pocket_dataset = self.load_pockets_dataset_fewshot("/root/autodl-tmp/Drug-fewshot/data/lit_pcba/" + target + "/pockets.lmdb")
        pocket_data = torch.utils.data.DataLoader(
            pocket_dataset,
            batch_size=5,
            collate_fn=pocket_dataset.collater,
            shuffle=False,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=g,
        )

        optimizer = torch.optim.SGD(params, lr=self.args.lr, momentum=0.9)
        epoch = self.args.epoch_train


      

        model.train()

        for i in tqdm(range(epoch)):
            for _, sample_pocket in enumerate(pocket_data):
                sample_pocket = unicore.utils.move_to_cuda(sample_pocket)

                pocket_dist = sample_pocket["net_input"]["pocket_src_distance"]
                pocket_et = sample_pocket["net_input"]["pocket_src_edge_type"]
                pocket_st = sample_pocket["net_input"]["pocket_src_tokens"]
                smi_names = sample_pocket["pocket_name"]
                pocket_rep_ori = None
                for _, sample in enumerate(mol_data_train):
                        sample = unicore.utils.move_to_cuda(sample)
                        
                        mol_dist = sample["net_input"]["mol_src_distance"]
                        mol_et = sample["net_input"]["mol_src_edge_type"]
                        mol_st = sample["net_input"]["mol_src_tokens"]

                        smi_names = sample["smi_name"]
                        mol_rep_ori = None

              
                        mol_emb, pocket_emb = model(
                            mol_src_tokens=mol_st,
                            mol_src_distance=mol_dist,
                            mol_src_edge_type=mol_et,
                            pocket_src_tokens=pocket_st,
                            pocket_src_distance=pocket_dist,
                            pocket_src_edge_type=pocket_et,
                            mol_rep_ori=mol_rep_ori,
                            pocket_rep_ori=pocket_rep_ori,
                            inference=True
                        )
                        mol_emb = F.normalize(mol_emb, dim=-1)
                        pocket_emb = F.normalize(pocket_emb, dim=-1)
                        loss = self.compute_classification_loss(mol_emb, pocket_emb, sample['target'])
    


                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
        model.eval()
        pocket_reps = []
        res_single_list = []
        res_single_tmp_list = []
        with torch.no_grad():
            for _, sample in enumerate(tqdm(mol_data_test)):
                # 为每个测试mol收集所有pocket的预测结果
                mol_predictions = []
                sample = unicore.utils.move_to_cuda(sample)
                smi_names = sample["smi_name"]
                mol_rep_ori = None
                for _, sample_pocket in enumerate(pocket_data):

                    sample_pocket = unicore.utils.move_to_cuda(sample_pocket)
                    smi_names = sample_pocket["pocket_name"]
                    pocket_rep_ori = None
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
                        mol_rep_ori=mol_rep_ori,
                        pocket_rep_ori=pocket_rep_ori,
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

    def test_dude_target(self, target, model, **kwargs):
        ft = self.args.ft
        model_original_weight = unicore.utils.move_to_cuda(
            torch.load('/root/autodl-tmp/Drug-fewshot/checkpoint_best.pt')["model"])
        model.load_state_dict(model_original_weight, strict=False)
        model = model.to('cuda').float()


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
        data_path_train = f"/root/autodl-tmp/Drug-fewshot/data/dude/raw/all/{target}/mols_few_{ft}pos_{ft}neg{suffix}.lmdb"
        data_path_test = f"/root/autodl-tmp/Drug-fewshot/data/dude/raw/all/{target}/mols_remain_{ft}pos_{ft}neg{suffix}.lmdb"


        mol_dataset_train = self.load_mols_dataset(data_path_train, "atoms", "coordinates")
        mol_dataset_test = self.load_mols_dataset(data_path_test, "atoms", "coordinates")
        num_data = len(mol_dataset_train)
        bsz = 32

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
        pocket_dataset = self.load_pockets_dataset_fewshot("/root/autodl-tmp/Drug-fewshot/data/dude/raw/all/" + target + "/pocket.lmdb")
        pocket_data = torch.utils.data.DataLoader(
            pocket_dataset,
            batch_size=5,
            collate_fn=pocket_dataset.collater,
            shuffle=False,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=g,
        )
        optimizer = torch.optim.SGD(params, lr=self.args.lr, momentum=0.9)

        epoch = self.args.epoch_train


        model.train()

        for i in tqdm(range(epoch)):
            for _, sample_pocket in enumerate(pocket_data):
                sample_pocket = unicore.utils.move_to_cuda(sample_pocket)
                pocket_dist = sample_pocket["net_input"]["pocket_src_distance"]
                pocket_et = sample_pocket["net_input"]["pocket_src_edge_type"]
                pocket_st = sample_pocket["net_input"]["pocket_src_tokens"]
                smi_names = sample_pocket["pocket_name"]
                pocket_rep_ori = None
                for _, sample in enumerate(mol_data_train):

                    sample = unicore.utils.move_to_cuda(sample)
  
                    mol_dist = sample["net_input"]["mol_src_distance"]
                    mol_et = sample["net_input"]["mol_src_edge_type"]
                    mol_st = sample["net_input"]["mol_src_tokens"]

                    smi_names = sample["smi_name"]
                    mol_rep_ori = None

                    mol_emb, pocket_emb = model(
                        mol_src_tokens=mol_st,
                        mol_src_distance=mol_dist,
                        mol_src_edge_type=mol_et,
                        pocket_src_tokens=pocket_st,
                        pocket_src_distance=pocket_dist,
                        pocket_src_edge_type=pocket_et,
                        mol_rep_ori=mol_rep_ori,
                        pocket_rep_ori=pocket_rep_ori,
                        inference=True
                    )
                    mol_emb = F.normalize(mol_emb, dim=-1)
                    pocket_emb = F.normalize(pocket_emb, dim=-1)
                    loss = self.compute_classification_loss(mol_emb, pocket_emb, sample['target'])

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
        model.eval()
        pocket_reps = []
        res_single_list = []
        res_single_tmp_list = []
        with torch.no_grad():
            for _, sample in enumerate(tqdm(mol_data_test)):
                mol_predictions = []
                sample = unicore.utils.move_to_cuda(sample)
                smi_names = sample["smi_name"]
                mol_rep_ori = None

                for _, sample_pocket in enumerate(pocket_data):

                    sample_pocket = unicore.utils.move_to_cuda(sample_pocket)
                    smi_names = sample_pocket["pocket_name"]
                    pocket_rep_ori = None
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
                        # train=False,
                        inference=True,
                        mol_rep_ori=mol_rep_ori,
                        pocket_rep_ori=pocket_rep_ori,
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
        base_dir = "/root/autodl-tmp/Drug-fewshot/data/lit_pcba/"
        targets = os.listdir(base_dir)
        filtered_targets = []
        for t in targets:
            lmdb_path = os.path.join(base_dir, t, f"mols_remain_{self.args.ft}pos_{self.args.ft}neg.lmdb")
            if os.path.exists(lmdb_path):
                filtered_targets.append(t)
        # save filtered target names as npz
        np.savez(os.path.join(base_dir, f"targets_with_{self.args.ft}pos_{self.args.ft}neg.npz"), targets=np.array(filtered_targets))
        targets = filtered_targets

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

        df.to_csv(f'pcba_target_metrics_{self.args.ft}_{self.args.lr}_{self.args.epoch_train}_fewshot_{self.args.sample_time}_moltoken{self.args.mol_token}_pockettoken{self.args.pocket_token}_{datetime.now():%Y%m%d-%H%M%S}_layeradapter.csv', index=False)

        
        return

    def collect_params(self, model):
        model.requires_grad_(False)  

        params, names = [], []
        target_keywords = ['layer_adapters']


        for submodel in [model.mol_model, model.pocket_model]:
            for name, module in submodel.named_modules():
                if isinstance(module, LayerNorm):
                    module.requires_grad_(True)
                    for pname, param in module.named_parameters():
                        if pname in ['weight', 'bias']:
                            param.requires_grad_(True)
                            params.append(param)
                            names.append(f"{name}.{pname}")


        for submodel in [model.mol_model, model.pocket_model]:
            for name, param in submodel.named_parameters():
                if any(key in name for key in target_keywords):
                    param.requires_grad_(True)
                    params.append(param)
                    names.append(name)


        for name, param in model.named_parameters():
            if name.startswith(('mol_project', 'pocket_project')):
                param.requires_grad_(True)
                params.append(param)
                names.append(name)

        return params, names

    

    def test_dude(self, model, **kwargs):
        base_dir = "/root/autodl-tmp/Drug-fewshot/data/dude/raw/all/"
        targets = os.listdir(base_dir)
        filtered_targets = []
        for t in targets:
            lmdb_path = os.path.join(base_dir, t, f"mols_remain_{self.args.ft}pos_{self.args.ft}neg.lmdb")
            if os.path.exists(lmdb_path):
                filtered_targets.append(t)
        np.savez(os.path.join(base_dir, f"targets_with_{self.args.ft}pos_{self.args.ft}neg.npz"),
                 targets=np.array(filtered_targets))
        targets = filtered_targets
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
        df.to_csv(f'dude_target_metrics_{self.args.ft}_{self.args.lr}_{self.args.epoch_train}_fewshot_{self.args.sample_time}_moltoken{self.args.mol_token}_pockettoken{self.args.pocket_token}_{datetime.now():%Y%m%d-%H%M%S}_laryeradapter.csv', index=False)

        return
























import os
import pickle
import gzip
import lmdb
import random
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans


def read_lmdb(lmdb_path):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    txn = env.begin()
    keys = list(txn.cursor().iternext(values=False))
    inactive_list, active_list = [], []
    for idx in keys:
        datapoint_pickled = txn.get(idx)
        data = pickle.loads(datapoint_pickled)
        if data["label"] == 1:
            active_list.append(data)
        else:
            inactive_list.append(data)
    env.close()
    return active_list, inactive_list


def write_lmdb(out_list, save_path):
    env = lmdb.open(
        save_path,
        subdir=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=64,
        map_size=1099511627776,
    )
    with env.begin(write=True) as lmdb_txn:
        for i, item in enumerate(out_list):
            lmdb_txn.put(str(i).encode("ascii"), pickle.dumps(item))


def pad_features(feat_list):
    """把不同长度的 [X,256] pad 到相同长度"""
    max_len = max(feat.shape[0] for feat in feat_list)
    padded_feats = []
    for feat in feat_list:
        pad_len = max_len - feat.shape[0]
        if pad_len > 0:
            pad = np.zeros((pad_len, feat.shape[1]), dtype=feat.dtype)
            feat_padded = np.vstack([feat, pad])
        else:
            feat_padded = feat
        padded_feats.append(feat_padded.flatten())  # flatten 成 1D
    return np.stack(padded_feats)


import numpy as np
from scipy.spatial.distance import cdist
import random

def feature_split(active_list, inactive_list, n_active, features_dict):
    """基于 Unimol feature max-min greedy 选 few active，保证多样性"""
    active_with_feat = []
    feats_raw = []

    for mol in active_list:
        smi = mol["smi"]
        if smi in features_dict:
            feat = features_dict[smi]
            active_with_feat.append(mol)
            feats_raw.append(feat[0])  

    if len(active_with_feat) < n_active:
        return None, None

    feats_array = np.stack(feats_raw)  # [num_active, dim]

    # max-min greedy 选择 few
    selected_idx = []
    remaining_idx = list(range(len(active_with_feat)))

    # 先随机选一个
    first = random.choice(remaining_idx)
    selected_idx.append(first)
    remaining_idx.remove(first)

    while len(selected_idx) < n_active:
        # 计算剩余样本与已选样本的最小距离
        dist_matrix = cdist(feats_array[remaining_idx], feats_array[selected_idx], metric='euclidean')
        min_dist = dist_matrix.min(axis=1)  # 每个剩余样本到 selected 集合的最小距离
        pick = remaining_idx[np.argmax(min_dist)]  # 选最远的
        selected_idx.append(pick)
        remaining_idx.remove(pick)

    few_active = [active_with_feat[i] for i in selected_idx]
    remain_active = [active_with_feat[i] for i in remaining_idx]

    # inactive 随机采样
    if len(inactive_list) >= n_active:
        rand_idx = random.sample(range(len(inactive_list)), k=n_active)
        few_inactive = [inactive_list[i] for i in rand_idx]
        remain_inactive = [inactive_list[i] for i in range(len(inactive_list)) if i not in rand_idx]
    else:
        return None, None

    few_final = few_active + few_inactive
    remain_final = remain_active + remain_inactive

    print(f"[Feature Split] {len(few_active)} active + {len(few_inactive)} inactive "
          f"=> remain {len(remain_active)} active + {len(remain_inactive)} inactive")

    return few_final, remain_final


# ------------------ 主循环 ------------------
if __name__ == "__main__":
    random.seed(42)
    targets = os.listdir("./data/dude/raw/all/")
    few_sizes = [2, 4, 8, 16]

    for target in targets:
        data_path = f"./data/dude/raw/all/{target}/mols.lmdb"
        active_list, inactive_list = read_lmdb(data_path)

        feat_path = f"./embeddings_unimol/dude/{target}.npz.pkl.gz"
        with gzip.open(feat_path, "rb") as f:
            features_dict = pickle.load(f)

        for n in few_sizes:
            few_final, remain_final = feature_split(active_list, inactive_list, n_active=n, features_dict=features_dict)
            if few_final is None:
                continue

            few_path = data_path.replace("mols.lmdb", f"mols_few{n}_feature_2.lmdb")
            remain_path = data_path.replace("mols.lmdb", f"mols_remain{n}_feature_2.lmdb")

            write_lmdb(few_final, few_path)
            write_lmdb(remain_final, remain_path)

            print(f"[OK] target={target}, few{n} saved. "
                  f"(active={len(active_list)}, inactive={len(inactive_list)})")

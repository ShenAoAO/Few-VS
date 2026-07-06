import os
import pickle
import lmdb
# import selfies as sf
from tqdm import tqdm, trange


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
    inactive_list = []
    active_list = []
    for idx in tqdm(keys):
        datapoint_pickled = txn.get(idx)
        data = pickle.loads(datapoint_pickled)
        if data['label'] ==1:
            active_list.append(data)
        else:
            inactive_list.append(data)
        # out_list.append(data)
        # print(len(data["coordinates"]))
    env.close()
    return active_list, inactive_list


def write_lmdb(out_list, save_path):
    env = lmdb.open(
        save_path,
        subdir=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=62,
        map_size=1099511627776
    )

    with env.begin(write=True) as lmdb_txn:
        for i in tqdm(range(len(out_list))):
            lmdb_txn.put(str(i).encode('ascii'), pickle.dumps(out_list[i]))

import random
import numpy as np
random.seed(41)
targets = os.listdir("./data/dude/raw/all/")
# for i, target in enumerate(targets):
#     data_path = "./data/lit_pcba/" + target + "/mols.lmdb"
#     data_list,active_list = read_lmdb(data_path)
#     random_indices = random.sample(range(len(active_list)), 10)
#     random_list = [data_list[i] for i in random_indices]
#     remaining_list = [data_list[i] for i in range(len(data_list)) if i not in random_indices]
#     write_lmdb(random_list, data_path.replace('mols.lmdb', 'mols_few_10.lmdb'))
#     write_lmdb(remaining_list, data_path.replace('mols.lmdb', 'mols_remain_10.lmdb'))

for i, target in enumerate(targets):
    data_path = "./data/dude/raw/all/" + target + "/mols.lmdb"
    active_list, inactive_list = read_lmdb(data_path)
    # random_indices = random.sample(range(len(active_list)), 10)
    # random_list = [active_list[i] for i in random_indices]
    # remaining_list = [active_list[i] for i in range(len(active_list)) if i not in random_indices]+inactive_list
    # write_lmdb(random_list, data_path.replace('mols.lmdb', 'mols_few_10.lmdb'))
    # write_lmdb(remaining_list, data_path.replace('mols.lmdb', 'mols_remain_10.lmdb'))
    if len(active_list)<2:
        continue
    random_indices = random.sample(range(len(active_list)), 2)
    random_indices_in = random.sample(range(len(inactive_list)),2)
    random_list = [active_list[i] for i in random_indices]
    random_list_in = [inactive_list[i] for i in random_indices_in]
    remaining_list = [active_list[i] for i in range(len(active_list)) if i not in random_indices]+[inactive_list[j] for j in range(len(inactive_list)) if j not in random_indices_in]
    write_lmdb(random_list+random_list_in, data_path.replace('mols.lmdb', 'mols_few_2pos_2neg_2.lmdb'))
    write_lmdb(remaining_list, data_path.replace('mols.lmdb', 'mols_remain_2pos_2neg_2.lmdb'))
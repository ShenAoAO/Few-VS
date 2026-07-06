import os
import pickle
import lmdb
import random
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


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
    for idx in keys:
        datapoint_pickled = txn.get(idx)
        data = pickle.loads(datapoint_pickled)
        if data['label'] == 1:
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
        map_size=1099511627776
    )
    with env.begin(write=True) as lmdb_txn:
        for i in range(len(out_list)):
            lmdb_txn.put(str(i).encode('ascii'), pickle.dumps(out_list[i]))


def get_scaffold(smiles, include_chirality=False):
    """提取 Bemis-Murcko scaffold"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, isomericSmiles=include_chirality)


def scaffold_split(active_list, inactive_list, n_active):
    """基于 scaffold 选 few active，保证 remain scaffold 不重合"""
    # 给 active 分子加 scaffold
    active_with_scaf = []
    for mol in active_list:
        smiles = mol["smi"]
        scaf = get_scaffold(smiles) if smiles else None
        if scaf:
            active_with_scaf.append((mol, scaf))

    if len(active_with_scaf) < n_active:
        return None, None  # active 不够

    # scaffold 去重挑选 n_active
    random.shuffle(active_with_scaf)
    few_active, used_scaffolds = [], set()
    for mol, scaf in active_with_scaf:
        if scaf not in used_scaffolds:
            few_active.append(mol)
            used_scaffolds.add(scaf)
        if len(few_active) >= n_active:
            break

    if len(few_active) < n_active:
        return None, None  # scaffold 不够

    # remain active = scaffold 不在 few 里的
    # few_scaffolds = {get_scaffold(mol["smi"]) for mol in few_active}
    # remain_active = [mol for mol, scaf in active_with_scaf if scaf not in few_scaffolds]

    remain_active = [mol for mol, scaf in active_with_scaf if scaf not in used_scaffolds]

    # few inactive = 随机抽 n_active 个
    if len(inactive_list) >= n_active:
        random_indices_in = random.sample(range(len(inactive_list)), k=n_active)
        few_inactive = [inactive_list[i] for i in random_indices_in]
        remain_inactive = [inactive_list[i] for i in range(len(inactive_list)) if i not in random_indices_in]
    else:
        return None, None  # 负样本不够

    few_final = few_active + few_inactive
    remain_final = remain_active + remain_inactive
    return few_final, remain_final


# ------------------ 主循环 ------------------
random.seed(41)
targets = os.listdir("./data/lit_pcba/")
few_sizes = [2, 4, 8, 16]  # few-active 数量

for target in targets:
    data_path = f"./data/lit_pcba/{target}/mols.lmdb"
    active_list, inactive_list = read_lmdb(data_path)

    for n in few_sizes:
        few_final, remain_final = scaffold_split(active_list, inactive_list, n_active=n)
        if few_final is None:  # 样本数不足，跳过
            continue

        # 文件名区分
        few_path = data_path.replace('mols.lmdb', f'mols_few{n}_scaffold_2.lmdb')
        remain_path = data_path.replace('mols.lmdb', f'mols_remain{n}_scaffold_2.lmdb')

        write_lmdb(few_final, few_path)
        write_lmdb(remain_final, remain_path)

        print(f"[OK] target={target}, few{n} saved. (active={len(active_list)}, inactive={len(inactive_list)})")
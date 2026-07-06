# Interaction-Aware Adaptation for Few-Shot Virtual Screening on Specific Protein Targets



## Abstract

Identifying active compounds for a protein target from only a few known ligands remains a critical challenge in early-stage drug discovery, where experimental data are often extremely limited. Most existing few-shot virtual screening methods are ligand-centric, relying on molecular similarity or latent embeddings to extrapolate from scarce examples. However, the lack of explicit target awareness frequently leads to poor generalization and missed active candidates. We propose Few-VS, a target-aware learning paradigm designed for virtual screening under strict few-ligand settings. Few-VS enables rapid adaptation to a given target by introducing learnable prompt tokens that encode binding-relevant context, while lightweight adapter modules are used to improve alignment between pocket and ligand representations. These components are integrated within a Gated Prompt Adapter (GPA) architecture, in which interaction-informed signals dynamically modulate the contribution of the prompt, allowing the model to emphasize target-specific binding cues. Extensive experiments on four benchmark datasets show that Few-VS consistently outperforms zero-shot screening baselines. Importantly, it maintains strong retrieval performance even when active compounds share low structural similarity with the reference ligands, demonstrating robust generalization to previously unseen region.

---
## Requirements

Drug-few shares the same environment setup as **[Uni-Mol](https://github.com/dptech-corp/Uni-Mol/tree/main/unimol)**.


---

##  Data Preparation

1. Unzip the main dataset:
   ```bash
    unzip data.zip -d data/
   ```

2. Datasets included:
   - **DUD-E**
   - **PCBA**
   - **DEKOIS2.0**

3. Unzip precomputed embeddings:
   ```bash
    unzip embedding.zip -d embeddings/
   ```
   These contain **precomputed fine-grained molecular and pocket features**.

---

##  Folder Structure

```
project_root/
├── data/
│   ├── target/
│   │   ├── receptor.pdb
│   │   ├── crystal_ligand.mol2
│   │   ├── actives_final.ism
│   │   ├── decoys_final.ism
│   │   ├── mols.lmdb                 # all actives and decoys
│   │   ├── pocket.lmdb
│   │   ├── mol_few_xpos_xneg_n.lmdb  # random split (train)
│   │   ├── mol_remain_xpos_xneg_n.lmdb
│   │   ├── mol_fewx_scaffold_n.lmdb  # scaffold split (train)
│   │   ├── mol_remainx_scaffold_n.lmdb
│   │   ├── mol_fewx_feature_n.lmdb   # feature split (train)
│   │   ├── mol_remainx_feature_n.lmdb
│   └── ...
├── embeddings/                       # precomputed molecular embeddings
└── scripts/
    ├── main_results.sh
    └── ...
└── split
    ├── random_split.py
    ├── scaffold_split.py
    └── feature_split.py

```

---

## Data Splits

To generate training/testing splits:

```bash
   cd split
```

- **Random split**
  ```bash 
   python random_split.py
  ```

- **Scaffold split**
  ```bash
   python scaffold_split.py
  ```

- **Feature split**
  ```bash
   python feature_split.py
  ```

---

## Training & Inference

Run the main pipeline:

```bash
  bash main_results.sh
```

All training and inference results will be saved automatically in the corresponding output directories.

---




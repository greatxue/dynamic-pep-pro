import lmdb
import pickle
import torch
import numpy as np
import os
import csv

# ====== 配置 ======
lmdb_path = "/mlx_devbox/users/xuezhongkai/playground/dynamic-pep-pro/pep-data/PepMerge/pep_pocket_train_structure_cache.lmdb"
out_dir = "/mlx_devbox/users/xuezhongkai/playground/dynamic-pep-pro/pep-data/PepMerge/train"
os.makedirs(out_dir, exist_ok=True)

# ====== 氨基酸映射（按单字母字母序：A=ALA, C=CYS, D=ASP, ...）======
AA_MAP = [
    "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
    "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"
]

# ====== heavy atom 名字（简化，用于 PDB 写出）======
ATOM_NAMES = [
    "N", "CA", "C", "O", "CB", "CG", "CG1", "CG2",
    "CD", "CD1", "CD2", "CE", "CE1", "CE2", "CZ"
]


def write_pdb(coords, aatype, mask, chain_ids, save_path, chain_remap=None, include_chains=None):
    """
    写标准 PDB 文件。

    chain_remap:    dict，原始链名 → 新链名，e.g. {'B': 'A', 'A': 'B'}
    include_chains: set，只写这些原始链名；None 表示全写
    残基编号每条链（remap 后）从 1 单独开始计数。
    """
    atom_id = 1
    lines = []
    chain_res_counter = {}  # remap 后的链名 → 当前残基号

    for i in range(len(aatype)):
        orig_chain = chain_ids[i]

        if include_chains is not None and orig_chain not in include_chains:
            continue

        chain = chain_remap.get(orig_chain, orig_chain) if chain_remap else orig_chain

        # 每条链残基号从 1 开始
        if chain not in chain_res_counter:
            chain_res_counter[chain] = 0
        chain_res_counter[chain] += 1
        res_seq = chain_res_counter[chain]

        res_type = int(aatype[i])
        res_name = AA_MAP[res_type] if res_type < 20 else "UNK"

        for j in range(coords.shape[1]):
            if mask[i][j] < 0.5:
                continue

            x, y, z = coords[i][j]
            atom_name = ATOM_NAMES[j] if j < len(ATOM_NAMES) else f"X{j}"

            line = (
                f"ATOM  {atom_id:5d} {atom_name:<4} {res_name:<3} {chain}{res_seq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]}"
            )
            lines.append(line)
            atom_id += 1

    lines.append("END")
    with open(save_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ====== 打开 LMDB ======
env = lmdb.open(
    lmdb_path,
    readonly=True,
    lock=False,
    subdir=False
)

metadata_path = os.path.join(out_dir, "metadata.csv")
metadata_rows = []

with env.begin() as txn:
    cursor = txn.cursor()

    for idx, (key, value) in enumerate(cursor):
        data = pickle.loads(value)

        print(f"\n==== {key.decode()} ====")
        print(data.keys())

        coords    = data["pos_heavyatom"]
        mask      = data["mask_heavyatom"]
        aatype    = data["aa"]
        chain_ids = data["chain_id"]

        # 转 numpy
        if isinstance(coords, torch.Tensor):
            coords = coords.numpy()
        if isinstance(aatype, torch.Tensor):
            aatype = aatype.numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()

        chain_ids = [c.decode() if isinstance(c, bytes) else str(c) for c in chain_ids]

        print("coords:", coords.shape)
        print("aatype:", aatype.shape)
        print("chain_ids unique:", set(chain_ids))

        # ====== 确定 peptide / pocket 链 ======
        # 肽段是短链，受体口袋是长链
        unique_chains = sorted(set(chain_ids))
        chain_lengths = {c: chain_ids.count(c) for c in unique_chains}

        # 跳过只有一条链的样本（CroppingTransform2 无法处理）
        if len(unique_chains) < 2:
            print(f"[{idx+1}] SKIP: only 1 chain ({unique_chains}), skipping {key.decode()}")
            continue

        peptide_chain = min(chain_lengths, key=chain_lengths.get)  # 短链 = 肽段 binder
        pocket_chain  = max(chain_lengths, key=chain_lengths.get)  # 长链 = 受体 target

        # Proteina-Complexa 规范：Chain A = binder（peptide），Chain B = target（pocket）
        chain_remap = {peptide_chain: 'A', pocket_chain: 'B'}
        print(f"peptide(binder)={peptide_chain}({chain_lengths[peptide_chain]}残基) → A")
        print(f"pocket(target)={pocket_chain}({chain_lengths[pocket_chain]}残基) → B")

        # 第一个样本打印残基类型，验证 AA_MAP 是否正确
        if idx == 0:
            print("  [AA_MAP 验证] 前 20 个残基：")
            for i in range(min(20, len(aatype))):
                aa_idx = int(aatype[i])
                aa_name = AA_MAP[aa_idx] if aa_idx < 20 else "UNK"
                print(f"    res {i}: aatype={aa_idx} → {aa_name}  chain={chain_ids[i]}")

        # 保存路径
        base_name = key.decode()

        # 1. 整体 complex（A=peptide binder, B=pocket target）
        write_pdb(
            coords, aatype, mask, chain_ids,
            os.path.join(out_dir, f"{base_name}_complex.pdb"),
            chain_remap=chain_remap
        )

        # 2. peptide only → Chain A
        write_pdb(
            coords, aatype, mask, chain_ids,
            os.path.join(out_dir, f"{base_name}_peptide.pdb"),
            chain_remap=chain_remap,
            include_chains={peptide_chain}
        )

        # 3. pocket only → Chain B
        write_pdb(
            coords, aatype, mask, chain_ids,
            os.path.join(out_dir, f"{base_name}_pocket.pdb"),
            chain_remap=chain_remap,
            include_chains={pocket_chain}
        )

        complex_path = os.path.join(out_dir, f"{base_name}_complex.pdb")
        metadata_rows.append({
            "example_id": base_name,
            "path": complex_path,
            "num_residues_peptide": chain_lengths[peptide_chain],
            "num_residues_pocket": chain_lengths[pocket_chain],
        })

        print(f"[{idx+1}] saved: {base_name}")

# ====== 写 metadata CSV ======
with open(metadata_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["example_id", "path", "num_residues_peptide", "num_residues_pocket"])
    writer.writeheader()
    writer.writerows(metadata_rows)

print(f"\n✅ metadata CSV saved: {metadata_path}  ({len(metadata_rows)} entries)")

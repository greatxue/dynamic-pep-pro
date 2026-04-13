"""
生成 PepMerge 训练用 CSV 文件
输出两列：example_id, path（receptor_merge.pdb 的绝对路径）
"""

import os
import csv

# ── 配置 ──────────────────────────────────────────────────
DATA_DIR = "/Users/bytedance/Documents/peptides/dynamic-pep-pro/pep-data/PepMerge_release"
OUTPUT_CSV = "/Users/bytedance/Documents/peptides/dynamic-pep-pro/pep-data/pepmerge_receptor.csv"

# 过滤掉肽长度 < MIN_PEP_LEN 的样本（对应 binder_min_length: 3）
MIN_PEP_LEN = 3
# ─────────────────────────────────────────────────────────


def get_peptide_length(folder_path):
    fasta = os.path.join(folder_path, "peptide.fasta")
    if not os.path.exists(fasta):
        return 0
    seq = ""
    for line in open(fasta):
        if not line.startswith(">"):
            seq += line.strip()
    return len(seq)


rows = []
skipped_no_file = 0
skipped_short_pep = 0

for folder_name in sorted(os.listdir(DATA_DIR)):
    folder_path = os.path.join(DATA_DIR, folder_name)
    if not os.path.isdir(folder_path):
        continue

    pdb_path = os.path.join(folder_path, "receptor_merge.pdb")

    # 跳过没有 receptor_merge.pdb 的（约 125 个）
    if not os.path.exists(pdb_path):
        skipped_no_file += 1
        continue

    # 跳过肽太短的
    pep_len = get_peptide_length(folder_path)
    if pep_len < MIN_PEP_LEN:
        skipped_short_pep += 1
        continue

    rows.append({
        "example_id": folder_name,
        "path": os.path.abspath(pdb_path),
    })

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["example_id", "path"])
    writer.writeheader()
    writer.writerows(rows)

print(f"写入：{len(rows)} 条")
print(f"跳过（无 receptor_merge.pdb）：{skipped_no_file} 条")
print(f"跳过（肽长度 < {MIN_PEP_LEN}）：{skipped_short_pep} 条")
print(f"输出文件：{OUTPUT_CSV}")

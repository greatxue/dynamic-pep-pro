"""
Batch peptide generation for all PepMerge test cases.

For each pocket in metadata.csv, generates peptides matching the original
peptide length (10 samples per pocket).

Usage (run from Proteina-Complexa project root):
    python /path/to/pep-eval/run_batch_gen.py

Or with overrides:
    python /path/to/pep-eval/run_batch_gen.py \
        --metadata /path/to/metadata.csv \
        --test_dir /path/to/test \
        --output_dir /path/to/outputs \
        --n_samples 10
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

# =============================================================================
# Paths — edit these for your server environment
# =============================================================================
SERVER_ROOT = "/mlx_devbox/users/xuezhongkai/playground/dynamic-pep-pro"

DEFAULTS = dict(
    metadata   = f"{SERVER_ROOT}/pep-data/PepMerge/test/metadata.csv",
    test_dir   = f"{SERVER_ROOT}/pep-data/PepMerge/test",
    output_dir = f"{SERVER_ROOT}/pep-eval/gen_outputs",
    config_path= f"{SERVER_ROOT}/Proteina-Complexa/configs",
    config_name= "search_pepmerge_local_pipeline",
    project_dir= f"{SERVER_ROOT}/Proteina-Complexa",
    n_samples  = 10,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata",    default=DEFAULTS["metadata"])
    p.add_argument("--test_dir",    default=DEFAULTS["test_dir"])
    p.add_argument("--output_dir",  default=DEFAULTS["output_dir"])
    p.add_argument("--config_path", default=DEFAULTS["config_path"])
    p.add_argument("--config_name", default=DEFAULTS["config_name"])
    p.add_argument("--project_dir", default=DEFAULTS["project_dir"])
    p.add_argument("--n_samples",   type=int, default=DEFAULTS["n_samples"])
    p.add_argument("--start",       type=int, default=0,
                   help="Skip first N entries (for resuming)")
    p.add_argument("--end",         type=int, default=None,
                   help="Stop after this many entries total")
    p.add_argument("--dry_run",     action="store_true",
                   help="Print commands without executing")
    return p.parse_args()


def main():
    args = parse_args()

    # Read metadata
    rows = []
    with open(args.metadata, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    subset = rows[args.start : args.end]
    total = len(subset)
    print(f"Processing {total} entries (indices {args.start}–{args.start + total - 1})")

    os.makedirs(args.output_dir, exist_ok=True)

    for i, row in enumerate(subset):
        example_id  = row["example_id"]
        pep_len     = int(row["num_residues_peptide"])
        pocket_pdb  = os.path.join(args.test_dir, f"{example_id}_pocket.pdb")
        root_path   = os.path.join(args.output_dir, example_id)

        if not os.path.exists(pocket_pdb):
            print(f"[{i+1}/{total}] SKIP {example_id}: pocket PDB not found at {pocket_pdb}")
            continue

        print(f"\n[{i+1}/{total}] {example_id}  pep_len={pep_len}  → {root_path}")

        cmd = [
            "python", "src/proteinfoundation/generate.py",
            f"--config-path={args.config_path}",
            f"--config-name={args.config_name}",
            # Pocket PDB and chain
            f"++generation.dataloader.dataset.conditional_features.0.pdb_path={pocket_pdb}",
            f"++generation.dataloader.dataset.conditional_features.0.input_spec=B",
            # Fixed peptide length = ground-truth length
            f"++generation.dataloader.dataset.nres.low={pep_len}",
            f"++generation.dataloader.dataset.nres.high={pep_len}",
            f"++generation.dataloader.dataset.nres.nsamples={args.n_samples}",
            # Output directory (one per example)
            f"++root_path={root_path}",
            # Use example_id as run name for easier identification
            f"++run_name={example_id}",
        ]

        if args.dry_run:
            print("DRY RUN:", " ".join(cmd))
            continue

        result = subprocess.run(cmd, cwd=args.project_dir)
        if result.returncode != 0:
            print(f"  ERROR: generate.py exited with code {result.returncode} for {example_id}")
            # Continue with next entry instead of aborting
        else:
            print(f"  OK: {example_id}")

    print(f"\nDone. Outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()

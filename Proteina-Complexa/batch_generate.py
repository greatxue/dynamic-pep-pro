"""
Batch generation script for PepMerge test set.

For each pocket in metadata.csv, generates 10 peptide binders
of the same length as the reference peptide.

Usage (run from project root on server):
    python batch_generate.py

Or to resume from a specific index:
    python batch_generate.py --start 50

Or to run a specific range:
    python batch_generate.py --start 0 --end 10
"""

import argparse
import csv
import os
import subprocess
import sys


# =============================================================================
# Configuration — edit these paths before running on server
# =============================================================================
METADATA_CSV = "/mlx_devbox/users/xuezhongkai/playground/dynamic-pep-pro/pep-data/PepMerge/test/metadata.csv"
TEST_DIR     = "/mlx_devbox/users/xuezhongkai/playground/dynamic-pep-pro/pep-data/PepMerge/test"
OUTPUT_BASE  = "/mlx_devbox/users/xuezhongkai/playground/dynamic-pep-pro/Proteina-Complexa/outputs/pepmerge_eval"
CONFIG_PATH  = "/mlx_devbox/users/xuezhongkai/playground/dynamic-pep-pro/Proteina-Complexa/configs"
CONFIG_NAME  = "search_pepmerge_local_pipeline"

# Number of samples to generate per target
N_SAMPLES = 10
# =============================================================================


def read_metadata(csv_path):
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def already_done(root_path, config_name):
    """Check if this target already has results (allows resuming)."""
    csv_path = os.path.join(root_path, "..", f"results_{config_name}_0.csv")
    return os.path.exists(os.path.abspath(csv_path))


def run_one(example_id, pep_len, pocket_pdb, root_path):
    """Run generate.py for a single target."""
    cmd = [
        sys.executable,
        "src/proteinfoundation/generate.py",
        f"--config-path={CONFIG_PATH}",
        f"--config-name={CONFIG_NAME}",
        # Target pocket PDB
        f"++generation.dataloader.dataset.conditional_features.0.pdb_path={pocket_pdb}",
        f"++generation.dataloader.dataset.conditional_features.0.input_spec=B",
        # Fixed peptide length, 10 samples
        f"++generation.dataloader.dataset.nres.low={pep_len}",
        f"++generation.dataloader.dataset.nres.high={pep_len}",
        f"++generation.dataloader.dataset.nres.nsamples={N_SAMPLES}",
        # Batch size = N_SAMPLES so all go into one batch
        f"++generation.dataloader.batch_size={N_SAMPLES}",
        # Per-target output directory
        f"++root_path={root_path}",
        # Tag this run
        f"++run_name={example_id}",
    ]
    print(f"\n{'='*60}")
    print(f"[{example_id}]  pep_len={pep_len}  pocket={pocket_pdb}")
    print(f"  output -> {root_path}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[WARN] {example_id} exited with code {result.returncode}, continuing...")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end",   type=int, default=None, help="End index (exclusive), default=all")
    args = parser.parse_args()

    rows = read_metadata(METADATA_CSV)
    subset = rows[args.start : args.end]
    total = len(subset)

    print(f"Processing {total} targets (index {args.start} to {args.start + total - 1})")
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    success, skipped, failed = 0, 0, 0

    for i, row in enumerate(subset):
        example_id  = row["example_id"]
        pep_len     = int(row["num_residues_peptide"])
        pocket_pdb  = os.path.join(TEST_DIR, f"{example_id}_pocket.pdb")
        root_path   = os.path.join(OUTPUT_BASE, example_id)

        if not os.path.exists(pocket_pdb):
            print(f"[{i+1}/{total}] SKIP {example_id}: pocket PDB not found at {pocket_pdb}")
            skipped += 1
            continue

        if already_done(root_path, CONFIG_NAME):
            print(f"[{i+1}/{total}] SKIP {example_id}: results already exist")
            skipped += 1
            continue

        ok = run_one(example_id, pep_len, pocket_pdb, root_path)
        if ok:
            success += 1
        else:
            failed += 1

        print(f"\nProgress: {i+1}/{total}  (done={success}, skipped={skipped}, failed={failed})")

    print(f"\n{'='*60}")
    print(f"Batch complete: {success} succeeded, {skipped} skipped, {failed} failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

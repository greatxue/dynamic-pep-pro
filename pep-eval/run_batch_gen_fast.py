"""
Batch peptide generation — fast variant that loads the checkpoint ONCE.

Unlike run_batch_gen.py (which spawns a subprocess per example and reloads the
checkpoint every time), this script:
  1. Loads the checkpoint once before the loop.
  2. Creates the Lightning Trainer once before the loop.
  3. Per example: updates the Hydra config in-place, re-instantiates only the
     dataloader, and calls trainer.predict().

Usage (run from anywhere — paths are resolved via CLI args):
    python pep-eval/run_batch_gen_fast.py --dry_run --start 0 --end 3
    python pep-eval/run_batch_gen_fast.py --start 0 --end 2 --output_dir /tmp/fast_test
"""

import argparse
import csv
import os
import sys
from pathlib import Path

# =============================================================================
# Paths — edit these for your server environment
# =============================================================================
SERVER_ROOT = "/mlx_devbox/users/xuezhongkai/playground/dynamic-pep-pro"

DEFAULTS = dict(
    metadata   = f"{SERVER_ROOT}/pep-data/PepMerge/test/metadata.csv",
    test_dir   = f"{SERVER_ROOT}/pep-data/PepMerge/test",
    output_dir = f"{SERVER_ROOT}/pep-eval/gen_outputs_fast",
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
    p.add_argument("--batch_size",  type=int, default=None,
                   help="Dataloader batch size (defaults to n_samples)")
    p.add_argument("--start",       type=int, default=0,
                   help="Skip first N entries (for resuming)")
    p.add_argument("--end",         type=int, default=None,
                   help="Stop after this many entries total")
    p.add_argument("--dry_run",     action="store_true",
                   help="Print planned runs without executing")
    return p.parse_args()


def main():
    args = parse_args()
    if args.batch_size is None:
        args.batch_size = args.n_samples

    # ------------------------------------------------------------------
    # 1. Read metadata
    # ------------------------------------------------------------------
    rows = []
    with open(args.metadata, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    subset = rows[args.start : args.end]
    total = len(subset)
    print(f"Processing {total} entries (indices {args.start}–{args.start + total - 1})")

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Dry run — just print what would be done
    # ------------------------------------------------------------------
    if args.dry_run:
        for i, row in enumerate(subset):
            example_id = row["example_id"]
            pep_len    = int(row["num_residues_peptide"])
            pocket_pdb = os.path.join(args.test_dir, f"{example_id}_pocket.pdb")
            root_path  = os.path.join(args.output_dir, example_id)
            exists     = os.path.exists(pocket_pdb)
            print(
                f"[{i+1}/{total}] {example_id}  pep_len={pep_len}"
                f"  pocket={'OK' if exists else 'MISSING'}  → {root_path}"
            )
        return

    # ------------------------------------------------------------------
    # 3. Add project src/ to sys.path and chdir to project root
    #    (must happen before importing proteinfoundation)
    # ------------------------------------------------------------------
    project_dir = os.path.abspath(args.project_dir)
    src_dir = os.path.join(project_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    os.chdir(project_dir)

    # ------------------------------------------------------------------
    # 4. Apply patches + quiet_startup BEFORE any other proteinfoundation
    #    imports (mirrors the top of generate.py)
    # ------------------------------------------------------------------
    import proteinfoundation.patches.atomworks_patches  # noqa: F401
    from proteinfoundation.cli.startup import quiet_startup
    quiet_startup()

    # ------------------------------------------------------------------
    # 5. Heavy imports (after patches)
    # ------------------------------------------------------------------
    import hydra
    import lightning as L
    import torch
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import open_dict

    from proteinfoundation.generate import (
        load_ckpt_n_configure_inference,
        save_predictions,
        save_rewards_to_csv,
        validate_checkpoint_paths,
    )

    # ------------------------------------------------------------------
    # 6. Initialise Hydra with the config directory (works outside
    #    @hydra.main as long as we use initialize_config_dir + compose)
    # ------------------------------------------------------------------
    config_dir = os.path.abspath(args.config_path)
    GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=config_dir, version_base=None)
    cfg = hydra.compose(config_name=args.config_name)

    # ------------------------------------------------------------------
    # 7. Validate checkpoint paths once
    # ------------------------------------------------------------------
    validate_checkpoint_paths(cfg)

    # ------------------------------------------------------------------
    # 8. Load model ONCE — the expensive step
    # ------------------------------------------------------------------
    model = load_ckpt_n_configure_inference(cfg)
    torch.set_float32_matmul_precision("high")

    # ------------------------------------------------------------------
    # 9. Create Trainer ONCE
    # ------------------------------------------------------------------
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        inference_mode=False,
    )

    # ------------------------------------------------------------------
    # 10. Per-example loop
    # ------------------------------------------------------------------
    for i, row in enumerate(subset):
        example_id = row["example_id"]
        pep_len    = int(row["num_residues_peptide"])
        pocket_pdb = os.path.join(args.test_dir, f"{example_id}_pocket.pdb")
        root_path  = os.path.join(args.output_dir, example_id)

        if not os.path.exists(pocket_pdb):
            print(f"[{i+1}/{total}] SKIP {example_id}: pocket PDB not found at {pocket_pdb}")
            continue

        print(f"\n[{i+1}/{total}] {example_id}  pep_len={pep_len}  → {root_path}")
        os.makedirs(root_path, exist_ok=True)

        try:
            # a. Update per-example config fields in-place
            with open_dict(cfg):
                feat = cfg.generation.dataloader.dataset.conditional_features[0]
                feat.pdb_path   = pocket_pdb
                feat.input_spec = "B"
                cfg.generation.dataloader.dataset.nres.low      = pep_len
                cfg.generation.dataloader.dataset.nres.high     = pep_len
                cfg.generation.dataloader.dataset.nres.nsamples = args.n_samples
                cfg.generation.dataloader.batch_size            = args.batch_size

            # b. Reconfigure inference (cheap: just sets self.inf_cfg)
            model.configure_inference(cfg.generation, nn_ag=None)

            # c. Instantiate a fresh dataloader for this example
            dataloader = hydra.utils.instantiate(cfg.generation.dataloader)

            # d. Generate
            predictions = trainer.predict(model, dataloader)

            # e. Save PDBs
            _pdb_paths, reward_df = save_predictions(root_path, predictions, job_id=0)

            # f. Save rewards CSV if non-empty
            if len(reward_df) > 0:
                save_rewards_to_csv(
                    df=reward_df,
                    root_path=root_path,
                    config_name=args.config_name,
                    job_id=0,
                )

            print(f"  OK: {example_id}")

        except Exception as exc:
            print(f"  ERROR: {example_id}: {exc}")
            # Continue with next entry

    print(f"\nDone. Outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()

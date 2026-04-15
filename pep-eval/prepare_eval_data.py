#!/usr/bin/env python3
"""
Prepare generated PDB results for evaluation.

Converts the generation output structure into the format expected by
evaluate/calculate_*.py scripts:
  {output_dir}/{pdb_id}/ref.pdb
  {output_dir}/{pdb_id}/sample_0.pdb
  {output_dir}/{pdb_id}/sample_1.pdb
  ...

Chain convention in reference complexes: A = peptide, B = receptor
Chain convention in generated PDBs:      A = receptor, B = peptide
This script swaps A<->B in the generated files so they match the reference.

Also writes a mapping.txt: "{pdb_id} B A" (rec_chain pep_chain).
"""

import argparse
import os
import re
import shutil
from pathlib import Path


def swap_chains_ab(pdb_in: str, pdb_out: str) -> None:
    """
    Write pdb_out with chain A and chain B swapped.
    Uses a two-pass approach via a temp character to avoid collisions.
    Only modifies columns 21 (0-indexed) in ATOM/HETATM/TER lines.
    """
    CHAIN_COL = 21  # 0-indexed column for chain ID in PDB format

    lines = []
    with open(pdb_in, "r") as f:
        for line in f:
            record = line[:6].strip()
            if record in ("ATOM", "HETATM", "TER") and len(line) > CHAIN_COL:
                ch = line[CHAIN_COL]
                if ch == "A":
                    line = line[:CHAIN_COL] + "B" + line[CHAIN_COL + 1 :]
                elif ch == "B":
                    line = line[:CHAIN_COL] + "A" + line[CHAIN_COL + 1 :]
            lines.append(line)

    with open(pdb_out, "w") as f:
        f.writelines(lines)


def find_generated_pdbs(entry_dir: Path):
    """
    Collect generated PDB files from job subdirectories,
    sorted by job index extracted from directory name.
    Returns list of Path objects.
    """
    job_dirs = sorted(
        [d for d in entry_dir.iterdir() if d.is_dir()],
        key=lambda d: int(m.group(1)) if (m := re.search(r"_id_(\d+)_", d.name)) else 999,
    )
    pdbs = []
    for jd in job_dirs:
        # The PDB inside has the same name as the directory
        pdb = jd / f"{jd.name}.pdb"
        if pdb.exists():
            pdbs.append(pdb)
        else:
            # Fallback: any pdb in the dir
            candidates = list(jd.glob("*.pdb"))
            if candidates:
                pdbs.append(candidates[0])
    return pdbs


def main():
    parser = argparse.ArgumentParser(description="Prepare generated PDBs for evaluation.")
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Directory containing generation results (one subdir per pdb_id).",
    )
    parser.add_argument(
        "--test_dir",
        required=True,
        help="Directory containing reference *_complex.pdb files.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for eval-ready data.",
    )
    parser.add_argument(
        "--mapping_file",
        default=None,
        help="Path to write mapping.txt. Defaults to {output_dir}/mapping.txt.",
    )
    parser.add_argument(
        "--swap_chains",
        action="store_true",
        default=True,
        help="Swap chain A<->B in generated files (default: True).",
    )
    parser.add_argument(
        "--no_swap_chains",
        dest="swap_chains",
        action="store_false",
        help="Do NOT swap chains in generated files.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    test_dir = Path(args.test_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    mapping_file = Path(args.mapping_file) if args.mapping_file else output_dir / "mapping.txt"

    output_dir.mkdir(parents=True, exist_ok=True)

    entries = sorted([d.name for d in results_dir.iterdir() if d.is_dir()])
    print(f"Found {len(entries)} entries in {results_dir}")

    mapping_lines = []
    total_samples = 0

    for entry in entries:
        entry_dir = results_dir / entry
        out_entry_dir = output_dir / entry
        out_entry_dir.mkdir(parents=True, exist_ok=True)

        # --- Reference PDB ---
        ref_src = test_dir / f"{entry}_complex.pdb"
        if not ref_src.exists():
            print(f"  [SKIP] No reference complex found for {entry}: {ref_src}")
            continue

        ref_dst = out_entry_dir / "ref.pdb"
        shutil.copy2(ref_src, ref_dst)

        # --- Generated samples ---
        pdbs = find_generated_pdbs(entry_dir)
        if not pdbs:
            print(f"  [WARN] No generated PDBs found for {entry}")
            continue

        for i, src_pdb in enumerate(pdbs):
            dst_pdb = out_entry_dir / f"sample_{i}.pdb"
            if args.swap_chains:
                swap_chains_ab(str(src_pdb), str(dst_pdb))
            else:
                shutil.copy2(src_pdb, dst_pdb)

        total_samples += len(pdbs)

        # Mapping: rec_chain=B, pep_chain=A (after swap, peptide is A)
        # Format expected by eval scripts: pdb_id rec_chain pep_chain
        mapping_lines.append(f"{entry} B A")
        print(f"  {entry}: ref.pdb + {len(pdbs)} samples")

    with open(mapping_file, "w") as f:
        f.write("\n".join(mapping_lines) + "\n")

    print(f"\nDone. {len(mapping_lines)} entries, {total_samples} total samples.")
    print(f"Output:  {output_dir}")
    print(f"Mapping: {mapping_file}")
    print(f"\nNext step — run evaluations:")
    print(f"  cd /Users/bytedance/Documents/peptides/dynamic-pep-pro/pep-eval")
    print(f"  python evaluate/run_all_evaluations.py \\")
    print(f"    --data_dir {output_dir} \\")
    print(f"    --mapping_file {mapping_file} \\")
    print(f"    --output_dir evaluate/eval_results \\")
    print(f"    --metrics aar,rmsd,ssr,diversity,connectivity")


if __name__ == "__main__":
    main()

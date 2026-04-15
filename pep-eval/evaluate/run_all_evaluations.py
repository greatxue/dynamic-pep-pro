#!/usr/bin/env python3
"""One-click runner for evaluation scripts in this folder.

Example:
    python evaluate/run_all_evaluations.py \
        --data_dir /path/to/generated_pdbs \
        --mapping_file /path/to/test.txt
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_MAPPING = "/home/shiyiming/peptide_design/ManifoldSurface_Peptide_Generator/data/raw/LNR/test.txt"

ALL_METRICS = [
    "aar",
    "bsr",
    "rmsd",
    "ssr",
    "spatial_ssr_v2",
    "diversity",
    "consistency",
    "tm_score",
    "novelty",
    "dockq",
    "posecheck",
    "posecheck_parallel",
    "esmfold_rmsd",
    "rosetta_manifold",
    "connectivity",
]

DEFAULT_METRICS = ALL_METRICS.copy()


@dataclass
class Step:
    metric: str
    cmd: List[str]
    required: bool = True
    skip_reason: Optional[str] = None
    out_path: Optional[str] = None


def parse_metrics(raw: str) -> List[str]:
    value = raw.strip().lower()
    if value == "all":
        return ALL_METRICS.copy()

    metrics = [m.strip().lower() for m in value.split(",") if m.strip()]
    unknown = [m for m in metrics if m not in ALL_METRICS]
    if unknown:
        raise ValueError(f"Unknown metric(s): {unknown}. Supported: {ALL_METRICS}")

    # Keep user order but remove duplicates.
    seen = set()
    ordered = []
    for m in metrics:
        if m not in seen:
            seen.add(m)
            ordered.append(m)
    return ordered


def run_step(step: Step, cwd: Path, dry_run: bool) -> Dict[str, object]:
    print("\n" + "=" * 80)
    print(f"[RUN] {step.metric}")
    print(" ".join(step.cmd))

    started = time.time()

    if step.skip_reason:
        print(f"[SKIP] {step.metric}: {step.skip_reason}")
        return {
            "metric": step.metric,
            "status": "skipped",
            "return_code": None,
            "duration_sec": 0.0,
            "command": step.cmd,
            "reason": step.skip_reason,
        }

    if dry_run:
        print(f"[DRY-RUN] {step.metric} not executed")
        return {
            "metric": step.metric,
            "status": "dry-run",
            "return_code": None,
            "duration_sec": 0.0,
            "command": step.cmd,
        }

    completed = subprocess.run(step.cmd, cwd=str(cwd))
    elapsed = time.time() - started

    status = "success" if completed.returncode == 0 else "failed"
    print(f"[DONE] {step.metric} status={status} code={completed.returncode} time={elapsed:.1f}s")

    return {
        "metric": step.metric,
        "status": status,
        "return_code": completed.returncode,
        "duration_sec": round(elapsed, 3),
        "command": step.cmd,
    }


def build_steps(args: argparse.Namespace, scripts_dir: Path, output_dir: Path) -> List[Step]:
    py = sys.executable
    tm_score_path = output_dir / "tm_score_results.txt"

    steps_by_metric: Dict[str, Step] = {
        "aar": Step(
            metric="aar",
            cmd=[
                py,
                str(scripts_dir / "calculate_aar.py"),
                "--data_dir",
                args.data_dir,
                "--mapping_file",
                args.mapping_file,
                "--output_file",
                str(output_dir / "aar_results.txt"),
            ],
        ),
        "bsr": Step(
            metric="bsr",
            cmd=[
                py,
                str(scripts_dir / "calculate_bsr.py"),
                "--data_dir",
                args.data_dir,
                "--mapping_file",
                args.mapping_file,
                "--output_file",
                str(output_dir / "bsr_results.txt"),
                "--threshold",
                str(args.bsr_threshold),
            ],
        ),
        "rmsd": Step(
            metric="rmsd",
            cmd=[
                py,
                str(scripts_dir / "calculate_rmsd.py"),
                "--data_dir",
                args.data_dir,
                "--mapping_file",
                args.mapping_file,
                "--output_file",
                str(output_dir / "rmsd_results.txt"),
            ],
        ),
        "ssr": Step(
            metric="ssr",
            cmd=[
                py,
                str(scripts_dir / "calculate_ssr.py"),
                "--data_dir",
                args.data_dir,
                "--mapping_file",
                args.mapping_file,
                "--output_file",
                str(output_dir / "ssr_results.txt"),
            ],
        ),
        "spatial_ssr_v2": Step(
            metric="spatial_ssr_v2",
            cmd=[
                py,
                str(scripts_dir / "calculate_spatial_ssr_v2.py"),
                "--data_dir",
                args.data_dir,
                "--mapping_file",
                args.mapping_file,
                "--output_file",
                str(output_dir / "spatial_ssr_v2_results.txt"),
                "--radius",
                str(args.spatial_radius),
            ],
        ),
        "diversity": Step(
            metric="diversity",
            cmd=[
                py,
                str(scripts_dir / "calculate_diversity.py"),
                "--data_dir",
                args.data_dir,
                "--mapping_file",
                args.mapping_file,
                "--output_file",
                str(output_dir / "diversity_results.txt"),
            ],
        ),
        "consistency": Step(
            metric="consistency",
            cmd=[
                py,
                str(scripts_dir / "calculate_consistency.py"),
                "--data_dir",
                args.data_dir,
                "--mapping_file",
                args.mapping_file,
                "--output_file",
                str(output_dir / "consistency_results.txt"),
            ],
        ),
        "tm_score": Step(
            metric="tm_score",
            cmd=[
                py,
                str(scripts_dir / "calculate_tm_score.py"),
                "--data_dir",
                args.data_dir,
                "--mapping_file",
                args.mapping_file,
                "--output_file",
                str(tm_score_path),
                "--workers",
                str(args.workers),
            ],
        ),
        "novelty": Step(
            metric="novelty",
            cmd=[
                py,
                str(scripts_dir / "calculate_novelty.py"),
                "--tm_score_file",
                str(tm_score_path),
                "--data_dir",
                args.data_dir,
                "--mapping_file",
                args.mapping_file,
                "--output_file",
                str(output_dir / "novelty_results.csv"),
            ],
        ),
        "dockq": Step(
            metric="dockq",
            cmd=[
                py,
                str(scripts_dir / "calculate_dockq.py"),
                "--data_dir",
                args.data_dir,
                "--mapping_file",
                args.mapping_file,
                "--output_file",
                str(output_dir / "dockq_results.txt"),
                "--workers",
                str(args.workers),
            ],
        ),
        "posecheck": Step(
            metric="posecheck",
            cmd=[
                py,
                str(scripts_dir / "calculate_posecheck.py"),
                "--data_dir",
                args.data_dir,
                "--mapping_file",
                args.mapping_file,
                "--output_file",
                str(output_dir / "posecheck_results.csv"),
            ],
        ),
        "posecheck_parallel": Step(
            metric="posecheck_parallel",
            cmd=[
                py,
                str(scripts_dir / "calculate_posecheck_parallel.py"),
                "--data_dir",
                args.data_dir,
                "--reference_dir",
                args.reference_dir or "",
                "--mapping_file",
                args.mapping_file,
                "--output_file",
                str(output_dir / "posecheck_parallel_results.csv"),
            ],
        ),
        "esmfold_rmsd": Step(
            metric="esmfold_rmsd",
            cmd=[
                py,
                str(scripts_dir / "calculate_esmfold_rmsd.py"),
                "--data_dir",
                args.data_dir,
                "--mapping_file",
                args.mapping_file,
                "--model_name",
                args.esmfold_model_name,
                "--num_samples",
                str(args.esmfold_num_samples),
                "--output",
                str(output_dir / "esmfold_results"),
            ],
        ),
        "rosetta_manifold": Step(
            metric="rosetta_manifold",
            cmd=[
                py,
                str(scripts_dir / "calculate_rosetta_manifold.py"),
                "--data_dir",
                args.data_dir,
                "--mapping_file",
                args.mapping_file,
                "--output_file",
                str(output_dir / "rosetta_manifold_results.txt"),
                "--n_cpus",
                str(args.rosetta_n_cpus),
            ],
        ),
        "connectivity": Step(
            metric="connectivity",
            cmd=[
                py,
                str(scripts_dir / "calculate_connectivity.py"),
                "--data_dir",
                args.data_dir,
                "--mapping_file",
                args.mapping_file,
                "--output_file",
                str(output_dir / "connectivity_results.txt"),
            ],
        ),
    }

    selected = []
    for metric in args.metrics:
        step = steps_by_metric[metric]
        if metric == "posecheck_parallel" and not args.reference_dir:
            selected.append(
                Step(metric=metric, cmd=step.cmd, required=False, skip_reason="Missing --reference_dir")
            )
            continue
        selected.append(step)

    # Enforce dependency order: novelty requires tm_score output.
    if "novelty" in [s.metric for s in selected] and "tm_score" not in [s.metric for s in selected]:
        selected.insert(0, steps_by_metric["tm_score"])

    if args.skip_existing:
        for step in selected:
            # Always try to find expected output file
            out_path = None
            for i, part in enumerate(step.cmd):
                if part in ["--output_file"]:
                    out_path = Path(step.cmd[i+1])
                    break
                elif part == "--output" and step.metric == "esmfold_rmsd":
                    # For esmfold, check if any csv files are generated with the prefix
                    out_prefix = step.cmd[i+1]
                    base_dir = os.path.dirname(out_prefix)
                    if os.path.exists(base_dir):
                        for f in os.listdir(base_dir):
                            if os.path.basename(out_prefix) in f and f.endswith(".csv"):
                                out_path = Path(os.path.join(base_dir, f))
                                break
                    break
            
            if out_path and out_path.exists() and out_path.stat().st_size > 0:
                step.skip_reason = f"Output already exists ({out_path.name})"

    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-click runner for evaluate/calculate_*.py scripts")

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to generated PDB directory",
    )
    parser.add_argument(
        "--mapping_file",
        type=str,
        default=DEFAULT_MAPPING,
        help="Path to chain mapping file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluate/all_eval_results",
        help="Directory to store all outputs",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRICS),
        help=f"Comma-separated metrics or 'all'. Supported: {ALL_METRICS}",
    )
    parser.add_argument(
        "--reference_dir",
        type=str,
        default=None,
        help="Required only for posecheck_parallel",
    )

    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 4))
    parser.add_argument("--bsr_threshold", type=float, default=6.0)
    parser.add_argument("--spatial_radius", type=float, default=2.0)

    parser.add_argument("--esmfold_model_name", type=str, default="custom_model")
    parser.add_argument("--esmfold_num_samples", type=int, default=40)
    parser.add_argument("--rosetta_n_cpus", type=int, default=-1)

    parser.add_argument("--continue_on_error", action="store_true", help="Continue running after a failed metric")
    parser.add_argument("--skip_existing", action="store_true", help="Skip metric if output file already exists and is not empty")
    parser.add_argument("--dry_run", action="store_true", help="Only print commands without executing")

    args = parser.parse_args()
    args.metrics = parse_metrics(args.metrics)
    return args


def main() -> None:
    args = parse_args()

    scripts_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.data_dir).exists():
        raise FileNotFoundError(f"data_dir does not exist: {args.data_dir}")
    if not Path(args.mapping_file).exists():
        raise FileNotFoundError(f"mapping_file does not exist: {args.mapping_file}")

    steps = build_steps(args, scripts_dir, output_dir)

    print("=" * 80)
    print("One-click evaluation runner")
    print(f"Data dir: {args.data_dir}")
    print(f"Mapping : {args.mapping_file}")
    print(f"Output  : {output_dir}")
    print(f"Metrics : {[s.metric for s in steps]}")
    print("=" * 80)

    records = []
    has_failure = False

    for step in steps:
        result = run_step(step, cwd=scripts_dir, dry_run=args.dry_run)
        records.append(result)

        if result["status"] == "failed":
            has_failure = True
            if not args.continue_on_error:
                print("[STOP] Aborting due to failure. Use --continue_on_error to continue.")
                break
        elif result["status"] in ["success", "skipped"]:
            # RUN SUMMARY SCRIPT
            # find actual out_path if it was generated
            if step.out_path and os.path.exists(step.out_path):
                print(f"[SUMMARY] Generating statistics and plot for {step.metric}...")
                sum_cmd = [
                    sys.executable, 
                    str(scripts_dir / "summarize_results.py"),
                    "--input", step.out_path,
                    "--output_dir", str(output_dir)
                ]
                subprocess.run(sum_cmd, cwd=str(scripts_dir))

    report_path = output_dir / "run_all_evaluations_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    success_count = sum(1 for r in records if r["status"] == "success")
    skipped_count = sum(1 for r in records if r["status"] == "skipped")
    failed_count = sum(1 for r in records if r["status"] == "failed")

    print("\n" + "=" * 80)
    print("Finished")
    print(f"Success: {success_count}, Skipped: {skipped_count}, Failed: {failed_count}")
    print(f"Report : {report_path}")

    if has_failure and not args.dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()

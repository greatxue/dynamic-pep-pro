#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ManifoldSurface Rosetta Energy Calculation (Ray Parallel Ver.)
Adapted from calculate_native_binding_energy_ray.py.
Uses PepMimic's evaluation modules which already wrap PyRosetta functions.
"""

import os
import sys
import time
import argparse
import ray
from dataclasses import dataclass
from typing import List, Optional

# ===== Configuration =====
# Add PepMimic path to sys.path so we can import its modules
PEPMIMIC_PATH = '/home/shiyiming/peptide_design/PepMimic'
if PEPMIMIC_PATH not in sys.path:
    sys.path.insert(0, PEPMIMIC_PATH)

from evaluation.dG.energy import pyrosetta_fastrelax, pyrosetta_interface_energy


@dataclass
class ManifoldTask:
    """Manifold Generated Structure Task"""
    pdb_id: str
    model_name: str
    pdb_file: str      # Original generated PDB
    relaxed_pdb: str   # Output path for relaxed PDB
    lig_chain: str
    rec_chains: List[str] # Receptor chains
    status: str = 'created'
    dG: Optional[float] = None
    error: Optional[str] = None
    
    def mark_success(self):
        self.status = 'success'
    
    def mark_failure(self, error_msg):
        self.status = 'failed'
        self.error = error_msg


@ray.remote(num_cpus=1)
def run_relax_task(task: ManifoldTask):
    """Execute FastRelax"""
    # Ensure worker process has the path
    import sys
    if PEPMIMIC_PATH not in sys.path:
        sys.path.insert(0, PEPMIMIC_PATH)
    
    from evaluation.dG.energy import pyrosetta_fastrelax
    
    if task.status == 'failed':
        return task
    
    # If relaxed PDB exists and is valid, skip
    if os.path.exists(task.relaxed_pdb) and os.path.getsize(task.relaxed_pdb) > 0:
        return task
    
    try:
        # Note: pyrosetta_fastrelax(pdb_path, output_path, ligand_chain)
        # It expects the ligand chain ID to protect it/focus on it
        pyrosetta_fastrelax(task.pdb_file, task.relaxed_pdb, task.lig_chain)
        return task
    except Exception as e:
        task.mark_failure(f"FastRelax: {type(e).__name__}: {str(e)[:100]}")
        return task


@ray.remote(num_cpus=1)
def run_dg_task(task: ManifoldTask):
    """Calculate Binding Energy"""
    import sys
    if PEPMIMIC_PATH not in sys.path:
        sys.path.insert(0, PEPMIMIC_PATH)
    
    from evaluation.dG.energy import pyrosetta_interface_energy
    
    if task.status == 'failed':
        return task
    
    if not os.path.exists(task.relaxed_pdb):
        task.mark_failure("Relaxed PDB not found")
        return task
    
    try:
        found_chains = set()
        with open(task.relaxed_pdb, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    found_chains.add(line[21])
        valid_rec_chains = [c for c in task.rec_chains if c in found_chains]
        
        # pyrosetta_interface_energy(pdb_path, receptor_chains, ligand_chains, return_dict=False)
        # Returns float dG
        dG = pyrosetta_interface_energy(
            task.relaxed_pdb, 
            valid_rec_chains, 
            [task.lig_chain], 
            return_dict=False
        )
        task.dG = dG
        task.mark_success()
        return task
    except Exception as e:
        task.mark_failure(f"dG calculation: {type(e).__name__}: {str(e)[:100]}")
        return task


def read_chain_info(mapping_file):
    """Read ligand and receptor chain information from mapping file"""
    ligand_chains = {}
    receptor_chains = {}
    
    print(f"Reading chain info from: {mapping_file}")
    if not os.path.exists(mapping_file):
        print(f"Error: Mapping file not found at {mapping_file}!")
        return {}, {}

    with open(mapping_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                pdb_id = parts[0]
                rec_chain_str = parts[1]
                lig_chain = parts[2]
                
                ligand_chains[pdb_id] = lig_chain
                # Rec chains allow comma separated
                receptor_chains[pdb_id] = [c.strip() for c in rec_chain_str.split(',')]
    
    return ligand_chains, receptor_chains


def scan_manifold_tasks(data_dir, mapping_file):
    """Scan generated PDBs and create tasks"""
    tasks = []
    
    ligand_chains, receptor_chains_map = read_chain_info(mapping_file)
    
    if not os.path.exists(data_dir):
        print(f"Error: Generated PDBs directory not found: {data_dir}")
        return []

    # Iterate over PDB ID directories
    subdirs = sorted([d for d in os.listdir(data_dir) 
                     if os.path.isdir(os.path.join(data_dir, d))])
    
    count = 0
    for entry in subdirs:
        subdir_path = os.path.join(data_dir, entry)
        
        # Match PDB ID (handle suffix variations like 6E3Y or 6E3Y_P etc)
        pdb_id = entry
        if pdb_id not in ligand_chains:
            parts = entry.split('_')
            base_id = parts[0]
            if base_id in ligand_chains:
                pdb_id = base_id
            else:
                continue
            
        lig_chain = ligand_chains[pdb_id]
        rec_chains = receptor_chains_map[pdb_id]
        
        # Look for sample_*.pdb or entry_*.pdb
        files = os.listdir(subdir_path)
        # Filter generated pdbs, exclude rosetta ones and gt
        pdb_files = [f for f in files if (f.startswith("sample_") or f.startswith(entry)) and f.endswith(".pdb") and "_rosetta" not in f and "gt.pdb" not in f and "ref.pdb" not in f]
        
        # Manifold usually has 0-39
        
        for f in pdb_files:
            model_name_short = os.path.splitext(f)[0] # sample_0
            original_pdb = os.path.join(subdir_path, f)
            relaxed_pdb = os.path.join(subdir_path, f"{model_name_short}_rosetta.pdb")
            
            # Unique model name: 1bjr_sample_0
            full_model_name = f"{entry}_{model_name_short}"
            
            task = ManifoldTask(
                pdb_id=pdb_id,
                model_name=full_model_name,
                pdb_file=original_pdb,
                relaxed_pdb=relaxed_pdb,
                lig_chain=lig_chain,
                rec_chains=rec_chains
            )
            tasks.append(task)
            
    return tasks


def main():
    parser = argparse.ArgumentParser(description='ManifoldSurface Rosetta Energy (PepMimic Style)')
    parser.add_argument('--data_dir', type=str, required=True, help="Path to directory containing generated PDBs")
    parser.add_argument('--mapping_file', type=str, required=True, help="Path to chain mapping file")
    parser.add_argument('--output_file', type=str, required=True, help="Path to output summary file")
    parser.add_argument('--n_cpus', type=int, default=-1, help='Number of CPUs')
    parser.add_argument('--fastrelax', type=str, default='true', choices=['true', 'false'], help="Perform FastRelax before calculating dG (true or false)")
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"ManifoldSurface Rosetta Energy Calculation")
    print("=" * 60)
    
    # Init Ray
    if args.n_cpus > 0:
        ray.init(num_cpus=args.n_cpus, ignore_reinit_error=True)
    else:
        ray.init(ignore_reinit_error=True)
        
    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    # 1. Scan tasks
    print("Scanning tasks...")
    tasks = scan_manifold_tasks(args.data_dir, args.mapping_file)
    print(f"Found {len(tasks)} tasks.")
    
    if not tasks:
        print("No tasks found. Exiting.")
        return

    # 2. Submit Relax Tasks
    if args.fastrelax.lower() == 'true':
        print("\nStep 1: Submitting FastRelax tasks...")
        # Map future -> original task object
        relax_futures = {run_relax_task.remote(t): t for t in tasks}
        
        relaxed_tasks = []
        total = len(tasks)
        completed = 0
        start_time = time.time()
        
        while relax_futures:
            # Wait for at least one task to complete
            done_ids, _ = ray.wait(list(relax_futures.keys()), num_returns=1)
            for done_id in done_ids:
                original_task = relax_futures.pop(done_id)
                try:
                    task = ray.get(done_id)
                    if task.status != 'failed':
                        relaxed_tasks.append(task)
                    completed += 1
                    if completed % 100 == 0:
                        print(f"Relaxed: {completed}/{total}")
                except Exception as e:
                    completed += 1
                    print(f"Relax failed for {original_task.model_name}: {e}")
    else:
        print("\nStep 1: Skipping FastRelax tasks (--fastrelax=false)...")
        # Direct fallback to unrelaxed PDBs
        start_time = time.time()
        relaxed_tasks = []
        for t in tasks:
            t.relaxed_pdb = t.pdb_file
            relaxed_tasks.append(t)

    # 3. Submit dG Calculation Tasks
    print(f"\nStep 2: Submitting dG calculation tasks ({len(relaxed_tasks)} structures)...")
    
    dg_futures = {run_dg_task.remote(t): t for t in relaxed_tasks}
    completed_tasks = []
    completed = 0
    
    while dg_futures:
        done_ids, _ = ray.wait(list(dg_futures.keys()), num_returns=1)
        for done_id in done_ids:
            original_task = dg_futures.pop(done_id)
            try:
                task = ray.get(done_id)
                completed_tasks.append(task)
                completed += 1
                
                if completed % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f"Calculated: {completed}/{len(relaxed_tasks)} (Time: {elapsed:.1f}s)")
            except Exception as e:
                print(f"dG failed for {original_task.model_name}: {e}")

    # 4. Write output
    output_summary = args.output_file
    
    print("\nWriting results...")
    
    # Filter success
    success_tasks = [t for t in completed_tasks if t.status == 'success']
    
    # Sort
    def sort_key(t):
        try:
            parts = t.model_name.rsplit('_', 1) # pdbid_sample_num -> [pdbid_sample, num]
            return (t.pdb_id, int(parts[1]))
        except:
            return (t.pdb_id, t.model_name)
    success_tasks.sort(key=sort_key)
    
    with open(output_summary, 'w') as f:
        f.write("Complex\tModel\tRosetta_dG\n")
        for t in success_tasks:
            f.write(f"{t.pdb_id}\t{t.model_name}\t{t.dG:.4f}\n")
    
    print(f"Finished. Results saved to {output_summary}")
    ray.shutdown()

if __name__ == "__main__":
    main()

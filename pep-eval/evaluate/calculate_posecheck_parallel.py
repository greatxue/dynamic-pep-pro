import os
import sys
import argparse
import signal
import tempfile
import multiprocessing
import warnings
import uuid
import glob
import shutil
from pathlib import Path

# --- CRITICAL CONFIGURATION BEFORE OTHER IMPORTS ---
# Set these before numpy/scipy/torch are imported!
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# Ensure subprocesses can find executables in the current environment
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]

import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from Bio.PDB import PDBParser, PDBIO, Select
from rdkit import Chem
from rdkit import RDLogger

# --- Standard Import ---
try:
    from posecheck import PoseCheck
except ImportError:
    print("Error: PoseCheck module not found. Please ensure it is installed.")
    sys.exit(1)

# Configuration
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

TIMEOUT_SECONDS = 180

class TargetChainSelect(Select):
    """Selects everything EXCEPT the specified chains (for Receptor)"""
    def __init__(self, exclude_chains):
        self.exclude_chains = exclude_chains
    def accept_chain(self, chain):
        return 0 if chain.get_id() in self.exclude_chains else 1

class PeptideChainSelect(Select):
    """Selects ONLY the specified chains (for Peptide)"""
    def __init__(self, include_chains):
        self.include_chains = include_chains
    def accept_chain(self, chain):
        return 1 if chain.get_id() in self.include_chains else 0

# --- FIX LOGGING ---
def check_hydride_installation():
    """Verify hydride is working and can process a dummy file fast"""
    import shutil
    if not shutil.which("hydride"):
        print("[ERROR] hydride not found in PATH!")
        return False
    return True

check_hydride_installation()

# --- END FIX ---

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Calculation timed out")

def get_peptide_atoms(mol, pdb_file):
    """Adds PDB metadata to RDKit mol"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('ligand', pdb_file)
    
    pdb_atoms = []
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            for residue in chain:
                res_name = residue.resname
                res_num = residue.id[1]
                for atom in residue:
                    pdb_atoms.append({
                        'chain': chain_id,
                        'resname': res_name,
                        'resnum': res_num,
                        'name': atom.name
                    })
    
    if mol.GetNumAtoms() <= len(pdb_atoms):
        for i, atom in enumerate(mol.GetAtoms()):
            pdb_info = pdb_atoms[i]
            mi = Chem.AtomPDBResidueInfo()
            mi.SetChainId(pdb_info['chain'])
            mi.SetResidueName(pdb_info['resname'])
            mi.SetResidueNumber(pdb_info['resnum'])
            mi.SetName(pdb_info['name'])
            atom.SetMonomerInfo(mi)
    return mol

def calculate_internal_clashes(mol, tolerance=0.5):
    """
    Calculate Intra-molecular Clashes
    """
    try:
        conf = mol.GetConformer()
        pos = conf.GetPositions()
        num_atoms = mol.GetNumAtoms()
        
        topo_dist_matrix = Chem.GetDistanceMatrix(mol)
        pt = Chem.GetPeriodicTable()
        vdw_radii = np.array([pt.GetRvdw(mol.GetAtomWithIdx(i).GetAtomicNum()) for i in range(num_atoms)])
        
        dists = np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=-1)
        vdw_sum = vdw_radii[:, np.newaxis] + vdw_radii[np.newaxis, :]
        
        # Condition: Dist < VdW_sum and Topo > 3 (not bonded)
        clash_mask = (dists + tolerance < vdw_sum) & (topo_dist_matrix > 3)
        return np.count_nonzero(np.triu(clash_mask, k=1))
    except Exception:
        return float('nan')

# Global cache for worker processes
_WORKER_PC_CACHE = {}

def evaluate_single_conformation(complex_pdb, receptor_pdb, ligand_chain):
    """
    Evaluates a single complex PDB. 
    Splits it into Receptor and Ligand, then runs PoseCheck.
    """
    global _WORKER_PC_CACHE
    
    results = {
        'file': os.path.basename(complex_pdb),
        'clashes_inter': None,
        'clashes_inter_per_1000': None,
        'clashes_intra': None,
        'clashes_intra_per_1000': None,
        'num_atoms': 0,
        'success': False,
        'error': None
    }
    
    # Setup timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)
    
    rec_tmp_path = None
    lig_tmp_path = None
    
    # Suppress C++ warnings (DISABLE FOR DEBUGGING)
    # devnull = open(os.devnull, 'w')
    # old_stderr = os.dup(2)
    # sys.stderr.flush()
    # os.dup2(devnull.fileno(), 2)

    try:
        parser = PDBParser(QUIET=True)
        io = PDBIO()
        
        # 1. Prepare Ligand PDB (Extract ligand chain from complex/sample)
        # Note: 'complex_pdb' here is likely the generated full structure or just peptide
        # If peptide generation model outputs full complex, we extract peptide.
        # If it outputs just peptide (and we have separate receptor), we use that.
        
        # Check if complex_pdb has the ligand chain
        s_lig = parser.get_structure('lig', complex_pdb)
        lig_chains = [c.id for c in s_lig.get_chains()]
        
        if ligand_chain not in lig_chains:
             # Try assuming complex_pdb IS the ligand (if only 1 chain)
             if len(lig_chains) == 1:
                 ligand_chain = lig_chains[0]
             else:
                 raise ValueError(f"Ligand chain {ligand_chain} not found in {complex_pdb}")

        # Save Ligand to Temp
        fd, lig_tmp_path = tempfile.mkstemp(suffix=f'_lig_{ligand_chain}.pdb')
        os.close(fd)
        io.set_structure(s_lig)
        io.save(lig_tmp_path, PeptideChainSelect([ligand_chain]))
        
        # 3. Load into RDKit (Do this early to check validity)
        mol = Chem.MolFromPDBFile(lig_tmp_path, removeHs=False, sanitize=False)
        if not mol: raise ValueError("RDKit failed to load ligand PDB")
        
        mol = get_peptide_atoms(mol, lig_tmp_path)
        results['num_atoms'] = mol.GetNumAtoms()
        
        # 2. Prepare/Load Receptor (With Caching)
        pc = None
        
        # Only cache if we have a fixed reference receptor file
        use_cache = (receptor_pdb is not None and os.path.exists(receptor_pdb))
        cache_key = (receptor_pdb, ligand_chain)
        
        if use_cache and cache_key in _WORKER_PC_CACHE:
            pc = _WORKER_PC_CACHE[cache_key]
        else:
            # Need to load/process receptor
            fd, rec_tmp_path = tempfile.mkstemp(suffix='_rec.pdb')
            os.close(fd)
            
            if use_cache:
                # Clean receptor (remove optional ligand chain if present in ref)
                s_rec = parser.get_structure('rec', receptor_pdb)
                io.set_structure(s_rec)
                io.save(rec_tmp_path, TargetChainSelect([ligand_chain]))
            else:
                # Extract from complex (Dynamic receptor, cannot cache)
                io.set_structure(s_lig) # reused structure object
                io.save(rec_tmp_path, TargetChainSelect([ligand_chain]))
            
            # CHECK RECEPTOR SIZE
            if os.path.getsize(rec_tmp_path) < 100:
                msg = f"[WARN] Receptor file too small ({os.path.getsize(rec_tmp_path)} bytes): {rec_tmp_path}"
                print(msg)
                raise ValueError(msg) # FAIL FAST - don't hang hydride

            # Initialize PoseCheck and load protein (Expensive step: runs hydride)
            # print(f"[DEBUG] Starting load_protein_from_pdb for {rec_tmp_path}...")
            new_pc = PoseCheck()
            new_pc.load_protein_from_pdb(rec_tmp_path)
            # print(f"[DEBUG] Finished load_protein_from_pdb")
            
            if use_cache:
                _WORKER_PC_CACHE[cache_key] = new_pc
            
            pc = new_pc

        # 4. Calculate Metrics
        # Load ligands into the (possibly cached) PoseCheck object
        pc.load_ligands_from_mols([mol], add_hs=False)
        
        clashes = pc.calculate_clashes()
        results['clashes_inter'] = clashes[0] if clashes else 0
        
        intra = calculate_internal_clashes(mol)
        results['clashes_intra'] = intra
        if results['num_atoms'] > 0:
            results['clashes_intra_per_1000'] = (intra / results['num_atoms']) * 1000
            results['clashes_inter_per_1000'] = (results['clashes_inter'] / results['num_atoms']) * 1000

    except TimeoutException:
        print(f"[TIMEOUT] Processing {os.path.basename(complex_pdb)}")
        results['error'] = 'Timeout'
    except Exception as e:
        print(f"[ERROR] {os.path.basename(complex_pdb)}: {e}")
        results['error'] = str(e)
    finally:
        signal.alarm(0)
        # os.dup2(old_stderr, 2)
        # devnull.close()
        
        # Cleanup
        for p in [rec_tmp_path, lig_tmp_path]:
            if p and os.path.exists(p):
                try: os.remove(p)
                except: pass
                
    return results

def main():
    parser = argparse.ArgumentParser(description="Parallel PoseCheck Evaluation")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing generated PDBs (organized by target)")
    parser.add_argument("--reference_dir", type=str, required=True, help="Directory containing reference structure (native.pdb or structure.pdb)")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--mapping_file", type=str, required=True, help="File with 'target rec_chain pep_chain'")
    args = parser.parse_args()

    # Load Mapping (target -> (rec_chain, pep_chain))
    mapping = {}
    with open(args.mapping_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                mapping[parts[0]] = (parts[1], parts[2])

    # Find Targets
    targets = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    
    tasks = []
    
    print(f"Scanning {len(targets)} targets in {args.data_dir}...")
    
    for target in targets:
        if target not in mapping:
            continue
            
        rec_chain, pep_chain = mapping[target]
        target_dir = os.path.join(args.data_dir, target)
        
        # Find Reference PDB for Receptor (prioritize structure.pdb, then native.pdb)
        # Assuming reference_dir has subfolders or flat files matching target
        # Adjust logic based on dataset structure
        ref_pdb = None
        
        # 1. Check inside data_dir/target/ref.pdb (if manually placed)
        local_ref = os.path.join(target_dir, "ref.pdb")
        
        # 2. Check reference_dir/target.pdb
        global_ref = os.path.join(args.reference_dir, f"{target}.pdb")
        
        # 3. Check reference_dir/../pdbs/target.pdb (Try to fix path if reference_dir is wrong)
        # reference_dir was likely pointing to non-existent 'test_pdbs'
        # Try finding the pdb in parent/pdbs/...
        
        # Let's just fix the logic: If reference_dir is /path/to/test_pdbs but it doesn't exist, we fallback
        if os.path.exists(local_ref):
            ref_pdb = local_ref
        elif os.path.exists(global_ref):
            ref_pdb = global_ref
        else:
            # Maybe the user meant reference_dir/../pdbs/target.pdb?
            # Or assume reference_dir points to parent dir of pdbs
            pass
        
        # Scan for generated samples
        
        # Scan for generated samples
        # Supports pattern: sample_*.pdb or *.pdb (excluding ref, native, and rosetta)
        pdbs = glob.glob(os.path.join(target_dir, "*.pdb"))
        samples = [
            p for p in pdbs 
            if "ref" not in os.path.basename(p) 
            and "native" not in os.path.basename(p)
            and "rosetta" not in os.path.basename(p)
        ]
        
        for sample in samples:
            tasks.append({
                'complex_pdb': sample,
                'receptor_pdb': ref_pdb,
                'ligand_chain': pep_chain,
                'target': target,
                'model': os.path.basename(sample)
            })

    print(f"Found {len(tasks)} samples to evaluate.")
    
    # Parallel Execution
    results = []
    # Use a safe number of workers to avoid memory/subprocess issues
    n_workers = max(1, min(16, multiprocessing.cpu_count() // 2))
    print(f"Using {n_workers} workers.")

    # Initialize Output File with Headers
    output_columns = ['target', 'model_file', 'file', 'clashes_inter', 'clashes_inter_per_1000', 'clashes_intra', 'clashes_intra_per_1000', 'num_atoms', 'success', 'error']
    pd.DataFrame(columns=output_columns).to_csv(args.output_file, index=False)
    
    # DEBUG: Check first task
    if tasks:
        print(f"[DEBUG] First task config:")
        print(f"  Target: {tasks[0]['target']}")
        print(f"  Ref PDB: {tasks[0]['receptor_pdb']}")
        print(f"  Lig Chain: {tasks[0]['ligand_chain']}")
        if tasks[0]['receptor_pdb'] is None or not os.path.exists(tasks[0]['receptor_pdb']):
             print("  [WARNING] Ref PDB is missing! This will cause timeouts.")

    # Use 'spawn' to reset memory state for workers (avoids numpy/BLAS issues)
    # Note: 'spawn' is safer but slower startup. Given long task times, it's worth it.
    mp_ctx = multiprocessing.get_context('spawn')
    
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as executor:
        futures = {executor.submit(evaluate_single_conformation, t['complex_pdb'], t['receptor_pdb'], t['ligand_chain']): t for t in tasks}
        
        with tqdm(total=len(tasks), desc="Evaluating Clashes") as pbar:
            for future in as_completed(futures):
                task_info = futures[future]
                try:
                    res = future.result()
                    res['target'] = task_info['target']
                    res['model_file'] = task_info['model']
                    
                    # Append result to CSV immediately
                    pd.DataFrame([res], columns=output_columns).to_csv(args.output_file, mode='a', header=False, index=False)
                    
                    results.append(res)
                except Exception as e:
                    print(f"Worker Error: {e}")
                    
                pbar.update(1)

    # Final Summary (File is already saved incrementally)
    df = pd.DataFrame(results)
    
    print("\nEvaluation Complete.")
    print(f"Success: {df['success'].sum()} / {len(df)}")
    
    if df['success'].sum() > 0:
        print("\nSummary Metrics:")
        success_df = df[df['success'] == True]
        print(f"Avg Inter-molecular Clashes: {success_df['clashes_inter'].mean():.2f}")
        print(f"Avg Inter/1k Atoms:          {success_df['clashes_inter_per_1000'].mean():.2f}")
        print(f"Avg Intra-molecular Clashes: {success_df['clashes_intra'].mean():.2f}")
        print(f"Avg Intra/1k Atoms:          {success_df['clashes_intra_per_1000'].mean():.2f}")

if __name__ == "__main__":
    main()

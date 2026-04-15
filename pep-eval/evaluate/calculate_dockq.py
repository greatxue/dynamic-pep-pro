import os
import sys
import argparse
import pandas as pd
import concurrent.futures
from tqdm import tqdm

# Ensure that subprocess calls to 'python3' usage the current environment's python
# This is critical because PepMimic's dockq function calls 'python3' via subprocess
current_python_dir = os.path.dirname(sys.executable)
os.environ["PATH"] = current_python_dir + os.pathsep + os.environ.get("PATH", "")

# Add PepMimic path to import evaluation.dockq
sys.path.insert(0, '/home/shiyiming/peptide_design/PepMimic')

try:
    from evaluation.dockq import dockq
except ImportError as e:
    print(f"Error importing dockq from PepMimic: {e}")
    sys.exit(1)

def process_single_file(args):
    """
    Worker function for parallel processing.
    args: (gen_path, orig_pdb, pep_chain, rec_chain, entry_name, model_name)
    """
    gen_path, orig_pdb, pep_chain, rec_chain, entry_name, model_name = args
    if not os.path.exists(gen_path) or not os.path.exists(orig_pdb):
        return None
    try:
        score = dockq(gen_path, orig_pdb, pep_chain, rec_chain)
        if score is not None:
            return (entry_name, model_name, score)
    except Exception:
        pass
    return None

def load_chain_mapping(mapping_path):
    mapping = {}
    if os.path.exists(mapping_path):
        print(f"Loading chain mapping from {mapping_path}")
        with open(mapping_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    # e.g. 1SSC A B -> pdb=1SSC, rec=A, pep=B
                    pdb_id, rec_chain, pep_chain = parts[0], parts[1], parts[2]
                    mapping[pdb_id] = (rec_chain, pep_chain)
    return mapping

def main():
    parser = argparse.ArgumentParser(description="Calculate DockQ parallelized.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to generated_pdbs")
    parser.add_argument("--mapping_file", type=str, required=True, help="Path to LNR/test.txt")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output file")
    parser.add_argument("--dockq_script", type=str, help="Ignored")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Number of parallel workers")

    args = parser.parse_args()
    
    args.data_dir = os.path.abspath(args.data_dir)
    args.mapping_file = os.path.abspath(args.mapping_file)
    args.output_file = os.path.abspath(args.output_file)

    chain_mapping = load_chain_mapping(args.mapping_file)
    print(f"Loaded {len(chain_mapping)} chain mappings.")
    
    out_dir = os.path.dirname(args.output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    entries = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    
    tasks = []
    
    # Determine Native Directory based on mapping file location
    native_dir = os.path.join(os.path.dirname(args.mapping_file), "pdbs")
    print(f"Looking for native PDBs in: {native_dir}")
    
    for entry in entries:
        entry_dir = os.path.join(args.data_dir, entry)
        
        # Determine chains
        if entry in chain_mapping:
            rec_chain, pep_chain = chain_mapping[entry]
        else:
            parts = entry.split('_')
            if len(parts) >= 3:
                 rec_chain = parts[1]
                 pep_chain = parts[2]
            else: 
                 continue

        # Determine Native PDB
        # Priority 1: ref.pdb in generated dir (User Request: Use local reference frame)
        orig_pdb = os.path.join(entry_dir, "ref.pdb")
        
        if not os.path.exists(orig_pdb):
            # Priority 2: Native PDB in data/raw/LNR/pdbs/
            orig_pdb = os.path.join(native_dir, f"{entry}.pdb")
            
            if not os.path.exists(orig_pdb):
                 candidates = [f for f in os.listdir(entry_dir) if f.endswith('.pdb') and not f.startswith('sample_')]
                 if candidates:
                    orig_pdb = os.path.join(entry_dir, candidates[0])
                 else:
                    continue
        
        # Find generated files
        try:
            gen_files = [f for f in os.listdir(entry_dir) if f.startswith('sample_') and f.endswith('.pdb') and 'rosetta' not in f]
        except Exception:
            continue
            
        for gen_file in gen_files:
            gen_path = os.path.join(entry_dir, gen_file)
            tasks.append((gen_path, orig_pdb, pep_chain, rec_chain, entry, gen_file))

    print(f"Collected {len(tasks)} DockQ tasks. Executing with {args.workers} workers...")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_file, task): task for task in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Processing DockQ"):
            try:
                res = future.result()
                if res:
                    results.append(res)
            except Exception:
                pass
                
    # Write sorted results
    results.sort(key=lambda x: (x[0], x[1]))
    
    print(f"Writing {len(results)} results to {args.output_file}")
    with open(args.output_file, "w") as f_out:
        f_out.write("Complex\tModel\tDockQ\n")
        for ent, mod, score in results:
             f_out.write(f"{ent}\t{mod}\t{score:.4f}\n")
             
    print(f"Finished.")

if __name__ == "__main__":
    main()

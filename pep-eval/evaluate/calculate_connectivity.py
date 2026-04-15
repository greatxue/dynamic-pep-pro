import os
import sys
import argparse
import numpy as np
from Bio.PDB import PDBParser
import warnings

warnings.filterwarnings('ignore')

def load_chain_mapping(mapping_path):
    mapping = {}
    if os.path.exists(mapping_path):
        print(f"Loading chain mapping from {mapping_path}")
        with open(mapping_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    pdb_id, rec_chain, pep_chain = parts[0], parts[1], parts[2]
                    mapping[pdb_id] = (rec_chain, pep_chain)
    return mapping

def calculate_connectivity(pdb_path, pep_chain):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_path)
    except Exception:
        return None
        
    model = structure[0]
    if pep_chain not in model:
        return None
        
    chain = model[pep_chain]
    
    # Get all residues that are standard amino acids and contain at least CA
    residues = [res for res in chain if res.id[0] == ' ' and 'CA' in res]
    if len(residues) <= 1:
        return None  # Cannot compute connectivity for 0 or 1 residue
        
    total_expected_bonds = (len(residues) - 1) * 3  # (C-N), (N-CA), (CA-C) for each pair+residue
    connected_bonds = 0
    
    for i in range(len(residues)):
        res_i = residues[i]
        
        # 1. Check intra-residue N - CA bond (Ideal ~ 1.46)
        if 'N' in res_i and 'CA' in res_i:
            dist_n_ca = np.linalg.norm(res_i['N'].coord - res_i['CA'].coord)
            if dist_n_ca <= 1.46 + 0.25:
                connected_bonds += 1
                
        # 2. Check intra-residue CA - C bond (Ideal ~ 1.52)
        if 'CA' in res_i and 'C' in res_i:
            dist_ca_c = np.linalg.norm(res_i['CA'].coord - res_i['C'].coord)
            if dist_ca_c <= 1.52 + 0.25:
                connected_bonds += 1
                
        # 3. Check inter-residue C - N bond (Ideal ~ 1.33)
        if i < len(residues) - 1:
            res_next = residues[i+1]
            if 'C' in res_i and 'N' in res_next:
                dist_c_n = np.linalg.norm(res_i['C'].coord - res_next['N'].coord)
                if dist_c_n <= 1.33 + 0.25:
                    connected_bonds += 1
                
    # Modify total_expected_bonds logic since N-CA and CA-C are per residue, and C-N is per pair
    # Total internal bonds: len(residues) * 2
    # Total peptide bonds: len(residues) - 1
    total_expected_bonds = (len(residues) * 2) + (len(residues) - 1)
    
    return float(connected_bonds) / total_expected_bonds

def main():
    parser = argparse.ArgumentParser(description="Calculate peptide connectivity.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to directory containing generated PDBs")
    parser.add_argument("--mapping_file", type=str, default="/home/shiyiming/peptide_design/ManifoldSurface_Peptide_Generator/data/raw/LNR/test.txt",
                        help="Path to chain mapping file")
    parser.add_argument("--output_file", type=str, default="evaluate/all_eval_results/connectivity_results.txt",
                        help="Path to output summary file")
    
    args = parser.parse_args()

    
    chain_mapping = load_chain_mapping(args.mapping_file)
    print(f"Loaded {len(chain_mapping)} chain mappings.")
    
    out_dir = os.path.dirname(args.output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Use a set to track already processed complexes to skip existing results?
    # For now we'll overwrite or just recompute since it's fast.
    # To support append and skip, we can read existing.
    processed_set = set()
    mode = "w"
    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split()
                if len(parts) >= 2:
                    processed_set.add((parts[0], parts[1]))
        mode = "a"

    entries = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    
    processed_count = 0
    skipped_count = 0
    
    with open(args.output_file, mode) as f_out:
        if mode == "w":
            f_out.write("Complex\tModel\tConnectivity\n")
            f_out.flush()
            
        for entry in entries:
            entry_dir = os.path.join(args.data_dir, entry)
            
            if entry in chain_mapping:
                _, pep_chain = chain_mapping[entry]
            else:
                parts = entry.split('_')
                if len(parts) >= 3:
                     pep_chain = parts[2]
                elif len(parts) == 2:
                     pep_chain = parts[1]
                else: 
                     if entry not in chain_mapping:
                         continue
                         
            gen_files = sorted([f for f in os.listdir(entry_dir) if (f.startswith('sample_') and f.endswith('.pdb')) or (f.startswith(entry) and f.endswith('.pdb') and not f.endswith('gt.pdb'))])
            
            for gen_file in gen_files:
                complex_name = entry
                model_name = gen_file
                if (complex_name, model_name) in processed_set:
                    skipped_count += 1
                    continue
                    
                gen_path = os.path.join(entry_dir, gen_file)
                conn = calculate_connectivity(gen_path, pep_chain)
                
                if conn is not None:
                    f_out.write(f"{complex_name}\t{model_name}\t{conn:.4f}\n")
                    processed_count += 1
                    f_out.flush()
                    
    print(f"Finished. Processed {processed_count} files. Skipped {skipped_count}.")

if __name__ == '__main__':
    main()

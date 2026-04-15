"""
Calculate BSR (Binding Site Recovery) for generated peptides.

BSR measures how well the generated peptide contacts the same receptor residues as the native peptide.
BSR = |(Ref Binding Site) ∩ (Gen Binding Site)| / |Ref Binding Site|

Usage:
    python evaluate/calculate_bsr.py \
        --data_dir /home/shiyiming/peptide_design/ManifoldSurface_Peptide_Generator/generated_pdbs \
        --mapping_file /home/shiyiming/peptide_design/ManifoldSurface_Peptide_Generator/data/raw/LNR/test.txt \
        --output_file evaluate/bsr_results.txt
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import warnings
from Bio.PDB import PDBParser, Selection, NeighborSearch

# Suppress PDB warnings
warnings.filterwarnings("ignore")

def get_binding_site_residues(structure, rec_chain_id, lig_chain_id, dist_th=6.0):
    """
    Identify receptor residues within dist_th of the ligand.
    Returns: set of (chain_id, res_id) for receptor residues.
    """
    try:
        model = structure[0]
        
        # Extract atoms
        rec_atoms = []
        lig_atoms = []
        
        # Check chains exist
        if rec_chain_id not in model or lig_chain_id not in model:
            return set()

        rec_chain = model[rec_chain_id]
        lig_chain = model[lig_chain_id]

        for residue in rec_chain:
            # Skip HOH, etc.
            if residue.id[0] != ' ': continue
            for atom in residue:
                rec_atoms.append(atom)
                
        for residue in lig_chain:
            if residue.id[0] != ' ': continue
            for atom in residue:
                lig_atoms.append(atom)
                
        if not rec_atoms or not lig_atoms:
            return set()

        # NeighborSearch
        ns = NeighborSearch(rec_atoms)
        
        binding_site_residues = set()
        
        for atom in lig_atoms:
            # Find receptor atoms within distance
            neighbors = ns.search(atom.get_coord(), dist_th, level='R')
            for res in neighbors:
                binding_site_residues.add((res.get_parent().id, res.id))
                
        return binding_site_residues
        
    except Exception as e:
        print(f"Error calculating binding site: {e}")
        return set()

def calculate_bsr_score(ref_structure, gen_structure, rec_chain, lig_chain, dist_th=6.0):
    """
    Calculate BSR score.
    """
    # 1. Get Reference Binding Site (using Reference Structure)
    ref_bs = get_binding_site_residues(ref_structure, rec_chain, lig_chain, dist_th)
    
    if not ref_bs:
        return None  # No binding site in reference?
        
    # 2. Get Generated Binding Site (using Generated Structure)
    # Note: Generated structure contains BOTH Receptor and Generated Ligand.
    # We check which receptor residues the GENERATED ligand contacts.
    gen_bs = get_binding_site_residues(gen_structure, rec_chain, lig_chain, dist_th)
    
    # 3. Calculate intersection
    intersection = ref_bs.intersection(gen_bs)
    
    if len(ref_bs) == 0:
        return 0.0
        
    return len(intersection) / len(ref_bs)

def load_chain_mapping(mapping_path):
    """
    Load chain mapping from file.
    Format: PDB_ID REC_CHAIN PEP_CHAIN ...
    Returns: dict {pdb_id: (rec_chain, pep_chain)}
    """
    mapping = {}
    if os.path.exists(mapping_path):
        print(f"Loading chain mapping from {mapping_path}")
        with open(mapping_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    pdb_id, rec_chain, pep_chain = parts[0], parts[1], parts[2]
                    mapping[pdb_id] = (rec_chain, pep_chain)
    else:
        print(f"Warning: Chain mapping file not found at {mapping_path}")
    return mapping

def main():
    parser = argparse.ArgumentParser(description="Calculate BSR for generated peptides.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the directory containing generated PDBs")
    parser.add_argument("--mapping_file", type=str, default="/home/shiyiming/peptide_design/ManifoldSurface_Peptide_Generator/data/raw/LNR/test.txt",
                        help="Path to the chain mapping file")
    parser.add_argument("--output_file", type=str, default="evaluate/bsr_results.txt",
                        help="Path to the output summary file")
    parser.add_argument("--threshold", type=float, default=6.0, 
                        help="Distance threshold for contacting residues (Angstroms)")
    
    args = parser.parse_args()
    
    # Load mapping
    chain_mapping = load_chain_mapping(args.mapping_file)
    print(f"Loaded {len(chain_mapping)} chain mappings.")
    
    # Parse PDBs
    parser_pdb = PDBParser(QUIET=True)
    
    processed_count = 0
    
    # Initialize output directory if needed
    out_dir = os.path.dirname(args.output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    entries = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    
    with open(args.output_file, "w") as f_out:
        f_out.write("Complex\tModel\tBSR\n")
        
        for entry in entries:
            entry_dir = os.path.join(args.data_dir, entry)
            
            # Determine chains
            if entry in chain_mapping:
                rec_chain, pep_chain = chain_mapping[entry]
            else:
                print(f"Skipping {entry} (no chain mapping)")
                continue
                
            # Reference PDB
            ref_path = os.path.join(entry_dir, "ref.pdb")
            if not os.path.exists(ref_path):
                # Fallback: try to find PDB in the folder that isn't sample_X
                candidates = [f for f in os.listdir(entry_dir) if f.endswith('.pdb') and not f.startswith('sample_')]
                if candidates:
                    ref_path = os.path.join(entry_dir, candidates[0])
                else:
                    print(f"Reference not found for {entry}")
                    continue
            
            try:
                ref_structure = parser_pdb.get_structure('ref', ref_path)
            except Exception:
                print(f"Failed to parse ref {ref_path}")
                continue

            # Process generated files
            gen_files = sorted([f for f in os.listdir(entry_dir) if f.startswith('sample_') and f.endswith('.pdb')])
            
            for gen_file in gen_files:
                gen_path = os.path.join(entry_dir, gen_file)
                try:
                    gen_structure = parser_pdb.get_structure('gen', gen_path)
                    
                    bsr = calculate_bsr_score(ref_structure, gen_structure, rec_chain, pep_chain, dist_th=args.threshold)
                    
                    if bsr is not None:
                        f_out.write(f"{entry}\t{gen_file}\t{bsr:.4f}\n")
                        processed_count += 1
                        
                except Exception as e:
                    print(f"Error processing {gen_file}: {e}")
            
            f_out.flush()
            
    print(f"Finished. Results saved to {args.output_file}")
    print(f"Processed {processed_count} generated structures.")
    
    # Summary Statistics
    if processed_count > 0:
        try:
            df = pd.read_csv(args.output_file, sep='\t')
            # Filter valid rows
            if 'BSR' in df.columns:
                df['BSR'] = pd.to_numeric(df['BSR'], errors='coerce')
                df = df.dropna(subset=['BSR'])
                
                complex_means = df.groupby('Complex')['BSR'].mean()
                best_means = df.groupby('Complex')['BSR'].max()
                
                print("\n" + "="*40)
                print("BSR Summary Statistics")
                print("="*40)
                print(f"Mean BSR (avg of complex means): {complex_means.mean():.4f}")
                print(f"Mean Best BSR (avg of complex bests): {best_means.mean():.4f}")
                print(f"Total Complexes: {len(complex_means)}")
        except Exception as e:
            print(f"Could not calculate summary statistics: {e}")

if __name__ == "__main__":
    main()

"""
Calculate RMSD for generated peptides.
Self-contained script, does not depend on PepMimic package.

Usage:
    python evaluate/calculate_rmsd.py \
        --data_dir /home/shiyiming/peptide_design/ManifoldSurface_Peptide_Generator/generated_pdbs \
        --mapping_file /home/shiyiming/peptide_design/ManifoldSurface_Peptide_Generator/data/raw/LNR/test.txt \
        --output_file evaluate/rmsd_results.txt
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from math import sqrt
from Bio.PDB import PDBParser
from Bio.Align import substitution_matrices, PairwiseAligner

# =============================================================================
# Helper functions copied/adapted from PepMimic to avoid dependency
# =============================================================================

# From PepMimic/evaluation/rmsd.py
def compute_rmsd(a, b, aligned=False):  # amino acids level rmsd
    dist = np.sum((a - b) ** 2, axis=-1)
    rmsd = np.sqrt(dist.sum() / a.shape[0])
    return float(rmsd)

def kabsch_rotation(P, Q):
    """
    Using the Kabsch algorithm with two sets of paired point P and Q.
    """
    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)

    v, s, w = np.linalg.svd(C)
    d = (np.linalg.det(v) * np.linalg.det(w)) < 0.0

    if d:
        s[-1] = -s[-1]
        v[:, -1] = -v[:, -1]

    # Create Rotation matrix U
    U = np.dot(v, w)
    return U

def kabsch(a, b):
    # a, b are both [N, 3]
    a, b = np.array(a), np.array(b)
    a_mean = np.mean(a, axis=0)
    b_mean = np.mean(b, axis=0)
    a_c = a - a_mean
    b_c = b - b_mean

    rotation = kabsch_rotation(a_c, b_c)
    t = b_mean - np.dot(a_mean, rotation)
    a_aligned = np.dot(a, rotation) + t

    return a_aligned, rotation, t

# From PepMimic/evaluation/seq_metric.py
def align_sequences(sequence_A, sequence_B, **kwargs):
    """
    Performs a global pairwise alignment between two sequences.
    """
    sub_matrice = substitution_matrices.load('BLOSUM62')
    aligner = PairwiseAligner()
    aligner.substitution_matrix = sub_matrice
    if kwargs.get('local', False):
        aligner.mode = 'local'
    alns = aligner.align(sequence_A, sequence_B)

    best_aln = alns[0]
    aligned_A, aligned_B = best_aln

    base = sqrt(aligner.score(sequence_A, sequence_A) * aligner.score(sequence_B, sequence_B))
    if base == 0:
        seq_id = 0
    else:
        seq_id = aligner.score(sequence_A, sequence_B) / base
    return (aligned_A, aligned_B), seq_id

# Simple amino acid mapping
three_to_one = {
    'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'LEU': 'L', 'ILE': 'I',
    'PHE': 'F', 'TRP': 'W', 'TYR': 'Y', 'ASP': 'D', 'HIS': 'H',
    'ASN': 'N', 'GLU': 'E', 'LYS': 'K', 'GLN': 'Q', 'MET': 'M',
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'CYS': 'C', 'PRO': 'P'
}

def abrv_to_symbol(resname):
    return three_to_one.get(resname.upper(), '?')

# =============================================================================
# Main Logic
# =============================================================================

def get_ca_coords_and_seq(pdb_file, ligand_chain):
    """Get CA coordinates and sequence from PDB file for specific chain."""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('pdb', pdb_file)
        for model in structure:
            for chain in model:
                if chain.id == ligand_chain:
                    ca_coords = []
                    sequence = []
                    for res in chain:
                        if 'CA' in res:
                            resname = res.get_resname()
                            aa_symbol = abrv_to_symbol(resname)
                            # Only include standard amino acids
                            if aa_symbol != '?':
                                ca_coords.append(res['CA'].get_coord())
                                sequence.append(aa_symbol)
                    if len(ca_coords) > 0:
                        return np.array(ca_coords), ''.join(sequence)
    except Exception as e:
        print(f"Error parsing {pdb_file}: {e}")
    return None, None

def calc_aligned_rmsd(coords_gen, coords_ref):
    """Calculate RMSD with Kabsch alignment for same length."""
    if coords_gen.shape != coords_ref.shape:
        return None
    if len(coords_gen) == 0:
        return None
    coords_aligned, _, _ = kabsch(coords_gen, coords_ref)
    rmsd = compute_rmsd(coords_aligned, coords_ref, aligned=True)
    return rmsd

def calc_aligned_rmsd_with_seq_alignment(coords_gen, seq_gen, coords_ref, seq_ref):
    """Calculate RMSD with sequence alignment for different lengths."""
    if len(coords_gen) == 0 or len(coords_ref) == 0:
        return None
    
    (aligned_gen, aligned_ref), _ = align_sequences(seq_gen, seq_ref)
    
    matched_coords_gen = []
    matched_coords_ref = []
    
    i_gen, i_ref = 0, 0
    for aa_gen, aa_ref in zip(aligned_gen, aligned_ref):
        if aa_gen != '-' and aa_ref != '-':
            # Ensure we don't go out of bounds (misalignment safety)
            if i_gen < len(coords_gen) and i_ref < len(coords_ref):
                matched_coords_gen.append(coords_gen[i_gen])
                matched_coords_ref.append(coords_ref[i_ref])
        
        if aa_gen != '-':
            i_gen += 1
        if aa_ref != '-':
            i_ref += 1
    
    if len(matched_coords_gen) == 0:
        return None
    
    matched_coords_gen = np.array(matched_coords_gen)
    matched_coords_ref = np.array(matched_coords_ref)
    
    coords_aligned, _, _ = kabsch(matched_coords_gen, matched_coords_ref)
    rmsd = compute_rmsd(coords_aligned, matched_coords_ref, aligned=True)
    return rmsd

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
    else:
        print(f"Warning: Chain mapping file not found at {mapping_path}")
    return mapping

def main():
    parser = argparse.ArgumentParser(description="Calculate RMSD for generated peptides.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the directory containing generated PDBs")
    parser.add_argument("--mapping_file", type=str, default="/home/shiyiming/peptide_design/ManifoldSurface_Peptide_Generator/data/raw/LNR/test.txt",
                        help="Path to the chain mapping file")
    parser.add_argument("--output_file", type=str, default="evaluate/rmsd_results.txt",
                        help="Path to the output summary file")
    
    args = parser.parse_args()
    
    chain_mapping = load_chain_mapping(args.mapping_file)
    print(f"Loaded {len(chain_mapping)} chain mappings.")
    
    processed_count = 0
    
    out_dir = os.path.dirname(args.output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    entries = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    
    with open(args.output_file, "w") as f_out:
        f_out.write("Complex\tModel\tRMSD\n")
        
        for entry in entries:
            entry_dir = os.path.join(args.data_dir, entry)
            
            if entry in chain_mapping:
                _, pep_chain = chain_mapping[entry]
            else:
                # heuristic: if format is PDBID_REC_PEP, try to parse
                parts = entry.split('_')
                if len(parts) == 3:
                     pep_chain = parts[2]
                else: 
                    # Try to infer from mapping if key is just pdbid
                    # Here we assume entry matches key
                    # If not found, skip
                    if entry not in chain_mapping:
                         # print(f"Skipping {entry} (no chain mapping)")
                         continue
            
            # Use ref.pdb instead of {entry}.pdb
            orig_pdb = os.path.join(entry_dir, "ref.pdb")
            if not os.path.exists(orig_pdb):
                 # Fallback
                candidates = [f for f in os.listdir(entry_dir) if f.endswith('.pdb') and not f.startswith('sample_')]
                if candidates:
                    orig_pdb = os.path.join(entry_dir, candidates[0])
                else:
                    print(f"Reference not found for {entry}")
                    continue

            coords_ref, seq_ref = get_ca_coords_and_seq(orig_pdb, pep_chain)
            if coords_ref is None or len(coords_ref) == 0:
                print(f"Failed to extract ref chain {pep_chain} from {orig_pdb}")
                continue
            
            gen_files = sorted([f for f in os.listdir(entry_dir) if f.startswith('sample_') and f.endswith('.pdb')])
            
            for gen_file in gen_files:
                gen_path = os.path.join(entry_dir, gen_file)
                coords_gen, seq_gen = get_ca_coords_and_seq(gen_path, pep_chain)
                
                if coords_gen is None or len(coords_gen) == 0:
                    continue
                
                if len(coords_gen) == len(coords_ref):
                    rmsd = calc_aligned_rmsd(coords_gen, coords_ref)
                else:
                    rmsd = calc_aligned_rmsd_with_seq_alignment(coords_gen, seq_gen, coords_ref, seq_ref)
                
                if rmsd is not None:
                    f_out.write(f"{entry}\t{gen_file}\t{rmsd:.4f}\n")
                    processed_count += 1
            
            f_out.flush()
            
    print(f"Finished. Processed {processed_count} files.")
    
    # Statistics
    if processed_count > 0:
        try:
            df = pd.read_csv(args.output_file, sep='\t')
            if 'RMSD' in df.columns:
                df['RMSD'] = pd.to_numeric(df['RMSD'], errors='coerce')
                df = df.dropna(subset=['RMSD'])
                
                complex_means = df.groupby('Complex')['RMSD'].mean()
                best_means = df.groupby('Complex')['RMSD'].min() # RMSD is lower better
                
                print("\n" + "="*40)
                print("RMSD Summary Statistics")
                print("="*40)
                print(f"Mean RMSD (avg of complex means): {complex_means.mean():.4f}")
                print(f"Mean Best RMSD (avg of complex mins): {best_means.mean():.4f}")
                print(f"Total Complexes: {len(complex_means)}")
        except Exception as e:
            print(f"Could not calculate stats: {e}")

if __name__ == "__main__":
    main()

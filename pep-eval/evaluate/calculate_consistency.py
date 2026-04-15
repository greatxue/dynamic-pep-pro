import os
import sys
import argparse
import numpy as np
import pandas as pd
from math import sqrt
from Bio.PDB import PDBParser
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats.contingency import association
from Bio.Align import substitution_matrices, PairwiseAligner
from tqdm import tqdm

# Add PepMimic path
sys.path.insert(0, '/home/shiyiming/peptide_design/PepMimic')
try:
    from data.format import VOCAB
except ImportError:
    class VOCAB:
        @staticmethod
        def abrv_to_symbol(abrv):
            d = {'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLN':'Q', 'GLU':'E', 
                 'GLY':'G', 'HIS':'H', 'ILE':'I', 'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 
                 'PRO':'P', 'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'}
            return d.get(abrv.upper(), None)

def align_score_val(sequence_A, sequence_B):
    try:
        sub_matrix = substitution_matrices.load('BLOSUM62')
        aligner = PairwiseAligner()
        aligner.substitution_matrix = sub_matrix
        
        score = aligner.score(sequence_A, sequence_B)
        base_A = aligner.score(sequence_A, sequence_A)
        base_B = aligner.score(sequence_B, sequence_B)
        base = sqrt(base_A * base_B)
        
        return score / base if base > 0 else 0.0
    except Exception:
        return 0.0

def get_ca_coords_and_seq(pdb_file, lig_chain):
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('pdb', pdb_file)
        for model in structure:
            for chain in model:
                if chain.id == lig_chain:
                    ca_coords = []
                    sequence = []
                    for res in chain:
                        if 'CA' in res:
                            resname = res.get_resname()
                            aa_symbol = VOCAB.abrv_to_symbol(resname)
                            if aa_symbol: # Only valid AAs
                                ca_coords.append(res['CA'].get_coord())
                                sequence.append(aa_symbol)
                    if len(ca_coords) > 0:
                        return np.array(ca_coords), ''.join(sequence)
        return None, None
    except Exception:
        return None, None

def load_sequences_and_structs(entry_dir, pep_chain):
    gen_files = sorted([f for f in os.listdir(entry_dir) if f.startswith('sample_') and f.endswith('.pdb') and 'rosetta' not in f])
    seqs = []
    structs = []
    
    for f in gen_files:
        coords, seq = get_ca_coords_and_seq(os.path.join(entry_dir, f), pep_chain)
        if coords is not None and seq:
            seqs.append(seq)
            structs.append(coords)
            
    if not structs: 
        return [], np.array([])
    
    # Truncate to min length to allow numpy array creation for RMSD calculation
    # RMSD requires same number of atoms. For varying lengths, methods differ.
    # Simple approach: truncate to length of shortest peptide in the set.
    min_len = min(len(s) for s in structs)
    final_structs = np.array([s[:min_len] for s in structs])
    
    # Also truncate sequences for consistency? No, sequence alignment handles different lengths.
    
    return seqs, final_structs

def calculate_consistency(seqs, structs, seq_th=0.4, struct_th=4.0):
    """
    seqs: List of sequence strings
    structs: (N, L, 3) numpy array of CA coordinates
    seq_th: Distance threshold for sequence clustering (1 - similarity)
    struct_th: Distance threshold for structure clustering (RMSD in Angstroms)
    """
    if len(seqs) < 2: return 0.0
    n = len(seqs)
    
    # --- Sequence Clustering ---
    seq_dists = np.zeros((n, n))
    # Fill upper triangle
    # Optimized loop? For N=100, N^2 = 10000 comparisons. Fast enough.
    for i in range(n):
        for j in range(i+1, n):
            val = 1.0 - align_score_val(seqs[i], seqs[j])
            seq_dists[i, j] = val
            seq_dists[j, i] = val # Symmetric
            
    # Hierarchical Clustering
    try:
        # squareform checks for symmetry and 0 diagonal
        Z_seq = linkage(squareform(seq_dists), method='single')
        seq_clusters = fcluster(Z_seq, t=seq_th, criterion='distance')
    except Exception as e:
        print(f"Seq clustering error: {e}")
        return 0.0

    # --- Structure Clustering ---
    struct_dists = np.zeros((n, n))
    for i in range(n):
        # Broadcasting difference: (N, L, 3) - (L, 3) -> (N, L, 3)
        diff = structs - structs[i]
        # RMSD per structure: sqrt(mean(dX^2 + dY^2 + dZ^2))
        rms = np.sqrt(np.mean(np.sum(diff**2, axis=-1), axis=-1))
        struct_dists[i, :] = rms
        
    try:
        Z_struct = linkage(squareform(struct_dists), method='single')
        struct_clusters = fcluster(Z_struct, t=struct_th, criterion='distance')
    except Exception as e:
        print(f"Struct clustering error: {e}")
        return 0.0
        
    # --- Cramér's V ---
    # Create contingency table
    # Clusters are 1-based integers
    n_seq_c = np.max(seq_clusters)
    n_struct_c = np.max(struct_clusters)
    
    # If all in one cluster, perfect agreement if both are 1.
    if n_seq_c == 1 and n_struct_c == 1:
        return 1.0
    if n_seq_c == 1 or n_struct_c == 1:
        return 0.0 # No correlation possible if one variable is constant and other varies
        
    table = np.zeros((n_seq_c, n_struct_c), dtype=int)
    for s_c, st_c in zip(seq_clusters, struct_clusters):
        table[s_c - 1][st_c - 1] += 1
        
    try:
        score = association(table, method='cramer')
        return score
    except Exception:
        return 0.0

def load_chain_mapping(mapping_path):
    mapping = {}
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    # PDB_ID CHAIN_REC CHAIN_LIG
                    mapping[parts[0]] = (parts[1], parts[2])
    return mapping

def main():
    parser = argparse.ArgumentParser(description="Calculate Consistency (Cramer's V between Seq and Struct Clusters)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to directory containing PDB folders")
    parser.add_argument("--mapping_file", type=str, required=True, help="Path to chain mapping file (PDB REC LIG)")
    parser.add_argument("--output_file", type=str, default="evaluate/consistency_results.txt", help="Output file path")
    args = parser.parse_args()
    
    mapping = load_chain_mapping(args.mapping_file)
    entries = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    
    out_dir = os.path.dirname(args.output_file)
    if out_dir and not os.path.exists(out_dir): os.makedirs(out_dir)
    
    results = []
    
    # Only process entries present in mapping or parseable
    valid_entries = []
    for entry in entries:
        if entry in mapping:
            valid_entries.append(entry)
        elif len(entry.split('_')) >= 3:
            valid_entries.append(entry)
            
    print(f"Found {len(valid_entries)} valid entries.")

    for entry in tqdm(valid_entries, desc="Calculating Consistency"):
        if entry in mapping:
            _, pep_chain = mapping[entry]
        else:
             parts = entry.split('_')
             pep_chain = parts[2]
             
        entry_dir = os.path.join(args.data_dir, entry)
        
        try:
            seqs, structs = load_sequences_and_structs(entry_dir, pep_chain)
            if len(seqs) < 5: 
                # Need enough samples for meaningful clustering
                continue
                
            score = calculate_consistency(seqs, structs)
            results.append([entry, score])
        except Exception as e:
            print(f"Error processing {entry}: {e}")
            continue
        
    df = pd.DataFrame(results, columns=["Complex", "Consistency"])
    df.to_csv(args.output_file, sep='\t', index=False)
    
    if not df.empty:
        print(f"Saved consistency results to {args.output_file}")
        print(f"Mean Consistency: {df['Consistency'].mean():.4f}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()

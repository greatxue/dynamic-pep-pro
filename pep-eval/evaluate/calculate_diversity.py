import os
import sys
import argparse
import numpy as np
import pandas as pd
from math import sqrt
from Bio.PDB import PDBParser
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
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
                            ca_coords.append(res['CA'].get_coord())
                            resname = res.get_resname()
                            aa_symbol = VOCAB.abrv_to_symbol(resname)
                            if aa_symbol:
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
    
    # Truncate to min length
    min_len = min(len(s) for s in structs)
    final_structs = np.array([s[:min_len] for s in structs])
    return seqs, final_structs

def calculate_seq_diversity(seqs, th=0.4):
    if len(seqs) < 2: return 0.0
    n = len(seqs)
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            val = 1 - align_score_val(seqs[i], seqs[j])
            dists[i, j] = val
            dists[j, i] = val
            
    try:
        condensed = squareform(dists)
        Z = linkage(condensed, 'single')
        clusters = fcluster(Z, t=th, criterion='distance')
        return len(np.unique(clusters)) / n
    except Exception:
        return 0.0 

def calculate_struct_diversity(structs, th=4.0):
    if len(structs) < 2: return 0.0
    n = len(structs)
    
    dists = np.zeros((n, n))
    for i in range(n):
        # pair RMSD
        diff = structs - structs[i]
        rms = np.sqrt(np.mean(np.sum(diff**2, axis=-1), axis=-1))
        dists[i, :] = rms
        
    try:
        condensed = squareform(dists)
        Z = linkage(condensed, 'single')
        clusters = fcluster(Z, t=th, criterion='distance')
        return len(np.unique(clusters)) / n
    except Exception:
        return 0.0

def load_chain_mapping(mapping_path):
    mapping = {}
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    mapping[parts[0]] = (parts[1], parts[2])
    return mapping

def main():
    parser = argparse.ArgumentParser(description="Calculate Diversity (Seq, Struct, CoDesign)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--mapping_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="evaluate/diversity_results.txt")
    args = parser.parse_args()
    
    mapping = load_chain_mapping(args.mapping_file)
    entries = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    
    out_dir = os.path.dirname(args.output_file)
    if out_dir and not os.path.exists(out_dir): os.makedirs(out_dir)
    
    results = []
    
    for entry in tqdm(entries, desc="Calculating Diversity"):
        if entry in mapping:
            _, pep_chain = mapping[entry]
        else:
             p = entry.split('_')
             if len(p) >= 3: pep_chain = p[2]
             else: continue
             
        entry_dir = os.path.join(args.data_dir, entry)
        
        seqs, structs = load_sequences_and_structs(entry_dir, pep_chain)
        
        if len(seqs) < 2:
            continue
            
        seq_div = calculate_seq_diversity(seqs)
        struct_div = calculate_struct_diversity(structs)
        co_div = sqrt(seq_div * struct_div)
        
        results.append((entry, seq_div, struct_div, co_div))
        
    df = pd.DataFrame(results, columns=["Complex", "Seq_Div", "Struct_Div", "Co_Div"])
    df.to_csv(args.output_file, sep='\t', index=False)
    
    if not df.empty:
        print(f"Saved diversity results to {args.output_file}")
        print(f"Mean Co-Design Diversity: {df['Co_Div'].mean():.4f}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()

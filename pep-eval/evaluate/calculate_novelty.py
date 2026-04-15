import os
import sys
import argparse
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
from Bio.Align import substitution_matrices, PairwiseAligner
from math import sqrt
import warnings
from Bio import BiopythonWarning

warnings.simplefilter('ignore', BiopythonWarning)

# Fallback VOCAB if PepMimic not available
class VOCAB:
    d = {'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLN':'Q', 'GLU':'E', 
         'GLY':'G', 'HIS':'H', 'ILE':'I', 'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 
         'PRO':'P', 'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'}
    @staticmethod
    def abrv_to_symbol(abrv):
        return VOCAB.d.get(abrv.upper(), None)

def get_sequence(pdb_file, chain_id):
    """
    Extracts sequence from PDB chain.
    """
    if not os.path.exists(pdb_file): return ""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('pdb', pdb_file)
        sequence = []
        for model in structure:
            for chain in model:
                if chain.id == chain_id:
                    for res in chain:
                        if 'CA' in res:
                            resname = res.get_resname()
                            aa = VOCAB.abrv_to_symbol(resname)
                            if aa:
                                sequence.append(aa)
                    return "".join(sequence)
            break
        return ""
    except Exception:
        return ""

def align_score(seqA, seqB, matrix):
    """
    Calculates normalized alignment score (SeqID-like).
    """
    if not seqA or not seqB: return 0.0
    
    try:
        aligner = PairwiseAligner()
        aligner.substitution_matrix = matrix
        
        score_val = aligner.score(seqA, seqB)
        self_A = aligner.score(seqA, seqA)
        self_B = aligner.score(seqB, seqB)
        
        denom = sqrt(self_A * self_B)
        return score_val / denom if denom > 0 else 0.0
    except:
        return 0.0

def load_chain_mapping(mapping_path):
    mapping = {}
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    # complex receptor peptide
                    mapping[parts[0]] = parts[2]
                elif len(parts) == 2:
                    # simpler map: complex peptide
                    mapping[parts[0]] = parts[1]
    return mapping

def main():
    parser = argparse.ArgumentParser(description="Calculate Novelty (SeqID < 0.5 & TM < 0.5)")
    parser.add_argument("--tm_score_file", type=str, required=True, help="Output of calculate_tm_score.py")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to generated_pdbs")
    parser.add_argument("--mapping_file", type=str, required=True, help="Mapping file (Complex Rec Pep)")
    parser.add_argument("--output_file", type=str, default="evaluate/novelty_results.csv")
    args = parser.parse_args()
    
    if not os.path.exists(args.tm_score_file):
        print(f"Error: TM Score file {args.tm_score_file} not found.")
        sys.exit(1)
        
    try:
        df_tm = pd.read_csv(args.tm_score_file, sep='\t')
    except:
        print("Error reading TM score file.")
        sys.exit(1)
        
    # Check if 'TM_score' col exists. If it's old format, might be different
    # Our tm_score script produces: Complex, Model, TM_score
    required_cols = ['Complex', 'Model', 'TM_score']
    if not all(col in df_tm.columns for col in required_cols):
        # Try to infer col names if header missing? 
        # But our script adds header.
        print(f"Error: TM Score file missing columns. Found: {df_tm.columns}")
        sys.exit(1)
        
    mapping = load_chain_mapping(args.mapping_file)
    matrix = substitution_matrices.load('BLOSUM62')
    
    results = []
    
    unique_complexes = df_tm['Complex'].unique()
    
    for complex_id in unique_complexes:
        group = df_tm[df_tm['Complex'] == complex_id]
        
        # Determine peptide chain
        pep_chain = mapping.get(complex_id)
        if not pep_chain:
            parts = complex_id.split('_')
            if len(parts) >= 3:
                pep_chain = parts[2]
            else:
                # print(f"Skipping {complex_id}: No peptide chain info.")
                continue
        
        # Load Native Sequence
        ref_path = os.path.join(args.data_dir, complex_id, "ref.pdb")
        ref_seq = get_sequence(ref_path, pep_chain)
        
        if not ref_seq:
            # print(f"Skipping {complex_id}: No ref seq.")
            continue
            
        for idx, row in group.iterrows():
            model_file = row['Model']
            tm_score = row['TM_score']
            
            gen_path = os.path.join(args.data_dir, complex_id, model_file)
            gen_seq = get_sequence(gen_path, pep_chain)
            
            if not gen_seq:
                continue
                
            seq_id = align_score(ref_seq, gen_seq, matrix)
            
            # Novelty Definition: Both must be low
            is_novel = (seq_id < 0.5) and (tm_score < 0.5)
            
            results.append({
                'Complex': complex_id,
                'Model': model_file,
                'TM_score': tm_score,
                'Seq_ID': seq_id,
                'Is_Novel': is_novel
            })
            
    if results:
        df_out = pd.DataFrame(results)
        novelty_rate = df_out['Is_Novel'].mean()
        print(f"Global Novelty Rate: {novelty_rate:.4f}")
        df_out.to_csv(args.output_file, index=False)
        print(f"Saved results to {args.output_file}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()

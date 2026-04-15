"""
Calculate Secondary Structure Recovery (SSR) for generated peptides.
Self-contained script, does not depend on PepMimic package (except maybe for DSSP tool availability).

Usage:
    python evaluate/calculate_ssr.py \
        --data_dir /home/shiyiming/peptide_design/ManifoldSurface_Peptide_Generator/generated_pdbs \
        --mapping_file /home/shiyiming/peptide_design/ManifoldSurface_Peptide_Generator/data/raw/LNR/test.txt \
        --output_file evaluate/ssr_results.txt
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser, DSSP
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from math import sqrt
from Bio.Align import substitution_matrices, PairwiseAligner

# =============================================================================
# Helper functions copied/adapted from PepMimic to avoid dependency
# =============================================================================

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

def map_dssp_to_3state(dssp_code):
    """
    Map the 8-state DSSP code to 3-state (H, E, C).
    H (Alpha helix), G (3-10 helix), I (Pi helix) -> H
    E (Beta strand), B (Beta bridge) -> E
    T (Turn), S (Bend), - (None), and others -> C
    """
    if dssp_code in ['H', 'G', 'I']:
        return 'H'
    elif dssp_code in ['E', 'B']:
        return 'E'
    else:
        return 'C'

def get_secondary_structure(pdb_file, ligand_chain):
    """
    Extract secondary structure for the ligand chain from PDB file using DSSP.
    Returns 3-state secondary structure string (H, E, C).
    """
    try:
        pdb_file_abs = os.path.abspath(os.path.realpath(pdb_file))
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('pdb', pdb_file_abs)
        model = structure[0]
        
        dssp = None
        # Try finding dssp executable
        for dssp_cmd in ['mkdssp', 'dssp']:
            try:
                dssp = DSSP(model, pdb_file_abs, dssp=dssp_cmd)
                break
            except Exception:
                continue
        
        if dssp is None:
            # Try dssp_dict_from_pdb_file if available (older Biopython or specific setup)
            try:
                dssp_dict, _ = dssp_dict_from_pdb_file(pdb_file_abs)
                ss_labels = []
                found_chain = False
                for chain in model:
                    if chain.id == ligand_chain:
                        found_chain = True
                        for residue in chain:
                            res_id = residue.get_id()
                            dssp_key1 = (model.id, chain.id, res_id)
                            dssp_key2 = (chain.id, res_id)
                            
                            if dssp_key1 in dssp_dict:
                                raw_ss = dssp_dict[dssp_key1][1]
                                ss_labels.append(map_dssp_to_3state(raw_ss))
                            elif dssp_key2 in dssp_dict:
                                raw_ss = dssp_dict[dssp_key2][1]
                                ss_labels.append(map_dssp_to_3state(raw_ss))
                            else:
                                ss_labels.append('C')
                        break
                if found_chain and len(ss_labels) > 0:
                    return ''.join(ss_labels)
            except Exception:
                pass
            return None
        
        # Using DSSP class result
        ss_labels = []
        found_chain = False
        for chain in model:
            if chain.id == ligand_chain:
                found_chain = True
                for residue in chain:
                    res_id = residue.get_id()
                    dssp_key = (chain.id, res_id)
                    try:
                        if dssp_key in dssp:
                            # Index 2 is SS in Biopython DSSP
                            raw_ss = dssp[dssp_key][2]
                            ss_labels.append(map_dssp_to_3state(raw_ss))
                        else:
                            ss_labels.append('C')
                    except Exception:
                         ss_labels.append('C')
                break
        
        if found_chain and len(ss_labels) > 0:
            return ''.join(ss_labels)
            
    except Exception as e:
        # print(f"DSSP Error for {pdb_file}: {e}")
        pass
    return None

def get_sequence_from_pdb(pdb_file, chain_id):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('pdb', pdb_file)
        seq = []
        for model in structure:
            for chain in model:
                if chain.id == chain_id:
                    for res in chain:
                        if 'CA' in res:
                            resname = res.get_resname()
                            s = abrv_to_symbol(resname)
                            if s != '?':
                                seq.append(s)
                    return ''.join(seq)
    except Exception:
        pass
    return ""

def calculate_ssr_score(ref_ss, gen_ss, ref_seq, gen_seq):
    if not ref_ss or not gen_ss:
        return None
    
    # If equal length, simple match
    if len(ref_seq) == len(gen_seq) and len(ref_ss) == len(gen_ss):
        valid_positions = [i for i, ss in enumerate(ref_ss) if ss != '-']
        if not valid_positions:
            # If no SS in ref, match everything (fallback)
            matches = sum(1 for i in range(len(ref_ss)) if ref_ss[i] == gen_ss[i])
            return matches / len(ref_ss) if len(ref_ss) > 0 else 0.0
        
        matches = sum(1 for i in valid_positions if ref_ss[i] == gen_ss[i])
        return matches / len(valid_positions)
    
    # Alignment needed
    (aligned_ref_seq, aligned_gen_seq), _ = align_sequences(ref_seq, gen_seq)
    
    aligned_ref_ss = []
    aligned_gen_ss = []
    
    r_idx, g_idx = 0, 0
    for ra, ga in zip(aligned_ref_seq, aligned_gen_seq):
        if ra != '-':
            aligned_ref_ss.append(ref_ss[r_idx] if r_idx < len(ref_ss) else '-')
            r_idx += 1
        else:
            aligned_ref_ss.append('-')
            
        if ga != '-':
            aligned_gen_ss.append(gen_ss[g_idx] if g_idx < len(gen_ss) else '-')
            g_idx += 1
        else:
            aligned_gen_ss.append('-')
            
    # Calculate matches on aligned
    matches = 0
    ref_count = 0
    for ra, ga, rss, gss in zip(aligned_ref_seq, aligned_gen_seq, aligned_ref_ss, aligned_gen_ss):
        if ra != '-' and ga != '-':
            if rss != '-':
                ref_count += 1
                if rss == gss:
                    matches += 1
                    
    if ref_count == 0:
        # Fallback to all aligned positions
        count = 0
        matches = 0
        for ra, ga, rss, gss in zip(aligned_ref_seq, aligned_gen_seq, aligned_ref_ss, aligned_gen_ss):
            if ra != '-' and ga != '-':
                count += 1
                if rss == gss:
                    matches += 1
        return matches / count if count > 0 else 0.0
        
    return matches / ref_count

def load_chain_mapping(mapping_path):
    mapping = {}
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    pdb_id, rec_chain, pep_chain = parts[0], parts[1], parts[2]
                    mapping[pdb_id] = (rec_chain, pep_chain)
    return mapping

def main():
    parser = argparse.ArgumentParser(description="Calculate SSR for generated peptides.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the directory containing generated PDBs")
    parser.add_argument("--mapping_file", type=str, default="/home/shiyiming/peptide_design/ManifoldSurface_Peptide_Generator/data/raw/LNR/test.txt",
                        help="Path to the chain mapping file")
    parser.add_argument("--output_file", type=str, default="evaluate/ssr_results.txt",
                        help="Path to the output summary file")
    
    args = parser.parse_args()
    
    # Check for DSSP
    print("Checking for DSSP...")
    import shutil
    if not shutil.which('dssp') and not shutil.which('mkdssp'):
        print("Warning: DSSP executable (dssp or mkdssp) not found in PATH.")
        print("SSR calculation will likely fail.")
    
    chain_mapping = load_chain_mapping(args.mapping_file)
    print(f"Loaded {len(chain_mapping)} chain mappings.")
    
    processed_count = 0
    skipped_count = 0
    
    out_dir = os.path.dirname(args.output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    entries = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    
    with open(args.output_file, "w") as f_out:
        f_out.write("Complex\tModel\tSSR\tRef_SS\tGen_SS\n")
        
        for entry in entries:
            entry_dir = os.path.join(args.data_dir, entry)
            
            if entry in chain_mapping:
                _, pep_chain = chain_mapping[entry]
            else:
                parts = entry.split('_')
                if len(parts) == 3:
                     pep_chain = parts[2]
                else: 
                    if entry not in chain_mapping:
                         continue

            orig_pdb = os.path.join(entry_dir, "ref.pdb")
            if not os.path.exists(orig_pdb):
                candidates = [f for f in os.listdir(entry_dir) if f.endswith('.pdb') and not f.startswith('sample_')]
                if candidates:
                    orig_pdb = os.path.join(entry_dir, candidates[0])
                else:
                    continue

            # Get ref SS and Seq
            ref_ss = get_secondary_structure(orig_pdb, pep_chain)
            ref_seq = get_sequence_from_pdb(orig_pdb, pep_chain)
            
            if not ref_ss:
                skipped_count += 1
                continue
            
            gen_files = sorted([f for f in os.listdir(entry_dir) if f.startswith('sample_') and f.endswith('.pdb')])
            
            for gen_file in gen_files:
                gen_path = os.path.join(entry_dir, gen_file)
                gen_ss = get_secondary_structure(gen_path, pep_chain)
                gen_seq = get_sequence_from_pdb(gen_path, pep_chain)
                
                ssr = calculate_ssr_score(ref_ss, gen_ss, ref_seq, gen_seq)
                
                if ssr is not None:
                    # Write SSR score along with the SS strings for inspection
                    f_out.write(f"{entry}\t{gen_file}\t{ssr:.4f}\t{ref_ss}\t{gen_ss}\n")
                    processed_count += 1
                else:
                    skipped_count += 1
            
            f_out.flush()
            
    print(f"Finished. Processed {processed_count} files. Skipped {skipped_count}.")
    
    # Statistics
    if processed_count > 0:
        try:
            df = pd.read_csv(args.output_file, sep='\t')
            if 'SSR' in df.columns:
                df['SSR'] = pd.to_numeric(df['SSR'], errors='coerce')
                df = df.dropna(subset=['SSR'])
                
                complex_means = df.groupby('Complex')['SSR'].mean()
                best_means = df.groupby('Complex')['SSR'].max() # SSR is higher better
                
                print("\n" + "="*40)
                print("SSR Summary Statistics")
                print("="*40)
                print(f"Mean SSR (avg of complex means): {complex_means.mean():.4f}")
                print(f"Mean Best SSR (avg of complex maxs): {best_means.mean():.4f}")
                print(f"Total Complexes: {len(complex_means)}")
        except Exception as e:
            print(f"Could not calculate stats: {e}")

if __name__ == "__main__":
    main()

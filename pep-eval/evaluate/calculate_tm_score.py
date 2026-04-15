import os
import sys
import argparse
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.Align import substitution_matrices, PairwiseAligner
import concurrent.futures
from tqdm import tqdm

# Add PepMimic path
sys.path.insert(0, '/home/shiyiming/peptide_design/PepMimic')
try:
    from data.format import VOCAB
except ImportError:
    class VOCAB:
        d = {'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLN':'Q', 'GLU':'E', 
             'GLY':'G', 'HIS':'H', 'ILE':'I', 'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 
             'PRO':'P', 'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'}
        @staticmethod
        def abrv_to_symbol(abrv):
            return VOCAB.d.get(abrv.upper(), None)

def align_sequences(seq1, seq2):
    try:
        aligner = PairwiseAligner() # default: match=1, mismatch=0, gap_open=0, gap_extend=0
        aligner.mode = 'global'
        alignments = aligner.align(seq1, seq2)
        aln = alignments[0]
        
        # Biopython >1.79 prints FASTA format for alignment strings
        try:
            s = format(aln, "fasta")
            lines = s.strip().split('\n')
            # Lines are >target, seq, >query, seq
            seqs = [l.strip() for l in lines if not l.startswith('>')]
            if len(seqs) >= 2:
                return (seqs[0], seqs[1]), aln.score
        except:
             pass
             
        # Biopython <1.80 fallback (often tuple)
        try:
             s1 = aln[0]
             s2 = aln[1]
             return (str(s1), str(s2)), aln.score
        except:
             pass

        return (seq1, seq2), 0
    except Exception:
        return (seq1, seq2), 0


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
    except Exception as e:
        return None, None

def calculate_tm_score_algo(coords1, coords2, seq1=None, seq2=None):
    """
    Standard TM-score calculation algorithm
    """
    if len(coords1) == 0 or len(coords2) == 0:
        return 0.0
    
    # Sequence Alignment
    if seq1 and seq2 and len(seq1) > 0 and len(seq2) > 0:
        try:
            (aligned_seq1, aligned_seq2), _ = align_sequences(seq1, seq2)
            coords1_aligned, coords2_aligned = [], []
            idx1, idx2 = 0, 0
            for a1, a2 in zip(aligned_seq1, aligned_seq2):
                if a1 != '-' and idx1 < len(coords1):
                    if a2 != '-' and idx2 < len(coords2):
                        coords1_aligned.append(coords1[idx1])
                        coords2_aligned.append(coords2[idx2])
                        idx1 += 1
                        idx2 += 1
                    else:
                        idx1 += 1
                elif a2 != '-' and idx2 < len(coords2):
                    idx2 += 1
            
            if len(coords1_aligned) > 0 and len(coords2_aligned) > 0:
                coords1 = np.array(coords1_aligned)
                coords2 = np.array(coords2_aligned)
        except:
            pass
    
    L_align = min(len(coords1), len(coords2))
    L_ref = max(len(coords1), len(coords2))
    
    if L_align == 0 or L_ref == 0:
        return 0.0
    
    # Kabsch Alignment
    centroid1 = np.mean(coords1[:L_align], axis=0)
    centroid2 = np.mean(coords2[:L_align], axis=0)
    coords1_c = coords1[:L_align] - centroid1
    coords2_c = coords2[:L_align] - centroid2
    
    H = coords1_c.T @ coords2_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    coords1_rot = (R @ coords1_c.T).T + centroid2
    distances = np.linalg.norm(coords1_rot - coords2[:L_align], axis=1)
    
    if L_ref > 15:
        d0 = 1.24 * ((L_ref - 15) ** (1.0/3.0)) - 1.8
        d0 = max(d0, 0.5)
    else:
        d0 = 1.24
        
    tm_score = (1.0 / L_ref) * np.sum(1.0 / (1.0 + (distances / d0) ** 2))
    return tm_score

def process_entry(args):
    entry, entry_dir, native_pdb, pep_chain = args
    results = []
    
    if not os.path.exists(native_pdb):
        return []

    # Get Native
    ref_coords, ref_seq = get_ca_coords_and_seq(native_pdb, pep_chain)
    if ref_coords is None:
        return []

    gen_files = sorted([f for f in os.listdir(entry_dir) if f.startswith('sample_') and f.endswith('.pdb') and 'rosetta' not in f])
    
    for gen_file in gen_files:
        gen_path = os.path.join(entry_dir, gen_file)
        gen_coords, gen_seq = get_ca_coords_and_seq(gen_path, pep_chain)
        
        if gen_coords is not None:
            score = calculate_tm_score_algo(ref_coords, gen_coords, ref_seq, gen_seq)
            results.append((entry, gen_file, score))
            
    return results

def load_chain_mapping(mapping_path):
    mapping = {}
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                     # pdb rec pep
                    mapping[parts[0]] = (parts[1], parts[2])
    return mapping

def main():
    parser = argparse.ArgumentParser(description="Calculate TM-score")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--mapping_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="evaluate/tm_score_results.txt")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args()
    
    mapping = load_chain_mapping(args.mapping_file)
    native_dir = os.path.join(os.path.dirname(args.mapping_file), "pdbs")
    
    entries = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    tasks = []
    
    for entry in entries:
        if entry in mapping:
            _, pep_chain = mapping[entry]
        else:
             parts = entry.split('_')
             if len(parts) >= 3: pep_chain = parts[2]
             else: continue
             
        # Priority 1: ref.pdb (local frame)
        native_pdb = os.path.join(args.data_dir, entry, "ref.pdb")
        if not os.path.exists(native_pdb):
             # Priority 2: global pdb
             native_pdb = os.path.join(native_dir, f"{entry}.pdb")
        
        tasks.append((entry, os.path.join(args.data_dir, entry), native_pdb, pep_chain))
        
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_entry, task) for task in tasks]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(tasks)):
            all_results.extend(f.result())
            
    # Write
    out_dir = os.path.dirname(args.output_file)
    if out_dir and not os.path.exists(out_dir): os.makedirs(out_dir)
    
    with open(args.output_file, "w") as f:
        f.write("Complex\tModel\tTM_score\n")
        all_results.sort()
        for r in all_results:
            f.write(f"{r[0]}\t{r[1]}\t{r[2]:.4f}\n")
            
    print(f"Saved {len(all_results)} results to {args.output_file}")

if __name__ == "__main__":
    main()

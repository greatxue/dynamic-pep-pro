import os
import sys
import argparse
import pandas as pd
import numpy as np
import warnings
from Bio.PDB import PDBParser, Polypeptide
try:
    from Bio.PDB.Polypeptide import three_to_one as _three_to_one
except ImportError:
    # Fallback for older/newer versions if structure changed
    try:
        from Bio.PDB.Polypeptide import three_to_index, index_to_one
        def _three_to_one(s):
            return index_to_one(three_to_index(s))
    except ImportError:
        # Manual map as last resort
        _aa3to1 = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }
        def _three_to_one(s):
            return _aa3to1.get(s.upper(), None)

# Suppress PDB construction warnings
warnings.filterwarnings("ignore")

def three_to_one(resname):
    """
    Convert 3-letter amino acid code to 1-letter code.
    Returns None if not a standard amino acid.
    """
    try:
        return _three_to_one(resname)
    except (KeyError, ValueError, AttributeError):
        return None

def extract_sequence_from_pdb(pdb_path, chain_id):
    """
    Extracts the amino acid sequence for a specific chain from a PDB file.
    """
    try:
        if not os.path.exists(pdb_path):
            return None
            
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("struct", pdb_path)
        
        sequence = []
        
        # Iterate through models (usually just one)
        for model in structure:
            # Check if chain exists in this model
            if chain_id in model:
                chain = model[chain_id]
                for residue in chain:
                    # Filter for amino acids (skip waters, hetatoms that aren't AA)
                    if residue.id[0] == ' ': # Standard residue
                        resname = residue.get_resname()
                        one_letter = three_to_one(resname)
                        if one_letter:
                            sequence.append(one_letter)
                return "".join(sequence)
                
        return None
    except Exception as e:
        print(f"Error extracting sequence from {pdb_path}: {e}")
        return None

def calculate_aar_metric(candidate, reference):
    """
    Calculates simple Amino Acid Recovery (AAR): fraction of matching positions.
    Assumes equal length.
    """
    if len(candidate) != len(reference):
        pass # Should ideally be same length, but zip handles it by truncation
        
    hit = 0
    for a, b in zip(candidate, reference):
        if a == b:
            hit += 1
    return hit / len(reference) if len(reference) > 0 else 0

def slide_aar(candidate, reference):
    """
    Sliding window AAR to find best alignment without gaps.
    Pads candidate to allow partial overlaps.
    """
    if not reference:
        return 0.0
        
    special_token = ' ' # Represents gap/padding
    ref_len = len(reference)
    
    # Pad candidate to allow sliding
    padded_candidate = special_token * (ref_len - 1) + candidate + special_token * (ref_len - 1)
    
    max_score = 0.0
    
    # Slide window of size ref_len across padded_candidate
    for start in range(len(padded_candidate) - ref_len + 1):
        segment = padded_candidate[start:start + ref_len]
        
        # Calculate AAR for this alignment
        score = calculate_aar_metric(segment, reference)
        if score > max_score:
            max_score = score
            
    return max_score

def compute_aar_for_pair(original_pdb, generated_pdb, lig_chain, debug=False):
    """
    Wraps extraction and calculation logic.
    """
    ref_seq = extract_sequence_from_pdb(original_pdb, lig_chain)
    gen_seq = extract_sequence_from_pdb(generated_pdb, lig_chain)
    
    if not ref_seq:
        if debug: print(f"Could not extract reference sequence for {original_pdb}")
        return None
    
    if not gen_seq:
        if debug: print(f"Could not extract generated sequence for {generated_pdb}")
        return None
        
    aar_value = slide_aar(gen_seq, ref_seq)
    
    if debug:
        print(f"Ref seq: {ref_seq}")
        print(f"Gen seq: {gen_seq}")
        print(f"AAR: {aar_value}")
    
    return aar_value

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
    parser = argparse.ArgumentParser(description="Calculate AAR for generated peptides.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the directory containing generated PDBs (e.g. generated_pdbs)")
    parser.add_argument("--mapping_file", type=str, default="/home/shiyiming/peptide_design/ManifoldSurface_Peptide_Generator/data/raw/LNR/test.txt",
                        help="Path to the chain mapping file (pdb_id rec_chain pep_chain)")
    parser.add_argument("--output_file", type=str, default="AAR_summary.txt",
                        help="Path to the output summary file")
    
    args = parser.parse_args()

    base_dir = args.data_dir
    output_summary = args.output_file
    
    # Load mapping
    chain_mapping = load_chain_mapping(args.mapping_file)
    print(f"Loaded {len(chain_mapping)} chain mappings.")

    print(f"Data directory: {base_dir}")
    print(f"Output file: {output_summary}")

    entries = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    processed_count = 0
    
    with open(output_summary, "w") as summary_out:
        summary_out.write("Complex\tModel\tAAR\n")
        
        for entry in entries:
            entry_dir = os.path.join(base_dir, entry)
            
            # Determine peptide chain ID
            if entry in chain_mapping:
                _, pep_chain = chain_mapping[entry]
            else:
                # Default fallback or skip? Let's try 'P' or 'B' if not found, but better to skip if strict
                # Or try to infer from ref.pdb?
                print(f"Warning: No mapping for {entry}, skipping.")
                continue

            # Reference PDB (in the same folder, usually 'ref.pdb')
            # The generation script saves 'ref.pdb' in the sample directory
            original_pdb = os.path.join(entry_dir, "ref.pdb")
            if not os.path.isfile(original_pdb):
                print(f"Reference not found for {entry}")
                continue
            
            # Identify generated samples
            generated_files = sorted([f for f in os.listdir(entry_dir) if f.startswith('sample_') and f.endswith('.pdb')])
            
            for gen_filename in generated_files:
                generated_pdb = os.path.join(entry_dir, gen_filename)
                
                # Calculate AAR using the peptide chain
                aar_val = compute_aar_for_pair(original_pdb, generated_pdb, pep_chain)
                
                if aar_val is not None:
                    summary_out.write(f"{entry}\t{gen_filename}\t{aar_val}\n")
                    processed_count += 1
            
            summary_out.flush()

    print(f"\nFinished. Processed {processed_count} files.")
    
    # 3. Quick Analysis
    if processed_count > 0:
        try:
            df = pd.read_csv(output_summary, sep='\t')
            # Clean possible duplicates or headers
            if 'Complex' in df.columns:
                 df = df[df['Complex'] != 'Complex']
            
            if 'AAR' in df.columns:
                df['AAR'] = pd.to_numeric(df['AAR'], errors='coerce')
                df = df.dropna(subset=['AAR'])
            
                complex_means = df.groupby('Complex')['AAR'].mean()
                best_means = df.groupby('Complex')['AAR'].max()
                
                print("\n" + "="*40)
                print("Summary Statistics")
                print("="*40)
                print(f"Mean AAR (avg of complex means): {complex_means.mean():.4f}")
                print(f"Mean Best AAR (avg of complex bests): {best_means.mean():.4f}")
                print(f"Total Complexes: {len(complex_means)}")
            
        except Exception as e:
            print(f"Could not run analysis: {e}")

if __name__ == "__main__":
    main()

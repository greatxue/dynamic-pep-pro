"""
Calculate Spatial SSR v2 (Ligand-Centered) for generated peptides.

Algorithm:
1. For each reference ligand residue (CA), define a sphere (radius R).
2. Collect predicted ligand residues falling into this sphere.
3. Check if the reference secondary structure type exists in the predicted set.
4. Compute recovery rate (Ref -> Pred).
5. Do the reverse (Pred -> Ref) for symmetry.
6. Metric = sqrt(Ref_Recovery * Pred_Recovery).

Usage:
    python evaluate/calculate_spatial_ssr_v2.py \
        --data_dir /home/shiyiming/peptide_design/ManifoldSurface_Peptide_Generator/generated_pdbs \
        --mapping_file /home/shiyiming/peptide_design/ManifoldSurface_Peptide_Generator/data/raw/LNR/test.txt \
        --output_file evaluate/spatial_ssr_v2_results.txt \
        --radius 2.0
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser, DSSP
from shutil import which
from scipy.spatial import cKDTree
import warnings

warnings.filterwarnings('ignore')

# Matches the 3-state mapping used in standard SSR
def map_dssp_to_3state(dssp_code):
    if dssp_code in ['H', 'G', 'I']:
        return 'H'
    elif dssp_code in ['E', 'B']:
        return 'E'
    else:
        return 'C'

def resolve_dssp_binary():
    return which('mkdssp') or which('dssp')

def get_secondary_structure(pdb_file, chain_id=None):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_file)
        model = structure[0]
        
        dssp_bin = resolve_dssp_binary()
        if dssp_bin is None:
            return None
            
        dssp = DSSP(model, pdb_file, dssp=dssp_bin)
        ss_list = []
        res_ids = []

        # DSSP keys are (chain_id, res_id)
        # We need to be careful about matching
        
        # Iterate over residues in the chain to maintain order and ID
        if chain_id and chain_id in model:
            chain = model[chain_id]
            for res in chain:
                dssp_key = (chain_id, res.id)
                if dssp_key in dssp:
                    raw_ss = dssp[dssp_key][2]
                    ss_list.append(map_dssp_to_3state(raw_ss))
                else:
                    ss_list.append('C') # Default to Coil if missing
                res_ids.append(res.id)
        else:
            return None, None

        return ss_list, res_ids
    except Exception:
        return None, None

def get_residue_ca_coord(structure, chain_id, res_id):
    model = structure[0]
    if chain_id not in model:
        return None
    chain = model[chain_id]
    if res_id not in chain:
        return None
    residue = chain[res_id]
    if 'CA' in residue:
        return residue['CA'].coord
    return None

def get_ligand_residues_in_sphere(structure, ligand_chain_id, center_ca, d_sphere=2.0):
    model = structure[0]
    if ligand_chain_id not in model:
        return []

    chain = model[ligand_chain_id]
    ligand_ca = []
    for residue in chain:
        if 'CA' in residue:
            ligand_ca.append((residue.id, residue['CA'].coord))

    if not ligand_ca:
        return []

    ligand_coords = np.array([a[1] for a in ligand_ca])
    ligand_res_ids = [a[0] for a in ligand_ca]
    tree = cKDTree(ligand_coords)

    covered_residues = []
    idxs = tree.query_ball_point(center_ca, r=d_sphere)
    for idx in idxs:
        covered_residues.append(ligand_res_ids[idx])
    return covered_residues

def build_chain_ss_dict(pdb_file, chain_id):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    ss_list, res_ids = get_secondary_structure(pdb_file, chain_id=chain_id)
    
    if ss_list is None:
        # Fallback: if DSSP fails, try to just load structure? 
        # But we need SS for this metric.
        return {}, structure

    ss_dict = dict(zip(res_ids, ss_list))
    return ss_dict, structure

def compute_spatial_ssr_v2(ref_pdb, pred_pdb, ligand_chain_id, d_sphere=2.0):
    
    # 1. Build Reference Info
    ref_lig_ss, ref_structure = build_chain_ss_dict(ref_pdb, ligand_chain_id)
    if not ref_lig_ss:
        return None 

    model = ref_structure[0]
    if ligand_chain_id not in model:
        return None

    ligand_chain = model[ligand_chain_id]
    ligand_res_ids = [res.id for res in ligand_chain if 'CA' in res] # Only CA containing residues
    
    if len(ligand_res_ids) == 0:
        return None

    # Precompute reference CA centers
    ref_ca_map = {}
    for lig_res_id in ligand_res_ids:
        ref_ca = get_residue_ca_coord(ref_structure, ligand_chain_id, lig_res_id)
        if ref_ca is not None:
            ref_ca_map[lig_res_id] = ref_ca

    # 2. Build Prediction Info
    pred_lig_ss, pred_structure = build_chain_ss_dict(pred_pdb, ligand_chain_id)
    if not pred_lig_ss:
        return None

    pred_model = pred_structure[0]
    if ligand_chain_id not in pred_model:
        return None

    pred_chain = pred_model[ligand_chain_id]
    pred_res_ids = [res.id for res in pred_chain if 'CA' in res]

    # === Ref -> Pred Recovery ===
    recovered_ref = 0
    valid_ref = 0

    for lig_res_id in ligand_res_ids:
        ref_ss = ref_lig_ss.get(lig_res_id, '?')
        if ref_ss == '?': continue
        
        ref_ca = ref_ca_map.get(lig_res_id)
        if ref_ca is None: continue

        covered_residues = get_ligand_residues_in_sphere(
            pred_structure, ligand_chain_id, ref_ca, d_sphere
        )
        
        pred_ss_list = []
        for res_id in covered_residues:
            if res_id in pred_lig_ss:
                pred_ss_list.append(pred_lig_ss[res_id])
        
        valid_ref += 1
        if ref_ss in pred_ss_list:
            recovered_ref += 1

    # === Pred -> Ref Recovery ===
    recovered_pred = 0
    valid_pred = 0

    for pred_res_id in pred_res_ids:
        pred_ss = pred_lig_ss.get(pred_res_id, '?')
        if pred_ss == '?': continue
        
        pred_ca = get_residue_ca_coord(pred_structure, ligand_chain_id, pred_res_id)
        if pred_ca is None: continue

        covered_residues = get_ligand_residues_in_sphere(
            ref_structure, ligand_chain_id, pred_ca, d_sphere
        )
        
        ref_ss_list = []
        for res_id in covered_residues:
            if res_id in ref_lig_ss:
                ref_ss_list.append(ref_lig_ss[res_id])
                
        valid_pred += 1
        if pred_ss in ref_ss_list:
            recovered_pred += 1

    ref_recovery = float(recovered_ref / valid_ref) if valid_ref > 0 else 0.0
    pred_recovery = float(recovered_pred / valid_pred) if valid_pred > 0 else 0.0
    spatial_ssr_v2 = float(np.sqrt(ref_recovery * pred_recovery))

    return spatial_ssr_v2

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

def main():
    parser = argparse.ArgumentParser(description="Calculate Spatial SSR v2 (Ligand-Centered).")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to directory containing generated PDBs")
    parser.add_argument("--mapping_file", type=str, default="/home/shiyiming/peptide_design/ManifoldSurface_Peptide_Generator/data/raw/LNR/test.txt",
                        help="Path to chain mapping file")
    parser.add_argument("--output_file", type=str, default="evaluate/spatial_ssr_v2_results.txt",
                        help="Path to output summary file")
    parser.add_argument("--radius", type=float, default=2.0,
                        help="Sphere radius in Angstrom (default: 2.0)")
    
    args = parser.parse_args()
    
    # Check for DSSP
    if resolve_dssp_binary() is None:
        print("Error: DSSP executable (mkdssp or dssp) not found in PATH.")
        sys.exit(1)
        
    chain_mapping = load_chain_mapping(args.mapping_file)
    print(f"Loaded {len(chain_mapping)} chain mappings.")
    
    processed_count = 0
    skipped_count = 0
    
    out_dir = os.path.dirname(args.output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    entries = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    
    with open(args.output_file, "w") as f_out:
        f_out.write("Complex\tModel\tSpatialSSR\n")
        
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
            
            # Reference PDB
            orig_pdb = os.path.join(entry_dir, "ref.pdb")
            if not os.path.exists(orig_pdb):
                candidates = [f for f in os.listdir(entry_dir) if f.endswith('gt.pdb') or f.endswith('ref.pdb') or (f.endswith('.pdb') and 'sample' not in f and not any(char.isdigit() for char in f.replace('.pdb','').split('_')[-1]))]
                if candidates:
                    orig_pdb = os.path.join(entry_dir, candidates[0])
                else:
                    target_pdb_id = entry.split('_')[0]
                    candidates_2 = [f for f in os.listdir(entry_dir) if f.endswith('.pdb')]
                    if candidates_2: # just pick the first one not sample, or gt
                        orig_pdb = os.path.join(entry_dir, candidates_2[0])
                    else:
                        continue
            
            # 1. Parse Ref Once
            ref_lig_ss, ref_structure = build_chain_ss_dict(orig_pdb, pep_chain)
            if not ref_lig_ss:
                skipped_count += 40 # approx
                continue
                
            # Precompute ref centers
            ref_model = ref_structure[0]
            if pep_chain not in ref_model:
                continue
            ref_lig_chain = ref_model[pep_chain]
            ref_lig_res_ids = [res.id for res in ref_lig_chain if 'CA' in res]
            
            ref_ca_map = {}
            for lig_res_id in ref_lig_res_ids:
                if 'CA' in ref_lig_chain[lig_res_id]:
                    ref_ca_map[lig_res_id] = ref_lig_chain[lig_res_id]['CA'].coord
            
            gen_files = sorted([f for f in os.listdir(entry_dir) if (f.startswith('sample_') and f.endswith('.pdb')) or (f.startswith(entry) and f.endswith('.pdb') and not f.endswith('gt.pdb'))])
            
            for gen_file in gen_files:
                gen_path = os.path.join(entry_dir, gen_file)
                
                # Optimized Inner Loop Logic
                pred_lig_ss, pred_structure = build_chain_ss_dict(gen_path, pep_chain)
                if not pred_lig_ss:
                    skipped_count += 1
                    continue
                
                pred_model = pred_structure[0]
                if pep_chain not in pred_model:
                    skipped_count += 1
                    continue
                    
                pred_chain = pred_model[pep_chain]
                pred_res_ids = [res.id for res in pred_chain if 'CA' in res]
                
                # === Ref -> Pred ===
                recovered_ref = 0
                valid_ref = 0
                for lig_res_id in ref_lig_res_ids:
                    ref_ss = ref_lig_ss.get(lig_res_id, 'C')
                    ref_ca = ref_ca_map.get(lig_res_id)
                    
                    covered = get_ligand_residues_in_sphere(pred_structure, pep_chain, ref_ca, args.radius)
                    pred_ss_list = [pred_lig_ss.get(rid, 'C') for rid in covered]
                    
                    valid_ref += 1
                    if ref_ss in pred_ss_list:
                        recovered_ref += 1
                        
                # === Pred -> Ref ===
                recovered_pred = 0
                valid_pred = 0
                for pred_res_id in pred_res_ids:
                    pred_ss = pred_lig_ss.get(pred_res_id, 'C')
                    if 'CA' not in pred_chain[pred_res_id]: continue
                    pred_ca = pred_chain[pred_res_id]['CA'].coord
                    
                    covered = get_ligand_residues_in_sphere(ref_structure, pep_chain, pred_ca, args.radius)
                    ref_ss_list = [ref_lig_ss.get(rid, 'C') for rid in covered]
                    
                    valid_pred += 1
                    if pred_ss in ref_ss_list:
                        recovered_pred += 1
                        
                r_rate = recovered_ref / valid_ref if valid_ref > 0 else 0
                p_rate = recovered_pred / valid_pred if valid_pred > 0 else 0
                score = np.sqrt(r_rate * p_rate)
                
                f_out.write(f"{entry}\t{gen_file}\t{score:.4f}\n")
                processed_count += 1
            
            f_out.flush()
            
    print(f"Finished. Processed {processed_count} files. Skipped {skipped_count}.")
    
    # Statistics
    if processed_count > 0:
        try:
            df = pd.read_csv(args.output_file, sep='\t')
            if 'SpatialSSR' in df.columns:
                df['SpatialSSR'] = pd.to_numeric(df['SpatialSSR'], errors='coerce')
                df = df.dropna(subset=['SpatialSSR'])
                
                complex_means = df.groupby('Complex')['SpatialSSR'].mean()
                best_means = df.groupby('Complex')['SpatialSSR'].max() 
                
                print("\n" + "="*40)
                print("Spatial SSR v2 Summary Statistics")
                print("="*40)
                print(f"Mean SpatialSSR (avg of complex means): {complex_means.mean():.4f}")
                print(f"Mean Best SpatialSSR (avg of complex maxs): {best_means.mean():.4f}")
                print(f"Total Complexes: {len(complex_means)}")
        except Exception as e:
            print(f"Could not calculate stats: {e}")

if __name__ == "__main__":
    main()

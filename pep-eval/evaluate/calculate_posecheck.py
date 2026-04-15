import argparse
import os
import sys
import subprocess
import tempfile
import numpy as np
import csv
import glob
from Bio.PDB import PDBParser, PDBIO, Select
from rdkit import Chem
from rdkit import RDLogger
import types
import importlib.util
from Bio import BiopythonWarning
import warnings

warnings.simplefilter('ignore', BiopythonWarning)
RDLogger.DisableLog('rdApp.*')

# --------------------------------------------------------------------------------
# HACK: Bypass broken pandas dependency in posecheck
# We manually load the modules we need, ensuring we don't trigger the top-level
# posecheck.__init__ which imports pandas.
# --------------------------------------------------------------------------------

# 1. Mock the packages to prevent __init__.py execution
sys.modules['posecheck'] = types.ModuleType('posecheck')
sys.modules['posecheck.utils'] = types.ModuleType('posecheck.utils')

# 2. Load posecheck.utils.chem first (dependency of clashes)
chem_path = "/home/shiyiming/miniconda3/envs/peptide_gen/lib/python3.11/site-packages/posecheck/utils/chem.py"
spec = importlib.util.spec_from_file_location("posecheck.utils.chem", chem_path)
chem_module = importlib.util.module_from_spec(spec)
sys.modules["posecheck.utils.chem"] = chem_module
spec.loader.exec_module(chem_module)

# 3. Load posecheck.utils.clashes
clashes_path = "/home/shiyiming/miniconda3/envs/peptide_gen/lib/python3.11/site-packages/posecheck/utils/clashes.py"
spec = importlib.util.spec_from_file_location("posecheck.utils.clashes", clashes_path)
clashes_module = importlib.util.module_from_spec(spec)
sys.modules["posecheck.utils.clashes"] = clashes_module
spec.loader.exec_module(clashes_module)

count_clashes = clashes_module.count_clashes
# --------------------------------------------------------------------------------


def run_hydride(input_pdb, output_pdb):
    """Runs hydride to add hydrogens to the protein."""
    import sys, os
    hydride_bin = os.path.join(os.path.dirname(sys.executable), "hydride")
    if not os.path.exists(hydride_bin):
        import shutil
        hydride_bin = shutil.which("hydride") or "hydride"
        
    cmd = [hydride_bin, "-i", str(input_pdb), "-o", str(output_pdb)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=45)
        return True
    except subprocess.CalledProcessError:
        return False
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] hydride timed out after 45s for {input_pdb}", file=sys.stderr)
        return False

def remove_connect_lines(pdb_path):
    with open(pdb_path, 'r') as f:
        lines = f.readlines()
    with open(pdb_path, 'w') as f:
        for line in lines:
            if not line.startswith("CONECT"):
                f.write(line)

def load_protein_molecule(pdb_path):
    """
    Loads a protein PDB, adds hydrogens using hydride, and returns RDKit Mol.
    """
    # Create temp files
    fd, tmp_prot_path = tempfile.mkstemp(suffix=".pdb")
    os.close(fd)
    
    try:
        # Run hydride
        if run_hydride(pdb_path, tmp_prot_path):
            remove_connect_lines(tmp_prot_path)
            mol = Chem.MolFromPDBFile(tmp_prot_path, removeHs=False, sanitize=False)
            return mol
        else:
            return None
    finally:
        if os.path.exists(tmp_prot_path):
            os.remove(tmp_prot_path)

def calculate_intra_clashes(mol, tolerance=0.5):
    """Calculate Intramolecular Clashes manually (simplified)"""
    try:
        if mol is None: return np.nan
        conf = mol.GetConformer()
        pos = conf.GetPositions()
        num_atoms = mol.GetNumAtoms()
        
        topo_dist_matrix = Chem.GetDistanceMatrix(mol)
        
        pt = Chem.GetPeriodicTable()
        vdw_radii = np.array([pt.GetRvdw(mol.GetAtomWithIdx(i).GetAtomicNum()) for i in range(num_atoms)])
        
        dists = np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=-1)
        vdw_sum = vdw_radii[:, np.newaxis] + vdw_radii[np.newaxis, :]
        
        # Check clashes: distance < sum of vdw, and atoms are not bonded (topo dist > 3)
        clash_mask = (dists + tolerance < vdw_sum) & (topo_dist_matrix > 3)
        clashes = np.count_nonzero(np.triu(clash_mask, k=1))
        
        return clashes
    except Exception:
        return np.nan

class ChainSelect(Select):
    """Select specific chain"""
    def __init__(self, chain_id):
        self.chain_id = chain_id
    
    def accept_chain(self, chain):
        if chain.get_id() == self.chain_id:
            return 1
        return 0

def main():
    parser = argparse.ArgumentParser(description="Calculate PoseCheck Metrics (Inter/Intra Clashes)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--mapping_file", type=str, help="Path to mapping file (pdb rec pep)", default=None)
    args = parser.parse_args()

    # Create output directory
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load mapping
    mapping = {}
    if args.mapping_file and os.path.exists(args.mapping_file):
        with open(args.mapping_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                     # pdb rec pep
                    mapping[parts[0]] = (parts[1], parts[2])
    
    processed = set()
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, 'r') as f_csv:
                reader = csv.DictReader(f_csv)
                for row in reader:
                    processed.add((row['Complex'], row['Model']))
            print(f"Loaded {len(processed)} previously evaluated samples. Resuming...")
        except Exception:
            pass

    results = []
    
    if not os.path.exists(args.data_dir):
        print(f"Data directory {args.data_dir} does not exist.")
        return

    entries = sorted(os.listdir(args.data_dir))
    for entry in entries:
        entry_dir = os.path.join(args.data_dir, entry)
        if not os.path.isdir(entry_dir):
            continue

        # Determine chains
        rec_chain = 'A'
        pep_chain = 'B'
        if entry in mapping:
            rec_chain, pep_chain = mapping[entry]
        
        ref_pdb = os.path.join(entry_dir, "ref.pdb")
        if not os.path.exists(ref_pdb):
            # Try to see if it's skipped or maybe just print warning
            # print(f"Skipping {entry}: ref.pdb not found.")
            continue
            
        print(f"Processing {entry}...")

        # Extract and Load Receptor
        fd, rec_tmp_file = tempfile.mkstemp(suffix=".pdb")
        os.close(fd)
        
        parser = PDBParser(QUIET=True)
        rec_mol = None
        try:
             structure = parser.get_structure("ref", ref_pdb)
             io = PDBIO()
             io.set_structure(structure)
             io.save(rec_tmp_file, ChainSelect(rec_chain))
             rec_mol = load_protein_molecule(rec_tmp_file)
        except Exception as e:
            print(f"  Failed to load receptor for {entry}: {e}")
        finally:
            if os.path.exists(rec_tmp_file): os.remove(rec_tmp_file)
            
        if rec_mol is None:
            print(f"  Failed to create RDKit Mol for receptor {entry}")
            continue

        # Process Samples
        sample_files = sorted(glob.glob(os.path.join(entry_dir, "sample_*.pdb")))
        for sample_file in sample_files:
            sample_name = os.path.basename(sample_file)
            
            if (entry, sample_name) in processed:
                continue
                
            fd, pep_tmp_file = tempfile.mkstemp(suffix=".pdb")
            os.close(fd)
            
            try:
                # We need to extract the peptide chain from the sample PDB
                structure = parser.get_structure("sample", sample_file)
                io = PDBIO()
                io.set_structure(structure)
                io.save(pep_tmp_file, ChainSelect(pep_chain))
                
                pep_mol = load_protein_molecule(pep_tmp_file)
                
                if pep_mol:
                    try:
                        inter = count_clashes(rec_mol, pep_mol)
                    except Exception as e:
                        print(f"  Clash calc failed for {sample_name}: {e}")
                        inter = np.nan
                        
                    intra = calculate_intra_clashes(pep_mol)
                    
                    row_data = {
                        "Complex": entry,
                        "Model": sample_name,
                        "Clashes_Inter": inter,
                        "Clashes_Intra": intra
                    }
                    results.append(row_data)
                    
                    # Incremental save
                    file_exists = os.path.exists(args.output_file)
                    with open(args.output_file, 'a', newline='') as f_csv:
                        writer = csv.DictWriter(f_csv, fieldnames=["Complex", "Model", "Clashes_Inter", "Clashes_Intra"])
                        if not file_exists or os.path.getsize(args.output_file) == 0:
                            writer.writeheader()
                        writer.writerow(row_data)
                    
                else:
                    print(f"  Failed to load peptide mol for {sample_name}")

            except Exception as e:
                print(f"  Error processing {sample_name}: {e}")
            finally:
                if os.path.exists(pep_tmp_file): os.remove(pep_tmp_file)

    # Stats
    if results or processed:
        try:
            # We already saved incrementally, just reload everything for stats
            all_res = []
            if os.path.exists(args.output_file):
                with open(args.output_file, 'r') as f_csv:
                    for row in csv.DictReader(f_csv):
                        if row['Clashes_Inter']:
                            row['Clashes_Inter'] = float(row['Clashes_Inter'])
                        if row['Clashes_Intra']:
                            row['Clashes_Intra'] = float(row['Clashes_Intra'])
                        all_res.append(row)
            results = all_res
            
            # Stats
            inter_vals = [r['Clashes_Inter'] for r in results if r['Clashes_Inter'] is not None and not np.isnan(r['Clashes_Inter'])]
            intra_vals = [r['Clashes_Intra'] for r in results if r['Clashes_Intra'] is not None and not np.isnan(r['Clashes_Intra'])]
            
            print(f"Saved results to {args.output_file}")
            
            if inter_vals:
                avg_inter = sum(inter_vals) / len(inter_vals)
                print(f"Average Inter-molecular Clashes: {avg_inter:.2f}")
            if intra_vals:
                avg_intra = sum(intra_vals) / len(intra_vals)
                print(f"Average Intra-molecular Clashes: {avg_intra:.2f}")
                
        except Exception as e:
            print(f"Error saving results: {e}")
            
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()

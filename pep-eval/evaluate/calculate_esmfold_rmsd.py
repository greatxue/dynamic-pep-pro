#!/usr/bin/env python3
import os
import sys
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from Bio.PDB import PDBParser
from Bio.SVDSuperimposer import SVDSuperimposer
from tqdm import tqdm
import warnings
import argparse

# 忽略 Biopython 的一些警告
warnings.filterwarnings('ignore')

# =============================================================================
# 1. 配置与常量
# =============================================================================

# 工作空间根目录（默认为脚本所在目录）
WORKSPACE = Path("/home/shiyiming/peptide_design")
ESM_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"

# 模型路径配置
MODEL_CONFIGS = {
    'pepbridge': {
        'ref_dir': WORKSPACE / "Pepbridge/LNR1_pdbs",
        'ref_pattern': "{pdb_id}_{chain}/{pdb_id}_{chain}_gt.pdb",
        'gen_pattern': "{pdb_id}_{chain}/{pdb_id}_{chain}_gen_{i}.pdb",
        'num_samples': 40
    },
    'pepflowww': {
        'ref_dir': WORKSPACE / "PepMimic/example_data/LNR1",
        'ref_pattern': "{pdb_id}/{pdb_id}.pdb",
        'gen_dir': WORKSPACE / "PepFlowww/outputs/LNR1_generated",
        'gen_pattern': "{pdb_id}_{chain}/{pdb_id}_{chain}_{i}.pdb",
        'num_samples': 40
    },
    'probayes': {
        'ref_dir': WORKSPACE / "ProBayes/pepbench_codesign/checkpoints/results/references",
        'ref_pattern': "{pdb_id}_ref.pdb",
        'gen_dir': WORKSPACE / "ProBayes/pepbench_codesign/checkpoints/results/candidates",
        'gen_pattern': "{pdb_id}/{pdb_id}_gen_{i}.pdb",
        'num_samples': 40
    },
    'pepmimic': {
        'ref_dir': WORKSPACE / "PepMimic/example_data/LNR1",
        'ref_pattern': "{pdb_id}/{pdb_id}.pdb",
        'gen_dir': WORKSPACE / "PepMimic/example_data/LNR1",
        'gen_pattern': "{pdb_id}/results/{pdb_id}/{pdb_id}_gen_{i}.pdb",
        'num_samples': 40
    }
}

# =============================================================================
# 2. 核心辅助函数
# =============================================================================

def esmfold_api(sequence, max_retries=5, retry_delay=2, verbose=False):
    """使用 ESM API 预测结构"""
    sequence = sequence.strip().upper()
    for attempt in range(max_retries):
        try:
            if verbose: print(f"  发送 API 请求 (尝试 {attempt + 1}/{max_retries})...")
            response = requests.post(ESM_API_URL, data=sequence, timeout=60)
            
            if response.status_code == 200:
                if verbose: print(f"  ✓ API 调用成功")
                return response.text
            elif response.status_code == 429:
                wait_time = retry_delay * (attempt + 1)
                print(f"   ⚠️ 速率限制，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                continue
            else:
                if verbose: print(f"  ✗ API 错误: {response.status_code}")
                raise Exception(f"API error: {response.status_code}")
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise e
    raise Exception(f"Failed after {max_retries} attempts")

def extract_sequence_from_pdb(pdb_file, chain_id):
    """从 PDB 中提取序列"""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('peptide', pdb_file)
        model = structure[0]
        if chain_id not in model: return None
        chain = model[chain_id]
        
        aa_map = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
            'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
            'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
            'TYR': 'Y', 'VAL': 'V'
        }
        sequence = "".join([aa_map[res.get_resname()] for res in chain 
                           if res.id[0] == ' ' and res.get_resname() in aa_map])
        return sequence
    except:
        return None

def get_ca_coords(pdb_file, chain_id):
    """提取 CA 原子坐标"""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('peptide', pdb_file)
        model = structure[0]
        if chain_id not in model: return None
        coords = [res['CA'].get_coord() for res in model[chain_id] 
                 if res.id[0] == ' ' and 'CA' in res]
        return np.array(coords) if coords else None
    except:
        return None

def calculate_rmsd(coords1, coords2):
    """计算对齐后的 RMSD"""
    if coords1 is None or coords2 is None or len(coords1) != len(coords2):
        return None
    sup = SVDSuperimposer()
    sup.set(coords1, coords2)
    sup.run()
    return sup.get_rms()

# =============================================================================
# 3. 评估逻辑
# =============================================================================

def evaluate_single_sample(gen_pdb_path, chain_id, sample_name, model_name, 
                           output_base_dir, verbose=False):
    """处理单个样本"""
    if not os.path.exists(gen_pdb_path):
        return {'sample_name': sample_name, 'status': 'error', 'error': 'file_not_found'}

    try:
        sequence = extract_sequence_from_pdb(gen_pdb_path, chain_id)
        if not sequence:
            return {'sample_name': sample_name, 'status': 'error', 'error': 'seq_extract_failed'}

        # ESMFold 结构目录
        esmfold_dir = output_base_dir / "esmfold_structures" / model_name
        esmfold_dir.mkdir(parents=True, exist_ok=True)
        esmfold_path = esmfold_dir / f"{sample_name}_esmfold.pdb"

        # 如果文件已存在，则跳过 API 调用（断点续传）
        if esmfold_path.exists():
            with open(esmfold_path, 'r') as f:
                esmfold_pdb_str = f.read()
        else:
            esmfold_pdb_str = esmfold_api(sequence, verbose=verbose)
            with open(esmfold_path, 'w') as f:
                f.write(esmfold_pdb_str)

        gen_coords = get_ca_coords(gen_pdb_path, chain_id)
        esmfold_coords = get_ca_coords(str(esmfold_path), 'A')

        rmsd = calculate_rmsd(gen_coords, esmfold_coords)
        
        return {
            'sample_name': sample_name,
            'model': model_name,
            'status': 'success',
            'sequence_length': len(sequence),
            'rmsd': rmsd,
            'rmsd_below_2': rmsd < 2.0 if rmsd is not None else False
        }
    except Exception as e:
        return {'sample_name': sample_name, 'status': 'error', 'error': str(e)}


def run_evaluation(model_name, output_dir, num_samples_override=None, verbose=False):
    """Run full model evaluation"""
    config = MODEL_CONFIGS[model_name]
    gen_dir = config.get('gen_dir', config.get('ref_dir')) # Fallback if gen_dir not set
    num_samples = num_samples_override or config['num_samples']
    
    sample_list = []
    
    # 1. Custom/Generic model with explicit mapping
    if 'mapping' in config:
        mapping = config['mapping']
        for pdb_id, ligand_chain in mapping.items():
            # Check if likely exists (assumed directory structure for generic is pdb_id based or flexible)
            # For now, we trust the mapping and file check happens in evaluate_single_sample
            # But to be safe, check if we can find at least the folder if pattern implies folder
            if "{pdb_id}" in config['gen_pattern']:
                # Heuristic: check if PDB folder exists if pattern starts with folder
                potential_dir = gen_dir / pdb_id
                if potential_dir.exists():
                     sample_list.append((pdb_id, ligand_chain))
            else:
                 sample_list.append((pdb_id, ligand_chain))
        print(f"📖 Loaded {len(sample_list)} samples from mapping for {model_name}")

    # 2. Legacy: ProBayes and PepMimic (read from test.txt hardcoded)
    elif model_name in ['probayes', 'pepmimic']:
        test_file = Path('/home/shiyiming/peptide_design/PepMimic/datasets/LNR/test.txt')
        if test_file.exists():
            with open(test_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        pdb_id = parts[0]
                        ligand_chain = parts[2]
                        target_dir = gen_dir / pdb_id
                        if target_dir.exists():
                            sample_list.append((pdb_id, ligand_chain))
            print(f"📖 Loaded {len(sample_list)} samples from test.txt for {model_name}")
        else:
            print(f"⚠️ test.txt not found: {test_file}")
            
    # 3. Directory Scan (PepBridge, PepFlowww style: PDB_CHAIN folder or file)
    else:
        if not gen_dir.exists():
            print(f"⚠️ Directory does not exist: {gen_dir}")
            return pd.DataFrame()

        for item in gen_dir.iterdir():
            if not item.is_dir(): continue
            parts = item.name.split('_')
            # Expecting PDB_CHAIN format
            if len(parts) >= 2:
                sample_list.append(('_'.join(parts[:-1]), parts[-1]))
        print(f"🔍 Scanned directory and found {len(sample_list)} samples for {model_name}")

    if not sample_list:
        print(f"❌ No valid samples found for {model_name}")
        return pd.DataFrame()
    
    print(f"\n>>> Start Eval: {model_name} (Samples: {len(sample_list)}, Per Sample: {num_samples})")
    
    results = []
    with tqdm(total=len(sample_list) * num_samples, desc=f"Eval {model_name}") as pbar:
        for pdb_id, chain_id in sample_list:
            for i in range(num_samples):
                sample_name = f"{pdb_id}_{chain_id}_gen{i}"
                
                # Construct path
                gen_pattern = config['gen_pattern']
                format_args = {'pdb_id': pdb_id, 'chain_id': chain_id, 'chain': chain_id, 'i': i}
                
                gen_pdb_path = gen_dir / gen_pattern.format(**format_args)
                
                res = evaluate_single_sample(gen_pdb_path, chain_id, sample_name, 
                                           model_name, output_dir, verbose=verbose)
                results.append(res)
                pbar.update(1)

    # Save results
    df = pd.DataFrame(results)
    output_csv = output_dir / f"{model_name}_results.csv"
    df.to_csv(output_csv, index=False)
    
    success_df = df[df['status'] == 'success']
    if len(success_df) > 0:
        rate = success_df['rmsd_below_2'].mean() * 100
        print(f"--- {model_name} Results ---")
        print(f"Success Rate: {len(success_df)}/{len(df)}")
        print(f"Avg RMSD: {success_df['rmsd'].mean():.3f} Å")
        print(f"RMSD < 2Å: {rate:.2f}%")
        print(f"Details saved to: {output_csv}")
    
    return df


# =============================================================================
# 4. 主程序入口
# =============================================================================

def load_chain_mapping(mapping_path):
    mapping = {}
    if os.path.exists(mapping_path):
        print(f"Loading chain mapping from {mapping_path}")
        with open(mapping_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    # format: pdb_id receptor_chain ligand_chain
                    # We store ligand_chain keyed by pdb_id
                    pdb_id, _, pep_chain = parts[0], parts[1], parts[2]
                    mapping[pdb_id] = pep_chain
    return mapping

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESM-Fold scRMSD Evaluation Script")
    
    # Generic mode arguments
    parser.add_argument("--data_dir", type=str, help="Directory containing generated PDBs (for custom model)")
    parser.add_argument("--mapping_file", type=str, help="Path to chain mapping file (pdb_id rec_chain lig_chain)")
    parser.add_argument("--model_name", type=str, default="custom_model", help="Name for the custom model (used in output)")
    
    # Predefined models
    parser.add_argument("--models", nargs="+", default=[], help="Predefined models to evaluate (pepbridge, pepflowww, etc.)")
    
    # Common arguments
    parser.add_argument("--output", type=str, default="esmfold_results", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=40, help="Number of samples per PDB")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    output_path = WORKSPACE / args.output
    output_path.mkdir(exist_ok=True, parents=True)
    
    models_to_run = []

    # 1. Register custom model if data_dir is provided
    if args.data_dir:
        if not args.mapping_file:
            print("❌ Error: --mapping_file is required when using --data_dir")
            sys.exit(1)
            
        mapping = load_chain_mapping(args.mapping_file)
        # Register in MODEL_CONFIGS
        MODEL_CONFIGS[args.model_name] = {
            'gen_dir': Path(args.data_dir),
            'gen_pattern': "{pdb_id}/sample_{i}.pdb", # Assumed pattern for custom
            'num_samples': args.num_samples,
            'mapping': mapping
        }
        models_to_run.append(args.model_name)
    
    # 2. Add requested predefined models
    for m in args.models:
        if m in MODEL_CONFIGS:
            models_to_run.append(m)
        else:
            print(f"⚠️ Warning: Unknown model '{m}'")

    if not models_to_run:
        print("❌ No models specified. Use --data_dir/--mapping_file for custom models, or --models for predefined ones.")
        print(f"Available predefined models: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    all_summaries = []
    
    for model in models_to_run:
        print(f"\nProcessing model: {model}")
        df = run_evaluation(model, output_path, args.num_samples, args.verbose)
        
        if not df.empty:
            success_df = df[df['status'] == 'success']
            if len(success_df) > 0:
                all_summaries.append({
                    'model': model,
                    'avg_rmsd': success_df['rmsd'].mean(),
                    'below_2_rate': success_df['rmsd_below_2'].mean()
                })
            
    if all_summaries:
        print("\n" + "="*40)
        print("Final Summary")
        print("="*40)
        summary_df = pd.DataFrame(all_summaries)
        print(summary_df.to_string(index=False))
        summary_df.to_csv(output_path / "summary_comparison.csv", index=False)


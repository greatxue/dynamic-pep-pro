import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

HIGHER_IS_BETTER = {"AAR", "BSR", "Consistency", "Seq_Div", "Struct_Div", "Co_Div", "DockQ", "TM_score", "SpatialSSR", "SSR", "Seq_ID", "Connectivity"}
LOWER_IS_BETTER = {"RMSD", "Clashes_Inter", "clashes_inter", "Clashes_Intra", "clashes_intra", "clashes_intra_per_1000", "clashes_inter_per_1000", "Rosetta_dG", "rmsd"}
HIGHER_IS_BETTER.add("rmsd_below_2")

def summarize_file(filepath, out_dir):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    sep = ',' if filepath.endswith('.csv') else r'\s+'
    try:
        df = pd.read_csv(filepath, sep=sep)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    # Map ESMFold format
    if 'sample_name' in df.columns and 'Complex' not in df.columns:
        df['Complex'] = df['sample_name'].apply(lambda x: x.split('_')[0] if isinstance(x, str) else x)
        df['Model'] = df['sample_name'].apply(lambda x: x.split('_')[-1] if isinstance(x, str) else x)
    
    # Map Posecheck format
    if 'target' in df.columns and 'Complex' not in df.columns:
        df['Complex'] = df['target']
    if 'model_file' in df.columns and 'Model' not in df.columns:
        df['Model'] = df['model_file']

    if 'clashes_inter' in df.columns and 'num_atoms' in df.columns and 'clashes_inter_per_1000' not in df.columns:
        df['clashes_inter'] = pd.to_numeric(df['clashes_inter'], errors='coerce')
        df['num_atoms'] = pd.to_numeric(df['num_atoms'], errors='coerce')
        df['clashes_inter_per_1000'] = (df['clashes_inter'] / df['num_atoms']) * 1000

    if 'rmsd_below_2' in df.columns:
        df['rmsd_below_2'] = df['rmsd_below_2'].astype(float)
            
    # Check and handle common case if multiple runs concatenated incorrectly (e.g. repeated headers)
    # Filter rows where 'Complex' column equals literal string 'Complex'
    if 'Complex' in df.columns:
        df = df[df['Complex'] != 'Complex']

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print(f"No numeric columns found in {filepath}")
        return

    has_complex = 'Complex' in df.columns
    has_model = 'Model' in df.columns

    base_name = os.path.splitext(os.path.basename(filepath))[0]
    os.makedirs(out_dir, exist_ok=True)
    summary_txt = os.path.join(out_dir, f"{base_name}_summary.txt")
    plot_png = os.path.join(out_dir, f"{base_name}_dist.png")

    num_plots = len(numeric_cols)
    fig, axes = plt.subplots(num_plots, 1, figsize=(8, 4 * num_plots))
    if num_plots == 1: axes = [axes]

    with open(summary_txt, 'w') as f:
        f.write(f"=== Summary for {base_name} ===\n\n")

        for i, col in enumerate(numeric_cols):
            f.write(f"--- Metric: {col} ---\n")
            valid_data = df[col].dropna()
            if valid_data.empty:
                f.write("No valid numeric data.\n\n")
                continue

            f.write("Global Stats (All Samples):\n")
            f.write(f"  Count  : {len(valid_data)}\n")
            f.write(f"  Mean   : {valid_data.mean():.4f}\n")
            f.write(f"  Median : {valid_data.median():.4f}\n")
            f.write(f"  StdDev : {valid_data.std():.4f}\n")
            f.write(f"  Min    : {valid_data.min():.4f}\n")
            f.write(f"  Max    : {valid_data.max():.4f}\n")
            if col == "Rosetta_dG":
                f.write(f"  < 0    : {(valid_data < 0).mean() * 100:.2f}%\n")
            f.write("\n")

            if has_complex and has_model:
                f.write("Per-Complex Aggregation (e.g., Mean of Best per Complex):\n")
                grouped = df.groupby('Complex')[col]
                
                if col in HIGHER_IS_BETTER:
                    best_per_complex = grouped.max().dropna()
                    f.write(f"  (Higher is better -> Taking Maximum per Complex)\n")
                elif col in LOWER_IS_BETTER:
                    best_per_complex = grouped.min().dropna()
                    f.write(f"  (Lower is better -> Taking Minimum per Complex)\n")
                else:
                    # Both
                    f.write(f"  Direction Unknown (Providing both Min and Max per complex)\n")
                    f.write(f"  Mean of Best(Max): {grouped.max().dropna().mean():.4f}\n")
                    f.write(f"  Mean of Best(Min): {grouped.min().dropna().mean():.4f}\n")
                    best_per_complex = None

                if best_per_complex is not None and not best_per_complex.empty:
                    f.write(f"  Mean of Best per Complex : {best_per_complex.mean():.4f}\n")
                    f.write(f"  Std of Best per Complex  : {best_per_complex.std():.4f}\n\n")
            
            ax = axes[i]
            if col == "Rosetta_dG":
                plot_data = valid_data[valid_data <= 1000]
                sns.histplot(plot_data, kde=True, ax=ax, bins=50)
                ax.set_title(f"Distribution of {col} (capped <= 1000) ({base_name})")
                ax.set_xlim(left=min(0, plot_data.min()), right=1000)
            else:
                sns.histplot(valid_data, kde=True, ax=ax, bins=30)
                ax.set_title(f"Distribution of {col} ({base_name})")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(plot_png, dpi=300)
        plt.close()

    print(f"--> Saved summary text: {summary_txt}")
    print(f"--> Saved summary plot: {plot_png}")

    # Generate a focused plot for Rosetta_dG if it exists
    if "Rosetta_dG" in df.columns:
        valid_data = df["Rosetta_dG"].dropna()
        plot_data_focused = valid_data[(valid_data >= -100) & (valid_data <= 100)]
        if not plot_data_focused.empty:
            plt.figure(figsize=(8, 4))
            sns.histplot(plot_data_focused, kde=True, bins=10)
            plt.title(f"Distribution of Rosetta_dG (-100 to 100) ({base_name})")
            plt.xlabel("Rosetta_dG")
            plt.ylabel("Frequency")
            plt.tight_layout()
            focus_png = os.path.join(out_dir, f"{base_name}_dist_focused.png")
            plt.savefig(focus_png, dpi=300)
            plt.close()
            print(f"--> Saved focused plot: {focus_png}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    summarize_file(args.input, args.output_dir)

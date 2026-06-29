import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_combined_eval_curves(ema_span=3, output_filename="multi_model_degradation_gap.png"):
    """
    Loads perfect and imperfect CSV files for 4 specific model configurations,
    computes the risk-adjusted score performance gap for each, applies an EMA filter,
    and graphs all curves together on a single plot with a descriptive legend.
    """
    # Define the 4 models, their readable legend labels, and clear color coding
    models_config = {
        "sac_r2_s2": {"label": "SAC (Unified Reward)", "color": "#1f77b4"},  # Balanced Blue
        "td3_r2_s1": {"label": "TD3 (Unified Reward)", "color": "#ff7f0e"},  # Safety Orange
        "sac_r1_s1": {"label": "SAC (Adaptive Reward)", "color": "#2ca02c"}, # Stability Green
        "td3_r1_s1": {"label": "TD3 (Adaptive Reward)", "color": "#d62728"}  # Warning Red
    }
    
    # Configure styling aesthetics
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
    
    has_data = False
    
    # Iterate through all 4 configurations and plot them on the same axis
    for base_path, cfg in models_config.items():
        file_p = f"artifacts/{base_path}_p.csv"
        file_imp = f"artifacts/{base_path}_imp.csv"
        
        # Verify file pairs exist before attempting downstream pandas operations
        if not os.path.exists(file_p) or not os.path.exists(file_imp):
            print(f"⚠ Warning: Missing evaluation pair for {cfg['label']} (Expected: {file_p} & {file_imp}). Skipping.")
            continue
            
        # Load datasets
        df_p = pd.read_csv(file_p)
        df_imp = pd.read_csv(file_imp)
        
        # Merge on global training steps to align indices
        df = pd.merge(
            df_p[['step', 'risk_adjusted_score']], 
            df_imp[['step', 'risk_adjusted_score']], 
            on='step', 
            suffixes=('_p', '_imp')
        )
        
        if df.empty:
            print(f"⚠ Warning: The merged dataset is empty for {cfg['label']}. Check alignment of 'step' numbers.")
            continue
            
        has_data = True
        
        df['score_diff'] = df['risk_adjusted_score_p'] - df['risk_adjusted_score_imp']
        
        df['score_diff_smooth'] = df['score_diff'].ewm(span=ema_span, adjust=False).mean()
        
        ax.plot(df['step'], df['score_diff'], color=cfg['color'], alpha=0.12, linestyle=':')
        
        ax.plot(
            df['step'], 
            df['score_diff_smooth'], 
            marker='o', 
            linewidth=2.5, 
            markersize=4, 
            color=cfg['color'], 
            label=cfg['label']
        )
        
    if not has_data:
        print("❌ Error: None of the target file pairs were successfully loaded. Verify file names match script keys.")
        plt.close(fig)
        return

    ax.axhline(0, color='#7f7f7f', linestyle='--', alpha=0.8, label="Zero Robustness Baseline")

    ax.set_title("Sim-to-Real Robustness Matrix: Performance Degradation Gap", fontsize=22, fontweight='bold', pad=15)
    ax.set_xlabel("Global Training Steps", fontsize=16, labelpad=8)
    ax.set_ylabel("Risk-Adjusted Score Delta ($\Delta$)", fontsize=20, labelpad=8)
    
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    
    # Legend position and styling parameters
    ax.legend(loc="upper right", frameon=True, facecolor='white', edgecolor='#e2e2e2', framealpha=0.95, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylim(-50, 350)
    
    plt.tight_layout()
    fig.savefig(output_filename, bbox_inches='tight')
    print(f"✔ Success: Multi-model comparison graph exported to '{output_filename}'")
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Duckiebot Multi-Model Performance Comparison Tool")
    parser.add_argument(
        "--span", 
        type=int, 
        default=3, 
        help="Smoothing window span factor for the EMA filter"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="combined_robustness_gap.png", 
        help="Filename string for saving the generated plot"
    )
    
    args = parser.parse_args()
    plot_combined_eval_curves(ema_span=args.span, output_filename=args.output)
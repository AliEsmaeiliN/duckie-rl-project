import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def parse_model_metadata(model_base_path):
    """
    Extracts the algorithm and reward structure automatically from strings like 'sac_r1_s1'.
    """
    basename = os.path.basename(model_base_path).lower()
    
    if "sac" in basename:
        algo_name = "SAC"
    elif "td3" in basename:
        algo_name = "TD3"
    else:
        algo_name = "RL Agent"
        
    if "r1" in basename:
        reward_suffix = "with Adaptive Reward"
    elif "r2" in basename:
        reward_suffix = "with Unified Reward"
    else:
        reward_suffix = ""

    plot_title = f"{algo_name} {reward_suffix} - Performance Degradation Gap".strip()
    
    match = re.match(r"^([^_]+_[^_]+)", os.path.basename(model_base_path))
    run_save_prefix = match.group(1) if match else os.path.basename(model_base_path)
    
    return plot_title, run_save_prefix

def plot_smoothed_eval_curves(model_base_input, ema_span=3):
    """
    Loads perfect and imperfect CSV files, computes the difference between their
    risk_adjusted_score values, applies an EMA filter, and graphs the performance gap.
    """
    file_p = f"{model_base_input}_p.csv"
    file_imp = f"{model_base_input}_imp.csv"
    
    if not os.path.exists(file_p) or not os.path.exists(file_imp):
        print(f"Error: Could not locate evaluation data pairs.\nExpected:\n - {file_p}\n - {file_imp}")
        return

    # Load dataframes
    df_p = pd.read_csv(file_p)
    df_imp = pd.read_csv(file_imp)
    
    df = pd.merge(
        df_p[['step', 'risk_adjusted_score']], 
        df_imp[['step', 'risk_adjusted_score']], 
        on='step', 
        suffixes=('_p', '_imp')
    )
    
    if df.empty:
        print("Error: The merged dataset is empty. Check if the 'step' numbers match up between both CSV files.")
        return

    df['score_diff'] = df['risk_adjusted_score_p'] - df['risk_adjusted_score_imp']
    
    df['score_diff_smooth'] = df['score_diff'].ewm(span=ema_span, adjust=False).mean()

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    plot_title, run_save_prefix = parse_model_metadata(model_base_input)

    ax.plot(df['step'], df['score_diff'], color='#e377c2', alpha=0.2, linestyle='--')
    
    ax.plot(
        df['step'], 
        df['score_diff_smooth'], 
        marker='s', 
        linewidth=2.5, 
        markersize=5, 
        color='#d62728', 
        label="Performance Degradation Gap\n(Perfect Score - Imperfect Score)"
    )

    ax.axhline(0, color='#2ca02c', linestyle=':', alpha=0.7, label="Zero Robustness Baseline")

    ax.set_title(plot_title, fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel("Global Policy Training Steps", fontsize=16, labelpad=8)
    ax.set_ylabel("Risk-Adjusted Score Delta ($\Delta$)", fontsize=16, labelpad=8)
    
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax.legend(loc="upper right", frameon=True, facecolor='white', edgecolor='#e2e2e2', framealpha=0.95, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    ax.set_ylim(-20, 500)

    
    plt.show(block=False)
    plt.pause(0.1) 

    user_choice = input("\n[Plot Active] Hit [ENTER] to save this graph, or type any character to skip: ")
    
    if user_choice.strip() == "":
        output_img_name = f"{run_save_prefix}_risk_adjusted_gap_plot.png"
        fig.savefig(output_img_name, bbox_inches='tight')
        print(f"✔ Success: Graph successfully exported as '{output_img_name}'")
    else:
        print("✘ Skipped: Plot window dismissed without saving.")
        
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Duckiebot Risk-Adjusted Delta Comparison Tool")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Base name string pattern for tracking logs (e.g. 'sac_r1_s1')"
    )
    parser.add_argument(
        "--span", 
        type=int, 
        default=3, 
        help="Smoothing window span factor for the EMA filter"
    )
    
    args = parser.parse_args()
    plot_smoothed_eval_curves(args.model_path, ema_span=args.span)
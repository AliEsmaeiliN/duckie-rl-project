import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def identify_combined_columns(df):
    """
    Identifies the global step column and distinguishes between the CR run
    and the No-CR run columns based on their unique naming identifiers.
    """
    step_col = None
    cr_col = None
    nocr_col = None
    
    # 1. Map the X-axis step column
    for col in df.columns:
        if col.strip().lower() == 'global_step':
            step_col = col
            break
            
    # 2. Map the specific WandB run return columns (ignoring MIN/MAX suffixes)
    for col in df.columns:
        if "charts/episodic_return" in col and not col.endswith("__MIN") and not col.endswith("__MAX"):
            if "sac__final__r2" in col.lower():
                cr_col = col
            elif "sac__unf1__v0" in col.lower():
                nocr_col = col
                
    if not step_col:
        raise KeyError("Could not find a valid 'global_step' column in the CSV file.")
    if not cr_col or not nocr_col:
        raise KeyError(f"Failed to isolate both run streams. Available columns:\n{list(df.columns)}")
        
    return step_col, cr_col, nocr_col

def plot_consolidated_comparison(file_path, ema_span=25):
    """
    Parses a single consolidated CSV log file, applies EMA smoothing,
    and handles interactive visualization/saving rules.
    """
    if not os.path.exists(file_path):
        print(f"Error: The target file '{file_path}' does not exist.")
        return

    print(f"Reading unified log file: {file_path}")
    df = pd.read_csv(file_path).copy()
    
    # Match columns inside the single file
    step_col, cr_col, nocr_col = identify_combined_columns(df)

    print("Applying Exponential Moving Average (EMA) processing...")
    df['cr_smooth'] = df[cr_col].ewm(span=ema_span, adjust=False).mean()
    df['nocr_smooth'] = df[nocr_col].ewm(span=ema_span, adjust=False).mean()

    # Style configuration
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # Graph Run 1: Curriculum Randomization (Emerald Green)
    ax.plot(df[step_col], df[cr_col], color='#2ca02c', alpha=0.10, linewidth=0.8)
    ax.plot(
        df[step_col], 
        df['cr_smooth'], 
        linewidth=2.5, 
        color='#2ca02c', 
        label="With Curriculum Randomization (CR)"
    )

    # Graph Run 2: Baseline No-CR (Slate Gray)
    ax.plot(df[step_col], df[nocr_col], color='#7f7f7f', alpha=0.10, linewidth=0.8)
    ax.plot(
        df[step_col], 
        df['nocr_smooth'], 
        linewidth=2.5, 
        color='#7f7f7f', 
        label="Without Curriculum Randomization (No CR)"
    )

    # Text and Layout Tweaks
    ax.set_title("SAC with Unified Reward (seed: 1): Impact of Curriculum Randomization", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Global Policy Training Steps", fontsize=11, labelpad=8)
    ax.set_ylabel("Episodic Return", fontsize=11, labelpad=8)
    
    # Comma grouping configuration for cleaner numbers
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    
    ax.legend(loc="lower right", frameon=True, facecolor='white', edgecolor='#e2e2e2', framealpha=0.95, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylim(-1000, 1500)
    plt.tight_layout()
    
    # Render view non-blockingly
    plt.show(block=False)
    plt.pause(0.1) 

    # Prompt safe guard save trigger
    user_choice = input("\n[Plot Active] Hit [ENTER] to save this graph, or type any character to skip: ")
    
    if user_choice.strip() == "":
        output_img_name = "sac_r2_curriculum_comparison.png"
        fig.savefig(output_img_name, bbox_inches='tight')
        print(f"✔ Success: Comparison plot successfully saved as '{output_img_name}'")
    else:
        print("✘ Skipped: Dismissed without saving.")
        
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Duckiebot Unified Consolidated Log Comparator")
    parser.add_argument(
        "--file_path", 
        type=str, 
        required=True, 
        help="Path to the single consolidated CSV data file"
    )
    parser.add_argument(
        "--span", 
        type=int, 
        default=25, 
        help="Smoothing window factor (EMA Span) for training lines"
    )
    
    args = parser.parse_args()
    plot_consolidated_comparison(args.file_path, ema_span=args.span)
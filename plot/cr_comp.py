import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_curriculum_evaluation(metric='avg_reward', ema_span=3):
    """
    Loads separate evaluation files for Curriculum and Non-Curriculum runs,
    applies subtle smoothing, and plots metrics along with standard deviation ribbons.
    """
    file_cr = "artifacts/sac_r2_s2_imp.csv"
    file_nocr = "artifacts/sac_r2_s2_nocr_imp.csv"
    
    if not os.path.exists(file_cr) or not os.path.exists(file_nocr):
        print(f"Error: Missing evaluation data logs.\nExpected in workspace:\n - {file_cr}\n - {file_nocr}")
        return

    print(f"Reading evaluation profiles. Metric selected: {metric}")
    df_cr = pd.read_csv(file_cr).copy()
    df_nocr = pd.read_csv(file_nocr).copy()
    
    if metric not in df_cr.columns:
        raise KeyError(f"Metric '{metric}' not found in logs. Choices: {list(df_cr.columns)}")

    df_cr['metric_smooth'] = df_cr[metric].ewm(span=ema_span, adjust=False).mean()
    df_nocr['metric_smooth'] = df_nocr[metric].ewm(span=ema_span, adjust=False).mean()
    
    if 'std_reward' in df_cr.columns:
        df_cr['std_smooth'] = df_cr['std_reward'].ewm(span=ema_span, adjust=False).mean()
        df_nocr['std_smooth'] = df_nocr['std_reward'].ewm(span=ema_span, adjust=False).mean()

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    metric_title_map = {
        "avg_reward": "Average Evaluation Return",
        "risk_adjusted_score": "Risk-Adjusted Score"
    }
    y_axis_label = metric_title_map.get(metric, metric.replace('_', ' ').title())
    plot_title = f"SAC with Unified Reward: Impact of Curriculum Randomization"

    ax.plot(df_cr['step'], df_cr[metric], color='#2ca02c', alpha=0.15, linestyle='--')
    ax.plot(
        df_cr['step'], 
        df_cr['metric_smooth'], 
        marker='o',
        linewidth=2.5, 
        markersize=4,
        color='#2ca02c', 
        label="With Curriculum Randomization (CR)"
    )
    if 'std_smooth' in df_cr.columns:
        ax.fill_between(
            df_cr['step'], 
            df_cr['metric_smooth'] - df_cr['std_smooth'], 
            df_cr['metric_smooth'] + df_cr['std_smooth'], 
            color='#2ca02c', 
            alpha=0.10
        )

    ax.plot(df_nocr['step'], df_nocr[metric], color='#7f7f7f', alpha=0.15, linestyle='--')
    ax.plot(
        df_nocr['step'], 
        df_nocr['metric_smooth'], 
        marker='D',
        linewidth=2.5, 
        markersize=4,
        color='#7f7f7f', 
        label="Without Curriculum Randomization (No CR)"
    )
    if 'std_smooth' in df_nocr.columns:
        ax.fill_between(
            df_nocr['step'], 
            df_nocr['metric_smooth'] - df_nocr['std_smooth'], 
            df_nocr['metric_smooth'] + df_nocr['std_smooth'], 
            color='#7f7f7f', 
            alpha=0.10
        )

    ax.set_title(plot_title, fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("Global Policy Training Steps", fontsize=11, labelpad=8)
    ax.set_ylabel(y_axis_label, fontsize=11, labelpad=8)
    
    
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax.legend(loc="lower right", frameon=True, facecolor='white', edgecolor='#e2e2e2', framealpha=0.95, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.show(block=False)
    plt.pause(0.1) 

    user_choice = input("\n[Plot Active] Hit [ENTER] to save this graph, or type any character to skip: ")
    
    if user_choice.strip() == "":
        output_img_name = f"sac_r2_curriculum_comparison_{metric}.png"
        fig.savefig(output_img_name, bbox_inches='tight')
        print(f"✔ Success: Comparison plot successfully saved as '{output_img_name}'")
    else:
        print("✘ Skipped: Window dismissed without exporting.")
        
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Duckiebot Multi-File Metric Evaluator Comparator")
    parser.add_argument(
        "--metric",
        type=str,
        default="avg_reward",
        choices=["avg_reward", "risk_adjusted_score", "success_rate"],
        help="Target logging parameter row to evaluate on y-axis (default: avg_reward)"
    )
    parser.add_argument(
        "--span", 
        type=int, 
        default=3, 
        help="Smoothing window factor (EMA Span value) for interval evaluations"
    )
    
    args = parser.parse_args()
    plot_curriculum_evaluation(metric=args.metric, ema_span=args.span)
#By Matteo Cederle
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from collections import defaultdict
import json


colors = {
    "SAC" : "#7678ed",
    "TD3" : "#f7b801",
}


markers = {
    "SAC" : "d",
    "TD3" : "^",
}


def infer_algo(paths):
    results = []

    for path_str in paths:
        p = Path(path_str)

        if "SAC" in str(p):
            results.append("SAC")
        else:
            results.append("TD3")

    return results


def group_paths_by_env(paths, labels):
    """Group (path, label) pairs by inferred environment name."""
    from collections import defaultdict
    groups = defaultdict(list)
    for path in paths:
        base = os.path.basename(os.path.normpath(path))
        env = base.split("_")[0]
        
        groups[env].append(path)
    return groups  # dict: env_name -> [(path, label), ...]


def exponential_smoothing(x, alpha=0.05):
    """
    Exponential moving average smoothing (standard in RL).
    """
    smoothed = np.zeros_like(x, dtype=np.float64)
    smoothed[0] = x[0]
    for t in range(1, len(x)):
        smoothed[t] = alpha * x[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


def load_algorithm_results(directory, smoothing_alpha=0.1, max_seeds=None):
    """
    Load seed_*.npz files from a directory and compute mean ± stderr.
    """
    seed_files = sorted(glob.glob(os.path.join(directory, "*.npy")))
    if len(seed_files) == 0:
        raise ValueError(f"No seed_*.npz files found in {directory}")

    if max_seeds is not None:
        seed_files = seed_files[:max_seeds]

    all_returns = []

    for f in seed_files:
        returns = np.load(f)

        returns_ = exponential_smoothing(returns, alpha=smoothing_alpha)
        all_returns.append(returns_)

    min_len = min(len(r) for r in all_returns)
    all_returns = np.array([r[:min_len] for r in all_returns])

    mean = np.mean(all_returns, axis=0)
    stderr = np.std(all_returns, axis=0, ddof=1) / np.sqrt(all_returns.shape[0])

    return mean, stderr

def format_x_axis(ax, max_step, tick_base, tick_power):
    """
    Format x-axis ticks as multiples of (base * 10^power)
    and annotate axis with ×10^power.
    """
    scale = 10 ** tick_power
    tick_interval = tick_base * scale

    ticks = np.arange(
        tick_interval,
        max_step + tick_interval,
        tick_interval,
    )

    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(t / scale)) for t in ticks], fontsize=16)

    ax.set_xlabel(f"Environment Steps (×$10^{tick_power})$", fontsize=16)


def plot_grid(
    groups,           # dict from group_paths_by_env
    smoothing_alpha,
    max_seeds,
    eval_frequency,
    output_path,
    x_tick_base,
    x_tick_power,
    n_cols=2,         # <-- flexibility
    shared_legend=True,
    figsize_per_cell=(6, 4)
):
    env_names = list(groups.keys())
    n_envs = len(env_names)
    n_cols = min(n_cols, n_envs)
    n_rows = (n_envs + n_cols - 1) // n_cols  # ceil division

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
        squeeze=False,
    )

    legend_handles, legend_labels = [], []

    all_alg = []

    for idx, env_name in enumerate(env_names):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        algos_ = infer_algo(groups[env_name])

        algos = []
        colors_plot = []
        markers_plot = []
        for algo in algos_:
            colors_plot.append(colors[algo])
            markers_plot.append(markers[algo])
            algos.append(algo)

        for i, directory in enumerate(groups[env_name]):

            mean, stderr = load_algorithm_results(directory, smoothing_alpha, max_seeds)
            steps = (np.arange(len(mean)) + 1) * eval_frequency

            idx_ = np.linspace(0, len(steps) - 1, 11, dtype=int)

            # Ensure last point is included
            if idx_[-1] != len(steps) - 1:
                idx_ = np.append(idx_, len(steps) - 1)

            line, = ax.plot(steps, mean, label=algos[i], color=colors_plot[i], marker=markers_plot[i], ls='-', markevery=idx_)
            ax.fill_between(steps, mean - stderr, mean + stderr, alpha=0.2, color=colors_plot[i])

            # Collect legend entries once (first env only)
            if shared_legend:
                if algos[i] not in all_alg:
                    legend_handles.append(line)
                    legend_labels.append(algos[i])
                    all_alg.append(algos[i])


        format_x_axis(ax, steps[-1], x_tick_base, x_tick_power)
        ax.set_title(env_name, fontsize=18)
        if col == 0:
            ax.set_ylabel("Average Episodic Return", fontsize=18)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
        ax.tick_params(axis='y', labelsize=16 - 2)
        ax.grid(True, alpha=0.3)
        if not shared_legend:
            ax.legend()

    # Hide unused subplots
    for idx in range(n_envs, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    if shared_legend:
        legend1 = fig.legend(
            legend_handles, legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.12), # -0.05
            ncol=len(legend_handles),
            frameon=True,
            fontsize=22,
            markerscale=2.0,  # scales markers relative to their size in the plot
        )
        fig.add_artist(legend1)  # needed to keep it when adding a second legend
        fig.tight_layout(rect=[0, 0.05, 1, 1])  # leave room for legend

    else:
        fig.tight_layout()

    # os.makedirs("plots/batch", exist_ok=True)
    fig.savefig(f"{output_path}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot RL evaluation returns across multiple seeds and algorithms."
    )

    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="List of directories, one per algorithm.",
    )

    parser.add_argument(
        "--labels",
        nargs="+",
        required=False,
        help="Labels corresponding to each directory.",
    )

    parser.add_argument(
        "--smoothing_alpha",
        type=float,
        default=0.1,
        help="EMA smoothing coefficient (default: 0.1).",
    )

    parser.add_argument(
        "--max_seeds",
        type=int,
        default=10,
        help="Maximum number of seeds to load per algorithm.",
    )

    parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="Path to save the plot (e.g., plot.png or plot.pdf).",
    )
    
    parser.add_argument(
    "--eval_frequency",
    type=int,
    default=20_000,
    help="Number of environment steps between evaluations (no eval at step 0).",
    )

    parser.add_argument(
    "--x_tick_base",
    type=int,
    default=2,
    help="Base tick spacing before scaling (default: 1).",
    )

    parser.add_argument(
        "--x_tick_power",
        type=int,
        default=5,
        help="Power of 10 for x-axis scaling (default: 5 → x10^5).",
    )

    parser.add_argument("--cols", type=int, default=1, help="Number of columns in the grid.")
    parser.add_argument("--shared_legend", action="store_true", help="Single legend for all subplots.")
    parser.add_argument("--figsize", nargs=2, type=float, default=[6, 4], metavar=("W", "H"),
                        help="Figure size per cell (width height).")

    return parser.parse_args()


def main():
    args = parse_args()

    groups = group_paths_by_env(args.paths, args.labels)

    plot_grid(
        groups=groups,
        smoothing_alpha=args.smoothing_alpha,
        max_seeds=args.max_seeds,
        eval_frequency=args.eval_frequency,
        output_path=args.output,
        x_tick_base=args.x_tick_base,
        x_tick_power=args.x_tick_power,
        n_cols=args.cols,
        shared_legend=args.shared_legend,
        figsize_per_cell=tuple(args.figsize)
    )


if __name__ == "__main__":
    main()

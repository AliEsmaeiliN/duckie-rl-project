"""
Sim-to-Real Robust Feature Comparator
======================================
Replaces fragile intra-vector min-max normalization with four scale-invariant metrics:

  1. Cosine Similarity       -- directional alignment, scale-invariant
  2. Spearman Rank Corr.     -- rank-order fidelity, outlier-robust
  3. Linear CKA              -- representation alignment across multiple samples
  4. Z-score L2 Distance     -- magnitude distance after standardization

All metrics are computed per-scenario and aggregated into publication-ready plots.
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import argparse
import yaml
import os
from scipy.stats import spearmanr
from itertools import combinations

from models import SACActor, TD3Actor
from utils.rl_env import DuckieOvalEnv



def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Scale-invariant directional alignment.
    Returns 1.0 for identical directions, 0.0 for orthogonal, -1.0 for opposite.
    Perfect for comparing latent geometry regardless of activation magnitude.
    """
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)


def spearman_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """
    Rank-order correlation. Robust to outlier activations and monotonic
    transforms — tells you if the *relative ordering* of feature importance
    is preserved across domains.
    """
    corr, _ = spearmanr(a, b)
    return float(corr)


def zscore_l2(a: np.ndarray, b: np.ndarray) -> float:
    """
    L2 distance after independent z-score standardization.
    Unlike min-max, z-score is not distorted by outlier activations or
    dead (near-zero) feature channels. Lower = more similar.
    Normalized by sqrt(D) so it's comparable across latent sizes.
    """
    def zscore(v):
        std = v.std() + 1e-8
        return (v - v.mean()) / std

    diff = zscore(a) - zscore(b)
    # Divide by sqrt(D) to get a scale-free [0, ~2] range in practice
    return float(np.linalg.norm(diff) / np.sqrt(len(a)))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear Centered Kernel Alignment (Kornblith et al., 2019).
    Measures representational similarity across a *set* of samples,
    invariant to orthogonal transforms and isotropic scaling.

    X, Y: (n_samples, n_features) — pass all scenarios stacked together,
          or call per-scenario with repeated single vectors (degenerate but usable).

    Returns value in [0, 1]: 1 = identical representations.
    """
    def center(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    # Gram matrices
    K = X @ X.T
    L = Y @ Y.T
    K_c = center(K)
    L_c = center(L)

    hsic = np.sum(K_c * L_c)
    norm = np.sqrt(np.sum(K_c * K_c) * np.sum(L_c * L_c)) + 1e-10
    return float(hsic / norm)



class Sim2RealComparator:
    def __init__(self, model_name, calib_path, save_dir, device="cpu", grayscale=True):
        self.device = torch.device(device)
        self.grayscale = grayscale
        self.tilt_strength = 0.0006
        self.model_name = model_name
        self.save_folder = save_dir

        self.is_downscaled = "ds" in model_name.lower()
        self.target_size = (42, 42) if self.is_downscaled else (84, 84)
        print(f"[{self.model_name}] Downscaled: {self.is_downscaled} | Resolution: {self.target_size}")

        self.model_path = os.path.join(
            os.path.dirname(os.path.dirname(self.save_folder)),
            f"{model_name}.cleanrl_model"
        )

        dummy_env = DuckieOvalEnv.create_wrapped(
            "dummy", grayscale=self.grayscale, downscaled=self.is_downscaled
        )

        if "td3" in model_name.lower():
            self.actor = TD3Actor(dummy_env, downscaled=self.is_downscaled).to(self.device)
        else:
            self.actor = SACActor(dummy_env, downscaled=self.is_downscaled).to(self.device)

        checkpoint = torch.load(
            os.path.expanduser(self.model_path),
            map_location=self.device,
            weights_only=True
        )
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor.eval()

        self.outputs = {}
        self._register_hooks()
        self._load_calibration(calib_path)
        self.maps_built = False

    # ── Calibration & Preprocessing ──────────────────────────────────────────

    def _load_calibration(self, calib_path):
        calib_path = os.path.expanduser(calib_path)
        with open(calib_path, 'r') as f:
            calib_data = yaml.safe_load(f)

        cam_mat = calib_data.get('camera_matrix', {})
        if isinstance(cam_mat, dict) and 'data' in cam_mat:
            cam_mat = cam_mat['data']
        dist_coefs = calib_data.get('distortion_coefficients',
                                    calib_data.get('distortion_coefs', {}))
        if isinstance(dist_coefs, dict) and 'data' in dist_coefs:
            dist_coefs = dist_coefs['data']

        self.camera_matrix = np.array(cam_mat, dtype=np.float32).reshape(3, 3)
        self.distortion_coefs = np.array(dist_coefs, dtype=np.float32)

    def _compute_tilt_homography(self, w, h):
        cx = w / 2
        shift = self.tilt_strength * w * h
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [cx - (cx - 0) * (1 - self.tilt_strength * h), shift],
            [cx + (w - cx) * (1 - self.tilt_strength * h), shift],
            [w, h], [0, h]
        ])
        return cv2.getPerspectiveTransform(src, dst)

    def _build_real_maps(self, w, h):
        new_cam, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.distortion_coefs, (w, h), 0, (w, h)
        )
        map_x, map_y = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.distortion_coefs, None, new_cam,
            (w, h), cv2.CV_32FC1
        )
        H = self._compute_tilt_homography(w, h)
        self.map_x = cv2.warpPerspective(map_x, H, (w, h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT)
        self.map_y = cv2.warpPerspective(map_y, H, (w, h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT)
        self.maps_built = True

    def preprocess_image(self, img_path, is_real_world=False):
        img = cv2.imread(os.path.expanduser(img_path))
        if img is None:
            raise FileNotFoundError(f"Could not load: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        if is_real_world:
            if not self.maps_built:
                self._build_real_maps(w, h)
            img = cv2.remap(img, self.map_x, self.map_y, cv2.INTER_LINEAR)
            top = int(h * 0.4)
            left, right = int(w * 0.2), int(w * 0.8)
            img = img[top:h, left:right]
            h1, w1 = img.shape[:2]
            img = img[h1 // 3:h1, 0:w1]
        else:
            img = img[h // 3:h, 0:w]

        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)

        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, 0)
        else:
            img = img.transpose(2, 0, 1)

        stacked = np.tile(img, (4, 1, 1))
        tensor = torch.Tensor(stacked).unsqueeze(0).to(self.device)
        return tensor

    # ── Hook registration ─────────────────────────────────────────────────────

    def _get_hook(self, name):
        def hook(model, input, output):
            self.outputs[name] = output.detach()
        return hook

    def _register_hooks(self):
        seq = self.actor.encoder.main
        seq[1].register_forward_hook(self._get_hook('Conv1'))
        seq[-1].register_forward_hook(self._get_hook('Latent'))

    # ── Core extraction ───────────────────────────────────────────────────────

    def extract_latents(self, sim_folder: str, real_folder: str) -> dict:
        """
        Returns raw (un-normalized) latent vectors for all scenarios.
        Shape: { 'sim': (4, D), 'real': (4, D) }
        """
        suffixes = ['cr', 'cl', 'ss', 'sl']
        sim_latents, real_latents = [], []

        for sfx in suffixes:
            for prefix, is_real, store in [
                ('sim', False, sim_latents),
                ('real', True, real_latents)
            ]:
                folder = sim_folder if prefix == 'sim' else real_folder
                path = os.path.join(folder, f"{prefix}_{sfx}.png")
                tensor = self.preprocess_image(path, is_real_world=is_real)
                with torch.no_grad():
                    _ = self.actor(tensor)
                store.append(self.outputs['Latent'][0].cpu().numpy())

        return {
            'sim': np.stack(sim_latents),    # (4, D)
            'real': np.stack(real_latents),  # (4, D)
        }

    def compute_metrics(self, sim_folder: str, real_folder: str) -> list[dict]:
        """
        Computes all four metrics per scenario + global CKA across all scenarios.
        Returns a list of dicts suitable for pd.DataFrame.
        """
        suffixes    = ['cr', 'cl', 'ss', 'sl']
        scenario_names = {
            'cr': 'Right Curve',
            'cl': 'Left Curve',
            'ss': 'Short Straight',
            'sl': 'Long Straight'
        }

        data = self.extract_latents(sim_folder, real_folder)
        sim_vecs  = data['sim']   # (4, D)
        real_vecs = data['real']  # (4, D)

        rows = []

        # ── Per-scenario scalar metrics ──
        for i, sfx in enumerate(suffixes):
            s, r = sim_vecs[i], real_vecs[i]
            rows.append({
                "Model":    self.model_name,
                "Scenario": scenario_names[sfx],
                "Metric":   "Cosine Similarity",
                "Value":    cosine_similarity(s, r),
                "Higher":   True,    # higher = more similar
            })
            rows.append({
                "Model":    self.model_name,
                "Scenario": scenario_names[sfx],
                "Metric":   "Spearman ρ",
                "Value":    spearman_correlation(s, r),
                "Higher":   True,
            })
            rows.append({
                "Model":    self.model_name,
                "Scenario": scenario_names[sfx],
                "Metric":   "Z-score L2",
                "Value":    zscore_l2(s, r),
                "Higher":   False,   # lower = more similar
            })

        # ── Global CKA across all 4 scenarios ──
        # Uses all samples jointly — this is where CKA shines
        cka_val = linear_cka(sim_vecs, real_vecs)
        for sfx in suffixes:
            rows.append({
                "Model":    self.model_name,
                "Scenario": scenario_names[sfx],
                "Metric":   "Linear CKA",
                "Value":    cka_val,     # same value for all — it's a global measure
                "Higher":   True,
            })

        # ── Per-feature shift profile (for sparsity plot) ──
        # Use z-score per-feature so dead neurons don't distort the profile
        def zscore(v):
            return (v - v.mean()) / (v.std() + 1e-8)

        feature_rows = []
        for i, sfx in enumerate(suffixes):
            s, r = sim_vecs[i], real_vecs[i]
            per_feat_diff = np.abs(zscore(s) - zscore(r))  # (D,)
            for feat_idx, val in enumerate(per_feat_diff):
                feature_rows.append({
                    "Model":     self.model_name,
                    "Scenario":  scenario_names[sfx],
                    "Feature":   feat_idx,
                    "ZScore_L2": float(val),
                })

        return rows, feature_rows


# ─────────────────────────────────────────────────────────────────────────────
#  Publication Plots
# ─────────────────────────────────────────────────────────────────────────────

SCENARIO_ORDER = ['Long Straight', 'Short Straight', 'Left Curve', 'Right Curve']
METRIC_ORDER   = ['Cosine Similarity', 'Spearman ρ', 'Linear CKA', 'Z-score L2']

PALETTE = {
    "Cosine Similarity": "#4C8EDA",
    "Spearman ρ":        "#2CA02C",
    "Linear CKA":        "#9467BD",
    "Z-score L2":        "#D62728",
}


def _style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
    })


def plot_metric_heatmap(df: pd.DataFrame, save_folder: str):
    """
    Scenario × Metric heatmap per model.
    Color encodes value; direction-aware so green = good for all metrics.
    """
    _style()
    models = df['Model'].unique()
    fig, axes = plt.subplots(1, len(models),
                             figsize=(5.5 * len(models), 4.5),
                             squeeze=False)

    for ax, model in zip(axes[0], models):
        sub = df[df['Model'] == model].copy()

        # Flip Z-score L2 so that high = good everywhere (for a unified colormap)
        sub['PlotValue'] = sub.apply(
            lambda row: -row['Value'] if not row['Higher'] else row['Value'],
            axis=1
        )
        pivot = (sub
                 .groupby(['Scenario', 'Metric'])['PlotValue']
                 .mean()
                 .unstack('Metric')
                 .reindex(index=SCENARIO_ORDER,
                          columns=[m for m in METRIC_ORDER if m in sub['Metric'].unique()]))

        sns.heatmap(
            pivot, ax=ax,
            cmap='RdYlGn', center=0.0,
            annot=True, fmt=".3f", annot_kws={"size": 9},
            linewidths=0.5, cbar_kws={'label': 'Score (Z-L2 negated; ↑ = more sim-real aligned)'}
        )
        ax.set_title(f"{model}", weight='bold', pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis='x', rotation=25)

    fig.suptitle(
        "Sim-to-Real Alignment Heatmap by Scenario & Metric",
        fontsize=14, weight='bold', y=1.02
    )
    plt.tight_layout()
    path = os.path.join(save_folder, "metric_heatmap.pdf")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Saved] {path}")


def plot_radar(df: pd.DataFrame, save_folder: str):
    """
    Radar / spider plot: each spoke = one scenario, each ring = one model.
    Separate radar per metric so scales don't clash.
    Cosine / Spearman / CKA are similarity (↑ better).
    Z-score L2 is distance (↓ better), inverted for radar.
    """
    _style()
    metrics  = [m for m in METRIC_ORDER if m in df['Metric'].unique()]
    models   = list(df['Model'].unique())
    scenarios = SCENARIO_ORDER
    N = len(scenarios)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    fig, axes = plt.subplots(
        1, len(metrics),
        figsize=(4.8 * len(metrics), 4.8),
        subplot_kw=dict(polar=True)
    )
    if len(metrics) == 1:
        axes = [axes]

    colors = sns.color_palette("tab10", len(models))

    for ax, metric in zip(axes, metrics):
        for model, color in zip(models, colors):
            sub = df[(df['Model'] == model) & (df['Metric'] == metric)]
            vals = []
            for sc in scenarios:
                row = sub[sub['Scenario'] == sc]
                v = float(row['Value'].mean()) if len(row) else 0.0
                # Invert Z-score L2 so radar: outward = better always
                if metric == 'Z-score L2':
                    # Map: 0 → 1, large → 0  (soft inversion with clip)
                    v = 1.0 / (1.0 + v)
                vals.append(v)
            vals += vals[:1]
            ax.plot(angles, vals, color=color, linewidth=2.0, label=model)
            ax.fill(angles, vals, color=color, alpha=0.12)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(scenarios, size=8.5)
        label = metric if metric != 'Z-score L2' else 'Z-score L2 (inverted ↑ = less drift)'
        ax.set_title(label, weight='bold', pad=14, size=10)
        ax.set_ylim(0, 1)
        ax.yaxis.set_tick_params(labelsize=7)

    axes[0].legend(loc='upper left', bbox_to_anchor=(-0.35, 1.15),
                   frameon=True, fontsize=9, title="Model")
    fig.suptitle("Sim-to-Real Alignment Radar by Metric & Scenario",
                 fontsize=13, weight='bold', y=1.04)
    plt.tight_layout()
    path = os.path.join(save_folder, "metric_radar.pdf")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Saved] {path}")


def plot_grouped_bars(df: pd.DataFrame, save_folder: str):
    """
    Grouped bar chart: X = scenario, hue = model, one panel per metric.
    Clean, direct comparison. Error bars show scenario-to-scenario spread
    (meaningful if you run multiple seeds; with 1 seed they're absent).
    """
    _style()
    metrics = [m for m in METRIC_ORDER if m in df['Metric'].unique()]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=False)
    axes = axes.flat

    for ax, metric in zip(axes, metrics):
        sub = df[df['Metric'] == metric].copy()
        # Aggregate (mean across any repeated seeds)
        agg = (sub.groupby(['Model', 'Scenario'])['Value']
               .agg(['mean', 'std'])
               .reset_index()
               .rename(columns={'mean': 'Mean', 'std': 'Std'}))
        agg['Scenario'] = pd.Categorical(
            agg['Scenario'], categories=SCENARIO_ORDER, ordered=True
        )
        agg = agg.sort_values('Scenario')

        models = agg['Model'].unique()
        x      = np.arange(len(SCENARIO_ORDER))
        width  = 0.8 / len(models)
        colors = sns.color_palette("tab10", len(models))

        for k, (model, color) in enumerate(zip(models, colors)):
            m_data = agg[agg['Model'] == model]
            means  = [m_data[m_data['Scenario'] == sc]['Mean'].values
                      for sc in SCENARIO_ORDER]
            stds   = [m_data[m_data['Scenario'] == sc]['Std'].values
                      for sc in SCENARIO_ORDER]
            means  = [v[0] if len(v) else 0 for v in means]
            stds   = [v[0] if len(v) else 0 for v in stds]
            offset = (k - (len(models) - 1) / 2) * width
            ax.bar(x + offset, means, width=width * 0.9,
                   label=model, color=color, alpha=0.85,
                   yerr=stds, capsize=3, error_kw={'linewidth': 1.0})

        is_similarity = metric != 'Z-score L2'
        ax.axhline(1.0 if is_similarity else 0.0,
                   color='black', linewidth=0.8, linestyle='--', alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIO_ORDER, rotation=15, ha='right')
        ax.set_title(metric, weight='bold')
        ax.set_ylabel("Value")
        ax.legend(fontsize=8, title="Model", frameon=True)
        ax.grid(axis='y', alpha=0.4)

    # Hide unused panel if metrics < 4
    for ax in list(axes)[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle(
        "Sim-to-Real Feature Alignment: All Metrics × Scenarios",
        fontsize=14, weight='bold'
    )
    plt.tight_layout()
    path = os.path.join(save_folder, "grouped_bars.pdf")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Saved] {path}")


def plot_sparsity_zscore(feat_df: pd.DataFrame, save_folder: str):
    """
    Replaces the old min-max sparsity curve.
    Uses z-score per-feature diff (robust to dead neurons).
    Log-scale, ranked descending — exactly your original intent, but correct.
    """
    _style()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    models = feat_df['Model'].unique()
    colors = sns.color_palette("tab10", len(models))

    for model, color in zip(models, colors):
        sub = feat_df[feat_df['Model'] == model]
        # Mean across scenarios per feature
        mean_per_feat = sub.groupby('Feature')['ZScore_L2'].mean()
        ranked = mean_per_feat.sort_values(ascending=False).values
        ax.plot(range(len(ranked)), ranked,
                label=model, color=color, linewidth=2.0, alpha=0.9)

    ax.set_yscale('log')
    ax.axvline(x=25, color='gray', linestyle=':', linewidth=1.2)
    ax.text(27, ax.get_ylim()[1] * 0.3,
            "Top-10% Volatile Zone\n(Feat. 0–25)",
            fontsize=9, color='dimgray', weight='semibold')

    ax.set_title(
        "Z-score Feature Drift Profile — Ranked by Mean Domain Shift",
        weight='bold', pad=14
    )
    ax.set_xlabel("Latent Feature Rank (Descending Mean |Δz-score|)")
    ax.set_ylabel("Mean |Δz-score| (Log Scale)")
    ax.set_xlim(0, feat_df['Feature'].max())
    ax.legend(title="Model", frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()

    path = os.path.join(save_folder, "sparsity_zscore.pdf")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Saved] {path}")


def print_summary_table(df: pd.DataFrame):
    """Prints a readable per-model, per-metric summary to stdout."""
    print("\n" + "=" * 60)
    print("  SIM-TO-REAL ALIGNMENT SUMMARY")
    print("=" * 60)
    summary = (df.groupby(['Model', 'Metric'])['Value']
               .agg(['mean', 'std', 'min', 'max'])
               .round(4))
    print(summary.to_string())
    print("=" * 60)

    print("\n  CKA is a global metric (same value across scenarios).")
    print("  Cosine / Spearman / CKA:  ↑ higher = better alignment")
    print("  Z-score L2:               ↓ lower  = better alignment\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sim-to-Real Robust Feature Comparator"
    )
    parser.add_argument("--models",  nargs='+', required=True,
                        help="Model names (must match .cleanrl_model files)")
    parser.add_argument("--device",  default="cuda")
    args = parser.parse_args()

    CALIB_PATH  = "artifacts/duckie_calib_data.txt"
    SIM_FOLDER  = "screenshots/sim"
    REAL_FOLDER = "screenshots/realbot"
    OUTPUT_DIR  = os.path.expanduser("~/workspace/rl_models/img/feature_extraction")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_metric_rows  = []
    all_feature_rows = []

    for model_name in args.models:
        print(f"\n{'='*55}\n  Processing: {model_name}\n{'='*55}")
        comp = Sim2RealComparator(
            model_name=model_name,
            calib_path=CALIB_PATH,
            save_dir=OUTPUT_DIR,
            device=args.device,
            grayscale=True
        )
        metric_rows, feat_rows = comp.compute_metrics(SIM_FOLDER, REAL_FOLDER)
        all_metric_rows.extend(metric_rows)
        all_feature_rows.extend(feat_rows)

    df      = pd.DataFrame(all_metric_rows)
    feat_df = pd.DataFrame(all_feature_rows)

    # Save raw data for your own analysis / thesis appendix
    df.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)
    feat_df.to_csv(os.path.join(OUTPUT_DIR, "feature_drift.csv"), index=False)

    print_summary_table(df)

    print("\nGenerating publication plots...")
    plot_metric_heatmap(df, OUTPUT_DIR)
    plot_radar(df, OUTPUT_DIR)
    plot_grouped_bars(df, OUTPUT_DIR)
    plot_sparsity_zscore(feat_df, OUTPUT_DIR)

    print("\nDone. All outputs written to:", OUTPUT_DIR)
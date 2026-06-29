import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import yaml
import os

from models import SACActor, TD3Actor
from utils.rl_env import DuckieOvalEnv 

class Sim2RealComparator:
    def __init__(self, model_name, calib_path, save_dir, device="cpu", grayscale=True):
        self.device = torch.device(device)
        self.grayscale = grayscale
        self.tilt_strength = 0.0006
        self.model_name = model_name
        self.save_folder = save_dir
            
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(self.save_folder)), f"{model_name}.cleanrl_model")
        
        dummy_env = DuckieOvalEnv.create_wrapped("dummy", grayscale=self.grayscale)
        
        if "td3" in model_name.lower():
            print(f"Instantiating TD3Actor for: {model_name}")
            self.actor = TD3Actor(dummy_env).to(self.device)
        else:
            print(f"Instantiating SACActor for: {model_name}")
            self.actor = SACActor(dummy_env).to(self.device)
        
        print(f"Loading model weights: {self.model_path}")
        checkpoint = torch.load(os.path.expanduser(self.model_path), map_location=self.device, weights_only=True)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor.eval()

        self.outputs = {}
        self._register_hooks()
        self._load_calibration(calib_path)
        self.maps_built = False

    def _load_calibration(self, calib_path):
        calib_path = os.path.expanduser(calib_path)
        with open(calib_path, 'r') as f:
            calib_data = yaml.safe_load(f)
        cam_mat = calib_data.get('camera_matrix', {}).get('data', calib_data.get('camera_matrix', {}))
        dist_coefs = calib_data.get('distortion_coefficients', {}).get('data', calib_data.get('distortion_coefs', {}))
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
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.distortion_coefs, (w, h), 0, (w, h))
        map_x, map_y = cv2.initUndistortRectifyMap(self.camera_matrix, self.distortion_coefs, None, new_camera_matrix, (w, h), cv2.CV_32FC1)
        H_tilt = self._compute_tilt_homography(w, h)
        self.map_x = cv2.warpPerspective(map_x, H_tilt, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        self.map_y = cv2.warpPerspective(map_y, H_tilt, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        self.maps_built = True

    def _get_hook(self, name):
        def hook(model, input, output):
            self.outputs[name] = output.detach()
        return hook

    def _register_hooks(self):
        encoder_seq = self.actor.encoder.main
        self.hook_latent = encoder_seq[-1].register_forward_hook(self._get_hook('Latent'))

    def preprocess_image(self, img_path, is_real_world=False):
        img_path = os.path.expanduser(img_path)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        if is_real_world:
            if not self.maps_built: self._build_real_maps(w, h)
            img = cv2.remap(img, self.map_x, self.map_y, cv2.INTER_LINEAR)
            img = img[int(h*0.4):h, int(w*0.2):int(w*0.8)]
            h1, w1 = img.shape[:2]
            img = img[int(h1/3):h1, 0:w1]
        else:
            img = img[int(h/3):h, 0:w]

        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_LINEAR) 
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, axis=0)
        else:
            img = img.transpose(2, 0, 1)

        stacked_img = np.tile(img, (4, 1, 1))
        return torch.Tensor(stacked_img).unsqueeze(0).to(self.device)

    def extract_features(self, sim_folder, real_folder):
        images_suffix = ['cr', 'cl', 'ss', 'sl']
        scenario_names = ['Right Curve', 'Left Curve', 'Short Straight', 'Long Straight']
        model_data = []

        for i in range(4):
            sim_path = os.path.join(sim_folder, f"sim_{images_suffix[i]}.png")
            real_path = os.path.join(real_folder, f"real_{images_suffix[i]}.png")
            
            sim_tensor = self.preprocess_image(sim_path, is_real_world=False)
            with torch.no_grad(): self.actor(sim_tensor)
            sim_latent = self.outputs['Latent'][0].cpu().numpy()

            real_tensor = self.preprocess_image(real_path, is_real_world=True)
            with torch.no_grad(): self.actor(real_tensor)
            real_latent = self.outputs['Latent'][0].cpu().numpy()

            # Original asymmetric normalization formula
            epsilon = 1e-5
            rel_diff = np.abs(sim_latent - real_latent) / (np.abs(sim_latent) + epsilon)
            
            for feature_idx, shift_val in enumerate(rel_diff):
                model_data.append({
                    "Model": self.model_name,
                    "Scenario": scenario_names[i],
                    "Latent Dimension": feature_idx,
                    "Relative Shift": float(shift_val)
                })

        return model_data


# --- Peak-Value Normalized Bar Plot Grid ---

def plot_thesis_bar_grid(df, save_folder):
    """
    Generates a 2x2 grid of bar plots where each model's relative shift metric is
    divided by its maximum value, cleanly bounding all plots inside the [0, 1] range.
    """
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })

    scenario_order = ['Long Straight', 'Short Straight', 'Left Curve', 'Right Curve']
    
    # Compute descriptive summary metrics on the peak-normalized data
    summary_stats = df.groupby(['Model', 'Scenario'])['Relative Shift'].agg(
        mean='mean',
        std='std',
        p95=lambda x: np.percentile(x, 95)
    ).reset_index()
    
    summary_stats['Scenario'] = pd.Categorical(summary_stats['Scenario'], categories=scenario_order, ordered=True)

    models = summary_stats['Model'].unique()
    fig, axs = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    axs = axs.flatten()

    palette = sns.color_palette("muted", len(scenario_order))
    scen_to_color = dict(zip(scenario_order, palette))

    for idx, model in enumerate(models):
        ax = axs[idx]
        model_data = summary_stats[summary_stats['Model'] == model].sort_values('Scenario')
        
        scenarios = model_data['Scenario'].tolist()
        means = model_data['mean'].tolist()
        stds = model_data['std'].tolist()
        p95s = model_data['p95'].tolist()
        colors = [scen_to_color[s] for s in scenarios]
        
        # Plot Peak-Normalized Means and Standard Deviations
        bars = ax.bar(
            scenarios, means, yerr=stds,
            color=colors, alpha=0.85, edgecolor='black', linewidth=0.8,
            capsize=5, error_kw=dict(elinewidth=1.2, ecolor='dimgray', alpha=0.7)
        )
        
        # Overlay the 95th percentile metrics clearly at the top of the bars
        for bar, p95_val in zip(bars, p95s):
            curr_scen = scenarios[bars.index(bar)]
            curr_std = stds[scenarios.index(curr_scen)]
            y_pos = max(bar.get_height() + curr_std, bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width()/2.0, min(y_pos + 0.02, 1.05), 
                f"95%ile:\n{p95_val:.2f}", 
                ha='center', va='bottom', fontsize=8, color='darkred', weight='semibold'
            )
            
        ax.set_title(f"Architecture: {model}", weight='bold')
        ax.set_ylabel("Max-Normalized Latent Shift" if idx % 2 == 0 else "")
        ax.set_xticklabels(scenarios, rotation=15)
        
        ax.set_ylim(0, 1.15)
        ax.grid(True, linestyle=':', alpha=0.6)

    fig.suptitle("Max-Normalized Sim-to-Real Latent Shift Comparison Profiles [Bounded 0-1]", fontsize=15, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    save_path = os.path.join(save_folder, "publication_bar_grid.pdf")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"[Success] Saved Peak-Normalized 2x2 Bar Grid to: {save_path}")
    plt.close()


# --- Unified Line Plot for Ordered Discrepancies (Sparsity Plot) ---

def plot_thesis_ranked_curves(df, save_folder):
    """
    Plots the ordered latent discrepancy profiles for all four models together in a single chart.
    Uses the model-specific peak normalization to bound the curves exactly inside [0, 1].
    """
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 13,
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = df['Model'].unique()
    colors = sns.color_palette("tab10", len(models))

    for idx, model in enumerate(models):
        model_df = df[df['Model'] == model]
        
        # Compute mean relative shift per dimension across scenarios
        mean_shifts = model_df.groupby('Latent Dimension')['Relative Shift'].mean()
        
        # Sort values descending to trace the rank profile curve
        sorted_shifts = mean_shifts.sort_values(ascending=False).values
        
        # Clean line presentation without shaded fill
        ax.plot(
            range(len(sorted_shifts)), 
            sorted_shifts, 
            label=model, color=colors[idx], linewidth=2.0, alpha=0.9
        )

    # Adding a vertical reference guide for the top volatile dimensions
    reference_idx = 25 
    ax.axvline(x=reference_idx, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
    ax.text(
        reference_idx + 3, 0.85, 
        f"Top 10% Volatile Zone\n(Features 0-{reference_idx})", 
        fontsize=9, color='dimgray', weight='semibold'
    )

    ax.set_title("Ranked Latent Feature Sim-to-Real Discrepancy Profile [Max-Normalized]", weight='bold', pad=15)
    ax.set_xlabel("Latent Feature Rank (Descending Order by Mean Shift Discrepancy)")
    ax.set_ylabel("Normalized Mean Relative Shift")
    
    # Perfectly bounded inside [0, 1] as requested
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 1.05) 
    
    ax.legend(title="Agent Architecture", loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    
    save_path = os.path.join(save_folder, "publication_ranked_curves.pdf")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"[Success] Saved Max-Normalized Unified Discrepancy Line Curves to: {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Peak-Normalized Evaluation Pipeline")
    parser.add_argument("--models", type=str, nargs='+', required=True, help="Models to evaluate")
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
    args = parser.parse_args()

    CALIB_PATH = "artifacts/duckie_calib_data.txt"
    SIM_FOLDER = "screenshots/sim"
    REAL_FOLDER = "screenshots/realbot"
    
    OUTPUT_DIR = os.path.expanduser("~/workspace/rl_models/img/feature_extraction")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_latent_data = []

    for model_name in args.models:
        print(f"Extracting features for model: {model_name}")
        comparator = Sim2RealComparator(
            model_name=model_name, calib_path=CALIB_PATH, save_dir=OUTPUT_DIR, device=args.device, grayscale=True
        )
        model_results = comparator.extract_features(sim_folder=SIM_FOLDER, real_folder=REAL_FOLDER)
        all_latent_data.extend(model_results)

    df = pd.DataFrame(all_latent_data)
    
    # Apply model-specific peak normalization to strictly bound everything between 0 and 1
    for model in df['Model'].unique():
        mask = df['Model'] == model
        model_max = df.loc[mask, 'Relative Shift'].max()
        df.loc[mask, 'Relative Shift'] /= (model_max + 1e-8)

    print("\nGenerating Thesis Figures...")
    plot_thesis_bar_grid(df, OUTPUT_DIR)
    plot_thesis_ranked_curves(df, OUTPUT_DIR)
    print("All tasks completed successfully!")
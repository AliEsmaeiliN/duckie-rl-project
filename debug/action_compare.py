"""
Sim-to-Real Action Space Comparator
Visualizes policy behavior directly in control space (v, omega) across Sim vs Real
"""

import os
import argparse
import yaml
import numpy as np
import torch
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from models import SACActor, TD3Actor 
from utils.rl_env import DuckieOvalEnv


class ActionComparator:
    def __init__(self, model_name, calib_path, device="cpu", grayscale=True):
        self.device = torch.device(device)
        self.grayscale = grayscale
        self.model_name = model_name
        self.tilt_strength = 0.0006

        # Rule: If model has "ds" in the name, its input is downsized
        self.is_downscaled = "ds" in model_name.lower()
        self.target_size = (42, 42) if self.is_downscaled else (84, 84)
        print(f"[{self.model_name}] Downscaled: {self.is_downscaled} | Resolution: {self.target_size}")

        self.model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "rl_models", f"{model_name}.cleanrl_model"
        )

        # Build a dummy environment context just to satisfy shape inspection
        dummy_env = DuckieOvalEnv.create_wrapped(
            "dummy", grayscale=self.grayscale, downscaled=self.is_downscaled
        )

        # Dynamically allocate correct model architecture
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

        self._load_calibration(calib_path)
        self.maps_built = False

    def _load_calibration(self, calib_path):
        calib_path = os.path.expanduser(calib_path)
        with open(calib_path, 'r') as f:
            calib_data = yaml.safe_load(f)

        cam_mat = calib_data.get('camera_matrix', {})
        if isinstance(cam_mat, dict) and 'data' in cam_mat:
            cam_mat = cam_mat['data']
        dist_coefs = calib_data.get('distortion_coefficients', calib_data.get('distortion_coefs', {}))
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
        self.map_x = cv2.warpPerspective(map_x, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        self.map_y = cv2.warpPerspective(map_y, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        self.maps_built = True

    def preprocess_image(self, img_path, is_real_world=False):
        img = cv2.imread(os.path.expanduser(img_path))
        if img is None:
            raise FileNotFoundError(f"Could not load vision sample: {img_path}")
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

        stacked = np.tile(img, (4, 1, 1)) # standard frame stack of 4
        tensor = torch.Tensor(stacked).unsqueeze(0).to(self.device)
        return tensor

    def evaluate_actions(self, sim_folder: str, real_folder: str) -> list[dict]:
        suffixes = ['cr', 'cl', 'ss', 'sl']
        scenario_names = {
            'cr': 'Right Curve', 'cl': 'Left Curve',
            'ss': 'Short Straight', 'sl': 'Long Straight'
        }
        
        rows = []
        for sfx in suffixes:
            for domain_name, is_real in [("Sim", False), ("Real", True)]:
                folder = real_folder if is_real else sim_folder
                prefix = "real" if is_real else "sim"
                path = os.path.join(folder, f"{prefix}_{sfx}.png")
                
                tensor = self.preprocess_image(path, is_real_world=is_real)
                
                with torch.no_grad():
                    # If model has get_action (SAC deterministic execution setup)
                    if hasattr(self.actor, "get_action"):
                        _, _, action = self.actor.get_action(tensor)
                    else:
                        action = self.actor(tensor)
                        
                act_np = action.cpu().numpy().reshape(-1)
                
                rows.append({
                    "Model": self.model_name,
                    "Scenario": scenario_names[sfx],
                    "Domain": domain_name,
                    "Velocity (v)": float(act_np[0]),
                    "Steering (\u03c9)": -float(act_np[act_np.shape[0]-1]) # Handle variable shapes
                })
        return rows


def plot_action_space(df: pd.DataFrame, save_path: str):
    """
    Renders a clean 2D scatter plot matrix mapping action gaps.
    Uses unified colors for execution domains (Sim=Navy, Real=Red-Orange),
    maps track positions to geometric shapes, and places legends on the right side.
    """
    # 1. Set professional typography and look
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 11,
        'xtick.labelsize': 15,
        'ytick.labelsize': 10,
    })
    
    # 2. Complete naming architecture map
    name_mapping = {
        "sac_vr2": "SAC with Unified Reward",
        "sac_vr2_ds": "SAC with Unified Reward",
        "sac_vr1": "SAC with Adaptive Reward",
        "td3_vr1": "TD3 with Adaptive Reward",
        "td3_vr2": "TD3 with Unified Reward"
    }
    
    # Shape dictionary for track positions
    shape_mapping = {
        'Long Straight': 'o',    # Circle
        'Short Straight': 's',   # Square
        'Left Curve': '^',       # Upward Triangle
        'Right Curve': 'D'       # Diamond
    }
    
    # Unified execution domain color scheme
    COLOR_SIM = "#1C4783"   # Dark Navy Blue
    COLOR_REAL = '#FF4500'  # Red-Orange

    df = df.copy()
    df['Model'] = df['Model'].map(lambda x: name_mapping.get(x, x))
    
    unique_models = df['Model'].unique()
    num_models = len(unique_models)
    
    # Strict 2x2 override logic
    if num_models == 4:
        cols = 2
        rows = 2
    else:
        cols = min(num_models, 3)
        rows = (num_models + cols - 1) // cols
    
    # Increased right margin padding (via figsize adjust) to prevent legend clipping
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5 + 2.5, rows * 4.5), squeeze=False)
    axes_flat = axes.flatten()
    
    # 3. Plot each model onto its designated subplot canvas
    for idx, model_name in enumerate(unique_models):
        ax = axes_flat[idx]
        model_df = df[df['Model'] == model_name]
        
        # Dead-center reference axes
        ax.axhline(0.0, color='#888888', linestyle='-', linewidth=0.8, alpha=0.3)
        ax.axvline(0.0, color='#888888', linestyle='-', linewidth=0.8, alpha=0.3)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#E5E5E5')
        
        # Group points by scenario to connect pairs with vectors
        for scenario in model_df['Scenario'].unique():
            scen_df = model_df[model_df['Scenario'] == scenario]
            sim_pt = scen_df[scen_df['Domain'] == 'Sim']
            real_pt = scen_df[scen_df['Domain'] == 'Real']
            
            if not sim_pt.empty and not real_pt.empty:
                x_sim, y_sim = sim_pt["Steering (\u03c9)"].values[0], sim_pt["Velocity (v)"].values[0]
                x_real, y_real = real_pt["Steering (\u03c9)"].values[0], real_pt["Velocity (v)"].values[0]
                marker = shape_mapping.get(scenario, 'o')
                
                # Draw connecting action shift arrow indicating the delta direction
                ax.annotate(
                    "", xy=(x_real, y_real), xytext=(x_sim, y_sim),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2, ls=":", alpha=0.7)
                )
                
                # Plot Simulation point (Unified Dark Navy Blue)
                ax.scatter(
                    x_sim, y_sim, color=COLOR_SIM, 
                    marker=marker, s=130, edgecolors='#444444', linewidths=1.0, zorder=3
                )
                
                # Plot Real World point (Unified Red-Orange)
                ax.scatter(
                    x_real, y_real, color=COLOR_REAL, 
                    marker=marker, s=130, edgecolors='#111111', linewidths=1.2, zorder=4
                )
                
        ax.set_title(model_name, weight='bold', pad=10)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.05, 1.05)
        if idx % 2 == 0:
            ax.set_ylabel("Linear Velocity ($v$)", fontsize=16)
        if idx >= 2:
            ax.set_xlabel("Steering Angle ($\omega$)", fontsize=14)


    # Hide unused subplots if the grid isn't perfectly filled
    for idx in range(num_models, len(axes_flat)):
        fig.delaxes(axes_flat[idx])
        
    # 4. Construct Clean Custom Legends (Stacked on the Right Side)
    from matplotlib.lines import Line2D
    
    # Legend A: Scenarios mapped to shapes
    scenario_elements = [
        Line2D([0], [0], marker=shape_mapping[k], color='w', label=k,
               markerfacecolor='#777777', markersize=10, markeredgecolor='k')
        for k in shape_mapping
    ]
    legend_scen = fig.legend(
        handles=scenario_elements, 
        title="Track Position \n (Shape Key)", 
        loc='center left', 
        bbox_to_anchor=(0.82, 0.65), # Anchored outside on right upper section
        ncol=1, 
        frameon=True
    )
    
    # Legend B: Domain execution shift colors
    domain_elements = [
        Line2D([0], [0], marker='o', color='w', label='Simulation (Sim)',
               markerfacecolor=COLOR_SIM, markersize=11, markeredgecolor='#444444'),
        Line2D([0], [0], marker='o', color='w', label='Real World (Real)',
               markerfacecolor=COLOR_REAL, markersize=11, markeredgecolor='#111111')
    ]
    fig.legend(
        handles=domain_elements, 
        title="Execution Context \n (Color Key)", 
        loc='center left', 
        bbox_to_anchor=(0.82, 0.40), # Anchored outside on right lower section
        ncol=1, 
        frameon=True
    )
    
    fig.gca().add_artist(legend_scen) # Lock secondary tracking legend object safely onto canvas
    
    
    # Add explicit right-side layout margins to hold the legend boundaries
    plt.tight_layout()
    plt.subplots_adjust(right=0.80) 
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Success] Beautiful matrix plot with right-side legends saved to -> {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Action Space Distribution Space Comparator")
    parser.add_argument("--models", nargs='+', required=True, help="CleanRL model keys without extensions")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    CALIB_PATH = "artifacts/duckie_calib_data.txt"
    SIM_FOLDER = "screenshots/sim"
    REAL_FOLDER = "screenshots/realbot"
    OUTPUT_IMG = "img/action_space_comparison.png"
    
    os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
    
    all_records = []
    for model in args.models:
        comparator = ActionComparator(model, calib_path=CALIB_PATH, device=args.device)
        all_records.extend(comparator.evaluate_actions(SIM_FOLDER, REAL_FOLDER))
        
    df = pd.DataFrame(all_records)
    
    # Render Command Summary
    print("\n" + "="*80 + "\n\tEXTRACTED POLICY CONTROL RESPONSES\n" + "="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    plot_action_space(df, OUTPUT_IMG)
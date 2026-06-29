import os
import argparse
import yaml
import numpy as np
import torch
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Assuming your local models.py provides these definitions
from models import SACActor, TD3Actor 
from utils.rl_env import DuckieOvalEnv

# --- Same image extraction and data collection infrastructure from previous iterations ---
class ActionComparator:
    def __init__(self, model_name, calib_path, device="cpu", grayscale=True):
        self.device = torch.device(device)
        self.grayscale = grayscale
        self.model_name = model_name
        self.tilt_strength = 0.0006
        self.is_downscaled = "ds" in model_name.lower()
        self.target_size = (42, 42) if self.is_downscaled else (84, 84)

        self.model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "rl_models", f"{model_name}.cleanrl_model"
        )
        dummy_env = DuckieOvalEnv.create_wrapped("dummy", grayscale=self.grayscale, downscaled=self.is_downscaled)
        if "td3" in model_name.lower():
            self.actor = TD3Actor(dummy_env, downscaled=self.is_downscaled).to(self.device)
        else:
            self.actor = SACActor(dummy_env, downscaled=self.is_downscaled).to(self.device)

        checkpoint = torch.load(os.path.expanduser(self.model_path), map_location=self.device, weights_only=True)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor.eval()
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
        dst = np.float32([[cx - (cx - 0) * (1 - self.tilt_strength * h), shift],
                          [cx + (w - cx) * (1 - self.tilt_strength * h), shift], [w, h], [0, h]])
        return cv2.getPerspectiveTransform(src, dst)

    def _build_real_maps(self, w, h):
        new_cam, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.distortion_coefs, (w, h), 0, (w, h))
        map_x, map_y = cv2.initUndistortRectifyMap(self.camera_matrix, self.distortion_coefs, None, new_cam, (w, h), cv2.CV_32FC1)
        H = self._compute_tilt_homography(w, h)
        self.map_x = cv2.warpPerspective(map_x, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        self.map_y = cv2.warpPerspective(map_y, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        self.maps_built = True

    def preprocess_image(self, img_path, is_real_world=False):
        img = cv2.imread(os.path.expanduser(img_path))
        if img is None: raise FileNotFoundError(f"Missing: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        if is_real_world:
            if not self.maps_built: self._build_real_maps(w, h)
            img = cv2.remap(img, self.map_x, self.map_y, cv2.INTER_LINEAR)
            img = img[int(h * 0.4):h, int(w * 0.2):int(w * 0.8)]
            img = img[img.shape[0] // 3:, 0:img.shape[1]]
        else:
            img = img[h // 3:h, 0:w]
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, 0)
        else:
            img = img.transpose(2, 0, 1)
        return torch.Tensor(np.tile(img, (4, 1, 1))).unsqueeze(0).to(self.device)

    def evaluate_actions(self, sim_folder: str, real_folder: str) -> list[dict]:
        suffixes = ['cr', 'cl', 'ss', 'sl']
        scenario_names = {'cr': 'Right Curve', 'cl': 'Left Curve', 'ss': 'Short Straight', 'sl': 'Long Straight'}
        rows = []
        for sfx in suffixes:
            for domain_name, is_real in [("Sim", False), ("Real", True)]:
                path = os.path.join(real_folder if is_real else sim_folder, f"{'real' if is_real else 'sim'}_{sfx}.png")
                tensor = self.preprocess_image(path, is_real_world=is_real)
                with torch.no_grad():
                    action = self.actor.get_action(tensor)[2] if hasattr(self.actor, "get_action") else self.actor(tensor)
                act_np = action.cpu().numpy().reshape(-1)
                rows.append({
                    "Model": self.model_name, "Scenario": scenario_names[sfx], "Domain": domain_name,
                    "Velocity (v)": float(act_np[0]), "Steering (\u03c9)": float(act_np[-1])
                })
        return rows


# --- THESIS RESTOCKED GRAPH STRUCTURE ---
def plot_robustness_comparison(df: pd.DataFrame, save_path: str):
    """
    Plots a 2x2 grid containing 4 subplots (one per driving scenario).
    Each panel groups all models side-by-side to immediately contrast their sim-to-real shift.
    """
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.family': 'serif', 'axes.labelsize': 12, 'axes.titlesize': 12,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
    })

    name_mapping = {
        "sac_vr2": "SAC Unified",
        "sac_vr2_ds": "SAC Unified (42x42)",
        "sac_vr1": "SAC Adaptive",
        "td3_vr1": "TD3 Adaptive",
        "td3_vr2": "TD3 Unified"
    }
    df = df.copy()
    df['Model'] = df['Model'].map(lambda x: name_mapping.get(x, x))

    # Definitive, qualitative styling color keys per model variant
    model_colors = {
        "SAC Unified": "#1f77b4",   # Classic Blue
        "SAC Unified": "#aec7e8",   # Muted/Light Blue
        "SAC Adaptive": "#ff7f0e",  # Classic Orange
        "TD3 Adaptive": "#2ca02c",  # Green
        "TD3 Unified": "#9467bd"    # Purple
    }

    scenarios = ['Long Straight', 'Short Straight', 'Left Curve', 'Right Curve']
    
    # Perfect 2x2 multi-panel layout for Thesis layout standard
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, scenario in enumerate(scenarios):
        ax = axes_flat[idx]
        scen_df = df[df['Scenario'] == scenario]
        
        # Draw dead-center tracking references
        ax.axhline(0.0, color='#888888', linestyle='-', linewidth=0.8, alpha=0.3)
        ax.axvline(0.0, color='#888888', linestyle='-', linewidth=0.8, alpha=0.3)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#E5E5E5')

        # Evaluate and map each architecture inside this specific context
        for model in scen_df['Model'].unique():
            m_df = scen_df[scen_df['Model'] == model]
            sim_pt = m_df[m_df['Domain'] == 'Sim']
            real_pt = m_df[m_df['Domain'] == 'Real']
            
            if not sim_pt.empty and not real_pt.empty:
                x_sim, y_sim = sim_pt["Steering (\u03c9)"].values[0], sim_pt["Velocity (v)"].values[0]
                x_real, y_real = real_pt["Steering (\u03c9)"].values[0], real_pt["Velocity (v)"].values[0]
                color = model_colors.get(model, '#333333')

                # Shift connection vector representation
                ax.annotate(
                    "", xy=(x_real, y_real), xytext=(x_sim, y_sim),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5, ls="-", alpha=0.8, mutation_scale=12)
                )
                
                # Simulation Baseline position: Translucent circle ("Intended trajectory")
                ax.scatter(x_sim, y_sim, color=color, marker='o', s=100, edgecolors=color, facecolors='none', lw=1.5, alpha=0.6)
                
                # Real World Response position: Solid Filled square ("Physical consequence")
                ax.scatter(x_real, y_real, color=color, marker='s', s=100, edgecolors='black', linewidths=0.8, zorder=4)

        ax.set_title(f"Track Scenario: {scenario}", weight='bold', pad=10)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        if idx in [2, 3]: ax.set_xlabel("Steering Angle ($\omega$)")
        if idx in [0, 2]: ax.set_ylabel("Linear Velocity ($v$)")

    # --- Constructing Academic Dual-Legends on Right Aspect ---
    # Legend A: Distinct Model Algorithms 
    model_elements = [
        Line2D([0], [0], marker='s', color='w', label=m, markerfacecolor=c, markersize=8)
        for m, c in model_colors.items() if m in df['Model'].unique()
    ]
    leg_model = fig.legend(handles=model_elements, title="Policy" , 
                           loc='center left', bbox_to_anchor=(0.83, 0.65), frameon=True)

    # Legend B: Domain State Transitions (Unfilled Circle -> Filled Square)
    domain_elements = [
        Line2D([0], [0], marker='o', color='w', label='Simulation', 
               markerfacecolor='none', markeredgecolor='#444444', markeredgewidth=1.5, markersize=9),
        Line2D([0], [0], marker='s', color='w', label='Real World', 
               markerfacecolor='#444444', markeredgecolor='black', markersize=9)
    ]
    fig.legend(handles=domain_elements, title="Domain", 
               loc='center left', bbox_to_anchor=(0.83, 0.40), frameon=True)
    
    fig.gca().add_artist(leg_model) # Preserve both legends natively
    

    plt.tight_layout()
    plt.subplots_adjust(right=0.81, hspace=0.22, wspace=0.15)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Thesis Output Verified] Robustness profile matrix saved directly to -> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thesis Systemic Robustness Comparator")
    parser.add_argument("--models", nargs='+', required=True, help="Models list input matching checkpoints")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    CALIB_PATH = "artifacts/duckie_calib_data.txt"
    SIM_FOLDER = "screenshots/sim"
    REAL_FOLDER = "screenshots/realbot"
    OUTPUT_IMG = "img/extracted_features/robustness_matrix_thesis.png"
    
    os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
    all_records = []
    for m in args.models:
        all_records.extend(ActionComparator(m, calib_path=CALIB_PATH, device=args.device).evaluate_actions(SIM_FOLDER, REAL_FOLDER))
        
    plot_robustness_comparison(pd.DataFrame(all_records), OUTPUT_IMG)
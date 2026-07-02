"""
Sim-to-Real Cosine Similarity Comparator
=========================================
Isolates scale-invariant directional alignment to directly compare representational 
geometry between standard and downscaled vision-based RL models.
"""

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


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Scale-invariant directional alignment.
    Returns 1.0 for identical directions, 0.0 for orthogonal, -1.0 for opposite.
    Perfect for comparing latent geometry regardless of activation magnitude.
    """
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)


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
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "rl_models", f"{model_name}.cleanrl_model"
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
        self.map_x = cv2.warpPerspective(map_x, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        self.map_y = cv2.warpPerspective(map_y, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
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

    def _get_hook(self, name):
        def hook(model, input, output):
            self.outputs[name] = output.detach()
        return hook

    def _register_hooks(self):
        seq = self.actor.encoder.main
        seq[-1].register_forward_hook(self._get_hook('Latent'))

    def extract_latents(self, sim_folder: str, real_folder: str) -> dict:
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
        suffixes = ['cr', 'cl', 'ss', 'sl']
        scenario_names = {
            'cr': 'Right Curve',
            'cl': 'Left Curve',
            'ss': 'Short Straight',
            'sl': 'Long Straight'
        }

        data = self.extract_latents(sim_folder, real_folder)
        sim_vecs = data['sim']
        real_vecs = data['real']

        rows = []
        for i, sfx in enumerate(suffixes):
            s, r = sim_vecs[i], real_vecs[i]
            rows.append({
                "Model": self.model_name,
                "Scenario": scenario_names[sfx],
                "Cosine Similarity": cosine_similarity(s, r),
            })
        return rows


# --- Global Plotting & Styling ---

SCENARIO_ORDER = ['Long Straight', 'Short Straight', 'Left Curve', 'Right Curve']

def _style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })

def plot_cosine_comparison(df: pd.DataFrame, save_folder: str):
    """
    Generates a high-quality grouped bar chart explicitly comparing 
    Cosine Similarity across models and driving scenarios.
    """
    _style()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    
    name_mapping = {
        "sac_vr2": "Original Baseline (84x84)",
        "sac_vr2_ds": "Downsampled Policy (42x42)",
        "sac_vr1": "Original Baseline (84x84)",
        "td3_vr1": "Downsampled Policy (42x42)",
        "td3_vr2": "Original Baseline (84x84)"
    }
    
    df = df.copy()
    df['Model'] = df['Model'].map(lambda x: name_mapping.get(x, x))
    df['Scenario'] = pd.Categorical(df['Scenario'], categories=SCENARIO_ORDER, ordered=True)
    df = df.sort_values('Scenario')

    custom_colors = {
        "Original Baseline (84x84)": "#1B365D",     # Deep Navy Blue
        "Downsampled Policy (42x42)": "#E06666"     # Soft Coral / Muted Red-Salmon for contrast
    }

    sns.barplot(
        data=df,
        x='Scenario',
        y='Cosine Similarity',
        hue='Model',
        palette=custom_colors,  # Using high-contrast publication friendly palette
        alpha=0.9,
        ax=ax
    )

    # Upper bound alignment indicator line
    ax.axhline(1.0, color='black', linewidth=1.0, linestyle='--', alpha=0.5, label='Perfect Alignment (1.0)')
    
    ax.set_xlabel("Track Evaluation Scenario", fontsize=12)
    ax.set_ylabel("Cosine Similarity Value", fontsize=16)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(title="Agent Configuration", frameon=True, loc='lower left')
    
    plt.tight_layout()
    path = os.path.join(save_folder, "ds_cosine_similarity_comparison.pdf")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Saved Plot] {path}")

def print_summary_table(df: pd.DataFrame):
    # Remap strings for terminal output readability too
    name_mapping = {
        "sac_vr2": "Original Baseline (84x84)",
        "sac_vr2_ds": "Downsampled Policy (42x42)"
    }
    df_print = df.copy()
    df_print['Model'] = df_print['Model'].map(lambda x: name_mapping.get(x, x))
    
    print("\n" + "=" * 75)
    print("  SIM-TO-REAL LATENT COSINE SIMILARITY SUMMARY")
    print("=" * 75)
    print(df_print.to_string(index=False))
    print("=" * 75)
    print("  Note: Higher values (closer to 1.0) indicate robust spatial alignment.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sim-to-Real Cosine Feature Comparator")
    parser.add_argument("--models", nargs='+', required=True, help="Model names (must match .cleanrl_model files)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    CALIB_PATH = "artifacts/duckie_calib_data.txt"
    SIM_FOLDER = "screenshots/sim"
    REAL_FOLDER = "screenshots/realbot"
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "img", "extracted_features")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_rows = []

    for model_name in args.models:
        print(f"\nEvaluating Weights for: {model_name}")
        comp = Sim2RealComparator(
            model_name=model_name,
            calib_path=CALIB_PATH,
            save_dir=OUTPUT_DIR,
            device=args.device,
            grayscale=True
        )
        rows = comp.compute_metrics(SIM_FOLDER, REAL_FOLDER)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "cosine_metrics.csv"), index=False)

    print_summary_table(df)
    plot_cosine_comparison(df, OUTPUT_DIR)
    print("Execution finalized cleanly.")
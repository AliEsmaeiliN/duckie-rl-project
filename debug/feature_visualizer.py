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
        if not os.path.exists(calib_path):
            raise FileNotFoundError(f"Calibration file not found at: {calib_path}")
            
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
            [w, h],
            [0, h]
        ])
        return cv2.getPerspectiveTransform(src, dst)

    def _build_real_maps(self, w, h):
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.distortion_coefs, (w, h), 0, (w, h)
        )
        map_x, map_y = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.distortion_coefs, None, new_camera_matrix, (w, h), cv2.CV_32FC1
        )
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
        self.hook_conv1 = encoder_seq[1].register_forward_hook(self._get_hook('Conv1'))
        self.hook_latent = encoder_seq[-1].register_forward_hook(self._get_hook('Latent'))

    def preprocess_image(self, img_path, is_real_world=False):
        img_path = os.path.expanduser(img_path)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        if is_real_world:
            if not self.maps_built:
                self._build_real_maps(w, h)
                
            img = cv2.remap(img, self.map_x, self.map_y, cv2.INTER_LINEAR)

            v_crop_frac = 0.4
            top = int(h * v_crop_frac)
            h_crop_frac = 0.2
            left = int(w * h_crop_frac)
            right = int(w * (1.0 - h_crop_frac))
            img = img[top:h, left:right]
            pre_crop_img = img.copy()
            h1, w1 = img.shape[:2]
            top_boundary = int(h1 / 3)
            img = img[top_boundary:h1, 0:w1]
        else:
            pre_crop_img = img.copy()
            top_boundary = int(h / 3)
            img = img[top_boundary:h, 0:w]

        img = cv2.resize(img, (42, 42), interpolation=cv2.INTER_LINEAR) 

        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, axis=0)
        else:
            img = img.transpose(2, 0, 1)

        num_frames = 4
        stacked_img = np.tile(img, (num_frames, 1, 1))

        tensor = torch.Tensor(stacked_img).unsqueeze(0).to(self.device)
        return tensor, pre_crop_img

    def extract_features(self, sim_folder, real_folder):
        images_suffix = ['cr', 'cl', 'ss', 'sl']
        scenario_names = ['Right Curve', 'Left Curve', 'Short Straight', 'Long Straight']
        model_data = []

        for i in range(4):
            sim_path = os.path.join(sim_folder, f"sim_{images_suffix[i]}.png")
            real_path = os.path.join(real_folder, f"real_{images_suffix[i]}.png")
            
            sim_tensor, sim_pre_crop = self.preprocess_image(sim_path, is_real_world=False)
            with torch.no_grad():
                _ = self.actor(sim_tensor)
            sim_conv1 = self.outputs['Conv1'][0].cpu().numpy()
            sim_latent = self.outputs['Latent'][0].cpu().numpy()

            real_tensor, real_pre_crop = self.preprocess_image(real_path, is_real_world=True)
            with torch.no_grad():
                _ = self.actor(real_tensor)
            real_conv1 = self.outputs['Conv1'][0].cpu().numpy()
            real_latent = self.outputs['Latent'][0].cpu().numpy()

            epsilon = 1e-5
            rel_diff = np.abs(sim_latent - real_latent) / (np.abs(sim_latent) + epsilon)
            
            for feature_idx, shift_val in enumerate(rel_diff):
                model_data.append({
                    "Model": self.model_name,
                    "Scenario": scenario_names[i],
                    "Latent Dimension": feature_idx,
                    "Relative Shift": float(shift_val)
                })

            if self.model_name == "sac_vr2":
                self._save_prepp_image(sim_pre_crop, "sim", images_suffix[i])
                self._save_prepp_image(real_pre_crop, "real", images_suffix[i])
                self._save_conv1_grid(sim_conv1, "sim", images_suffix[i])
                self._save_conv1_grid(real_conv1, "real", images_suffix[i])
                self._save_latent_heatmap(sim_latent, "sim", images_suffix[i])
                self._save_latent_heatmap(real_latent, "real", images_suffix[i])

        return model_data

    def _save_prepp_image(self, img, prefix, label):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(img)
        ax.axis('off')
        save_path = os.path.join(self.save_folder, f"{prefix}_{self.model_name}_{label}_prepp.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close(fig)

    def _save_conv1_grid(self, fm, prefix, label):
        num_filters = min(16, fm.shape[0])
        fig, axs = plt.subplots(4, 4, figsize=(6, 6))
        fig.patch.set_facecolor('black') 
        for i, ax in enumerate(axs.flat):
            if i < num_filters:
                ax.imshow(fm[i], cmap='viridis')
            ax.axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        save_path = os.path.join(self.save_folder, f"{prefix}_{self.model_name}_{label}_conv1.png")
        plt.savefig(save_path, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
        plt.close(fig)

    def _save_latent_heatmap(self, latent_vec, prefix, label):
        """Saves the latent vector as a 1x256 color-coded horizontal strip."""
        # Made the figure wide and short to accommodate the 1D strip nicely
        fig, ax = plt.subplots(figsize=(15, 1.5))
        
        # Reshape to exactly 1 row by N columns
        grid = latent_vec.reshape((1, len(latent_vec)))
        
        cax = ax.imshow(grid, cmap='magma', aspect='auto')
        fig.colorbar(cax, ax=ax, fraction=0.015, pad=0.02)
        
        ax.set_title(f"Latent Activation: {prefix.upper()} ({label})", weight='bold', pad=10)
        # We leave the X-axis visible so you know which feature index is which, but hide Y
        ax.get_yaxis().set_visible(False)
        ax.set_xlabel("Latent Feature Index")
        
        save_path = os.path.join(self.save_folder, f"{prefix}_{self.model_name}_{label}_latent_heatmap.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

# --- Global Plotting Functions ---

def plot_faceted_violin(df, save_folder):
    """
    Generates a publication-quality 2x2 faceted boxplot grid with a faint 
    underlying sample distribution layer. Outliers are cropped for scale clarity.
    """
    # 1. Establish strict publication formatting
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.family': 'serif',
        'text.usetex': False,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })

    # 2. Reorder scenarios logically (Straights grouped, Curves grouped)
    scenario_order = ['Long Straight', 'Short Straight', 'Left Curve', 'Right Curve']
    df = df.copy()
    df['Scenario'] = pd.Categorical(df['Scenario'], categories=scenario_order, ordered=True)

    # 3. Initialize the core FacetGrid
    g = sns.FacetGrid(
        data=df,
        col='Model',
        col_wrap=2,
        height=4.2,
        aspect=1.2,
        sharey=True
    )

    # 4. Map clean, non-decorative boxplots
    g.map_dataframe(
        sns.boxplot,
        x='Scenario',
        y='Relative Shift',
        hue='Scenario',
        showfliers=False,      # Hides the default ugly black outlier diamonds
        linewidth=1.2,
        width=0.5,             # Makes boxes slightly narrower and sharper
        palette='muted',
        legend=False
    )

    # 5. Overlay the actual sample density using a faint, tiny strip layer
    g.map_dataframe(
        sns.stripplot,
        x='Scenario',
        y='Relative Shift',
        color='black',
        alpha=0.12,            # Highly transparent to prevent visual noise
        size=2.0,
        jitter=0.20,
        dodge=False
    )

    # 6. Final axis polishing
    g.set_axis_labels("", r"Relative Shift Scale $\left( \frac{|z_{sim} - z_{real}|}{|z_{sim}|} \right)$")
    g.set_titles(col_template="{col_name}", weight='bold')

    for ax in g.axes.flat:
        ax.set_ylim(-0.1, 5.5) # Hard cap to keep structural features clearly visible
        for label in ax.get_xticklabels():
            label.set_rotation(20)
            
        ax.text(0.03, 0.93, "*Outliers (>5.5) omitted for scale visualization clarity", 
                transform=ax.transAxes, fontsize=8, color='gray', style='italic')

    g.fig.subplots_adjust(top=0.88, hspace=0.35)
    g.fig.suptitle("Sim-to-Real Latent Domain Shift Quantile Distributions by Architecture", 
                   fontsize=14, weight='bold')

    save_path = os.path.join(save_folder, "publication_box_grid.pdf")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"[Success] Saved publication-ready boxplot grid to: {save_path}")
    plt.close()


def plot_thesis_sparsity_curves(df, save_folder):
    """
    Generates a sleek, line-only latent sparsity curve using a logarithmic scale 
    and clear semantic demarcation lines.
    """
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 13,
    })

    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    models = df['Model'].unique()
    colors = sns.color_palette("tab10", len(models))

    for idx, model in enumerate(models):
        model_df = df[df['Model'] == model]
        
        # Aggregate across dimensional spaces
        mean_shifts = model_df.groupby('Latent Dimension')['Relative Shift'].mean()
        sorted_shifts = mean_shifts.sort_values(ascending=False).values
        
        # Sleek, clean line strokes without shaded regions
        ax.plot(
            range(len(sorted_shifts)), 
            sorted_shifts, 
            label=model, 
            color=colors[idx], 
            linewidth=2.0, 
            alpha=0.9
        )

    # Apply logarithmic Y transform to evaluate tail convergence accurately
    ax.set_yscale("log")
    
    # Add vertical reference line indicating structural feature compression (Top 10%)
    sparsity_threshold_idx = 25 
    ax.axvline(
        x=sparsity_threshold_idx, 
        color='gray', 
        linestyle=':', 
        linewidth=1.2, 
        alpha=0.8
    )
    ax.text(
        sparsity_threshold_idx + 2, 
        ax.get_ylim()[1] * 0.2, 
        f"Top 10% Volatile Zone\n(Features 0-{sparsity_threshold_idx})", 
        fontsize=9, 
        color='dimgray', 
        weight='semibold'
    )

    ax.set_title("Latent Representation Sparsity Profile (Log-Scale Error Rank)", weight='bold', pad=15)
    ax.set_xlabel("Latent Feature Rank (Descending Order by Mean Shift Discrepancy)")
    ax.set_ylabel("Mean Relative Shift (Log Scale)")
    ax.set_xlim(0, 255)
    
    ax.legend(title="Agent Architecture", loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    
    save_path = os.path.join(save_folder, "publication_sparsity_curves.pdf")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"[Success] Saved log-scale sparsity visualization to: {save_path}")
    plt.close()

def plot_overlaid_sparsity(df, save_folder):
    """
    Generates a sleek, high-contrast line metric analyzing latent compression space.
    Implements a logarithmic y-scale, thinner clean strokes, and explicit sparsity boundaries.
    """
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 13,
    })

    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    models = df['Model'].unique()
    colors = sns.color_palette("tab10", len(models))

    for idx, model in enumerate(models):
        model_df = df[df['Model'] == model]
        
        # Aggregate across dimensional spaces
        mean_shifts = model_df.groupby('Latent Dimension')['Relative Shift'].mean()
        sorted_shifts = mean_shifts.sort_values(ascending=False).values
        
        # Replaced thick fills with a clean, thin line-only presentation
        ax.plot(
            range(len(sorted_shifts)), 
            sorted_shifts, 
            label=model, 
            color=colors[idx], 
            linewidth=2.0, 
            alpha=0.9
        )

    # (a) Logarithmic transform to accurately trace tracking errors without blowing out graphs
    ax.set_yscale("log")
    
    # (b) Add vertical reference line indicating structural feature compression
    sparsity_threshold_idx = 25 # Top ~10% of features
    ax.axvline(
        x=sparsity_threshold_idx, 
        color='gray', 
        linestyle=':', 
        linewidth=1.2, 
        alpha=0.8
    )
    ax.text(
        sparsity_threshold_idx + 2, 
        ax.get_ylim()[1] * 0.2, 
        f"Top 10% Volatile Zone\n(Features 0-{sparsity_threshold_idx})", 
        fontsize=9, 
        color='dimgray', 
        weight='semibold'
    )

    ax.set_title("Latent Representation Sparsity Profile (Log-Scale Error Rank)", weight='bold', pad=15)
    ax.set_xlabel("Latent Feature Rank (Descending Order by Mean Shift Discrepancy)")
    ax.set_ylabel("Mean Relative Shift (Log Scale)")
    ax.set_xlim(0, 255)
    
    ax.legend(title="Agent Architecture", loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    
    save_path = os.path.join(save_folder, "publication_sparsity_curves.pdf")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"[Success] Saved log-scale sparsity visualization to: {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale-Agnostic Sim-to-Real Feature Evaluator")
    parser.add_argument("--models", type=str, nargs='+', required=True, 
                        help="List of models to evaluate (e.g., --models sac_v1 sac_v2 td3_v1 td3_v2)")
    parser.add_argument("--device", type=str, default="cuda", help="Execution context device (cuda/cpu)")
    args = parser.parse_args()

    CALIB_PATH = "artifacts/duckie_calib_data.txt"
    SIM_FOLDER = "screenshots/sim"
    REAL_FOLDER = "screenshots/realbot"
    
    OUTPUT_DIR = os.path.expanduser("~/workspace/rl_models/img/feature_extraction")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_latent_data = []

    for model_name in args.models:
        print(f"\n=======================================================")
        print(f"Processing Model: {model_name}")
        print(f"=======================================================")
        
        comparator = Sim2RealComparator(
            model_name=model_name, 
            calib_path=CALIB_PATH,
            save_dir=OUTPUT_DIR,   
            device=args.device,     
            grayscale=True     
        )
        
        model_results = comparator.extract_features(sim_folder=SIM_FOLDER, real_folder=REAL_FOLDER)
        all_latent_data.extend(model_results)

    df = pd.DataFrame(all_latent_data)

    print("\nGenerating Thesis-Grade Aggregated Plots...")
    plot_faceted_violin(df, OUTPUT_DIR)
    plot_overlaid_sparsity(df, OUTPUT_DIR)
    
    print("\nAll done! Check your output folder.")
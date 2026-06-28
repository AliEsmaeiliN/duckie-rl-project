import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import os

from models import SACActor, TD3Actor
from utils.rl_env import DuckieOvalEnv 

class Sim2RealComparator:
    def __init__(self, model_name, calib_path, device="cpu", grayscale=True):
        self.device = torch.device(device)
        self.grayscale = grayscale
        self.tilt_strength = 0.0006

        self.save_dir = os.path.expanduser("~/workspace/rl_models")
        self.save_folder = 'img/feature_extraction'
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)
            
        self.model_name = model_name
        self.model_path = os.path.join(self.save_dir, f"{model_name}.cleanrl_model")
        
        dummy_env = DuckieOvalEnv.create_wrapped("dummy", grayscale=self.grayscale)
        
        if "td3" in model_name.lower():
            print(f"Detected TD3 model sequence. Instantiating TD3Actor for: {model_name}")
            self.actor = TD3Actor(dummy_env).to(self.device)
        else:
            print(f"Detected SAC model sequence. Instantiating SACActor for: {model_name}")
            self.actor = SACActor(dummy_env).to(self.device)
        
        print(f"Loading model: {self.model_path}")
        checkpoint = torch.load(os.path.expanduser(self.model_path), map_location=self.device, weights_only=True)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor.eval()

        self.outputs = {}
        self._register_hooks()

        self._load_calibration(calib_path)
        self.maps_built = False

    def _load_calibration(self, calib_path):
        """Loads intrinsic parameters."""
        calib_path = os.path.expanduser(calib_path)
        if not os.path.exists(calib_path):
            raise FileNotFoundError(f"Calibration file not found at: {calib_path}")
            
        print(f"Loading calibration data from: {calib_path}")
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
        """Builds a homography that simulates tilting the camera downward."""
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
        """Pre-computes the combined undistort + tilt maps."""
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
        """
        Diverges preprocessing to match the Exact Sim vs Real pipelines.
        """
        img_path = os.path.expanduser(img_path)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Keep RGB for matplotlib plotting
        h, w = img.shape[:2]

        if is_real_world:
            if not self.maps_built:
                self._build_real_maps(w, h)
                
            # Apply Undistort + Tilt
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

        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_LINEAR) 

        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, axis=0)
        else:
            img = img.transpose(2, 0, 1)

        num_frames = 4
        stacked_img = np.tile(img, (num_frames, 1, 1))

        tensor = torch.Tensor(stacked_img).unsqueeze(0).to(self.device)
        return tensor, pre_crop_img

    def compare_multiple(self, sim_folder, real_folder):
        """
        Iterates through matching images, plots Conv1 for each pair, 
        and aggregates the latent shift.
        """
        latent_diffs = []
        
        # Iterating through your pairs 1, 2, and 3
        for i in range(1, 4):
            sim_path = os.path.join(sim_folder, f"sim{i}.png")
            real_path = os.path.join(real_folder, f"real{i}.png")
            
            print(f"\n--- Processing Pair {i}: {sim_path} vs {real_path} ---")
            
            # Forward Pass: Simulation
            sim_tensor, sim_pre_crop = self.preprocess_image(sim_path, is_real_world=False)
            with torch.no_grad():
                _ = self.actor(sim_tensor)
            sim_conv1 = self.outputs['Conv1'][0].cpu().numpy()
            sim_latent = self.outputs['Latent'][0].cpu().numpy()

            # Forward Pass: Real World
            real_tensor, real_pre_crop = self.preprocess_image(real_path, is_real_world=True)
            with torch.no_grad():
                _ = self.actor(real_tensor)
            real_conv1 = self.outputs['Conv1'][0].cpu().numpy()
            real_latent = self.outputs['Latent'][0].cpu().numpy()

            # Store the latent absolute difference for the aggregate chart
            latent_diffs.append(np.abs(sim_latent - real_latent))

            # Optional: Visualize pre-crop alignment for context
            # self._plot_pre_crop_views(sim_pre_crop, real_pre_crop, idx=i)

            # Save individual Conv1 comparisons
            self._plot_conv_comparison(sim_conv1, real_conv1, idx=i)

        # Plot and save Aggregated Latent Results across all 3 pairs
        self._plot_aggregated_latent_differences(latent_diffs)

    def _plot_pre_crop_views(self, sim_img, real_img, idx):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Pre-Crop View Alignment - Pair {idx}", fontsize=14, fontweight='bold')
        ax1.imshow(sim_img); ax1.set_title("Simulation (Raw View)"); ax1.axis('off')
        ax2.imshow(real_img); ax2.set_title("Real Bot (Undistort + Tilt Applied)"); ax2.axis('off')
        plt.tight_layout()
        plt.show()

    def _plot_conv_comparison(self, sim_fm, real_fm, idx):
        """Plots the first 16 filters of Conv1 in two side-by-side 4x4 grids."""
        num_filters = min(16, sim_fm.shape[0])
        
        fig = plt.figure(figsize=(14, 7))
        fig.suptitle(f"Layer 1 (Conv1) Activations [{self.model_name}] - Pair {idx}", fontsize=16, fontweight='bold')
        
        subfigs = fig.subfigures(1, 2, wspace=0.05)
        
        subfigs[0].suptitle('Simulation', fontsize=14)
        axs_sim = subfigs[0].subplots(4, 4)
        
        subfigs[1].suptitle('Real World', fontsize=14)
        axs_real = subfigs[1].subplots(4, 4)
        
        for i in range(num_filters):
            row, col = divmod(i, 4)
            axs_sim[row, col].imshow(sim_fm[i], cmap='viridis')
            axs_sim[row, col].axis('off')
            axs_real[row, col].imshow(real_fm[i], cmap='viridis')
            axs_real[row, col].axis('off')

        for i in range(num_filters, 16):
            row, col = divmod(i, 4)
            axs_sim[row, col].axis('off')
            axs_real[row, col].axis('off')

        # Save individual Conv layer visualizations
        save_path = os.path.join(self.save_folder, f"{self.model_name}_conv_pair{idx}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"--> Saved Conv comparison to: {save_path}")
        plt.close(fig) # Use plt.close to prevent stacking interactive windows if running in bulk

    def _plot_aggregated_latent_differences(self, diff_list):
        """Plots mean latent difference across all 3 seeds with STD error bars."""
        diffs = np.array(diff_list) # Shape: (3, latent_dim)
        mean_diff = np.mean(diffs, axis=0)
        std_diff = np.std(diffs, axis=0)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        x = range(len(mean_diff))
        
        ax.bar(x, mean_diff, yerr=std_diff, 
               color='teal', alpha=0.7, ecolor='black', capsize=2, label='Mean Difference')
        
        ax.set_title(f"Aggregated Sim-to-Real Shift [{self.model_name}] (N=3)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Latent Feature Index")
        ax.set_ylabel("Absolute Latent Difference")
        
        # Add a baseline for context
        overall_mean = np.mean(mean_diff)
        ax.axhline(overall_mean, color='red', linestyle='--', label=f'Overall Mean Diff ({overall_mean:.2f})')
        ax.legend()
        
        save_path = os.path.join(self.save_folder, f"{self.model_name}_latent_agg.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"--> Saved Aggregated Latent comparison to: {save_path}")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical Sim-to-Real Feature Comparator")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., sac_vr2)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference (cuda/cpu)")
    args = parser.parse_args()

    CALIB_PATH = "artifacts/duckie_calib_data.txt"
    SIM_FOLDER = "screenshots/sim"
    REAL_FOLDER = "screenshots/realbot"

    comparator = Sim2RealComparator(
        model_name=args.model, 
        calib_path=CALIB_PATH,
        device=args.device,     
        grayscale=True     
    )
    
    comparator.compare_multiple(sim_folder=SIM_FOLDER, real_folder=REAL_FOLDER)
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import os

from models import SACActor as Actor
from utils.rl_env import DuckieOvalEnv 

class Sim2RealComparator:
    def __init__(self, model_name, calib_path, device="cpu", grayscale=True):
        self.device = torch.device(device)
        self.grayscale = grayscale
        self.tilt_strength = 0.0006

        self.save_dir = os.path.expanduser("~/workspace/rl_models")
        self.model_name = model_name
        self.model_path = os.path.join(self.save_dir, f"{model_name}.cleanrl_model")
        
        # 1. Initialize Actor
        dummy_env = DuckieOvalEnv.create_wrapped("dummy", grayscale=self.grayscale)
        self.actor = Actor(dummy_env).to(self.device)
        
        # 2. Load Weights
        print(f"Loading model: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
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

    def compare(self, sim_img_path, real_img_path):
        print("\nProcessing Simulation Image...")
        sim_tensor, sim_pre_crop = self.preprocess_image(sim_img_path, is_real_world=False)
        with torch.no_grad():
            _ = self.actor(sim_tensor)
        sim_conv1 = self.outputs['Conv1'][0].cpu().numpy()
        sim_latent = self.outputs['Latent'][0].cpu().numpy()

        print("Processing Real-World Image...")
        real_tensor, real_pre_crop = self.preprocess_image(real_img_path, is_real_world=True)
        with torch.no_grad():
            _ = self.actor(real_tensor)
        real_conv1 = self.outputs['Conv1'][0].cpu().numpy()
        real_latent = self.outputs['Latent'][0].cpu().numpy()

        self._plot_pre_crop_views(sim_pre_crop, real_pre_crop)
        self._plot_conv_comparison(sim_conv1, real_conv1)
        self._plot_latent_difference(sim_latent, real_latent)

    def _plot_pre_crop_views(self, sim_img, real_img):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Pre-Crop View Alignment: Simulation vs. Real Bot", fontsize=14, fontweight='bold')
        
        ax1.imshow(sim_img)
        ax1.set_title("Simulation (Raw View)")
        ax1.axis('off')
        
        ax2.imshow(real_img)
        ax2.set_title("Real Bot (Undistort + Tilt Applied)")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()

    def _plot_conv_comparison(self, sim_fm, real_fm):
        """Plots the first 16 filters of Conv1 in two side-by-side 4x4 grids."""
        num_filters = min(16, sim_fm.shape[0])
        
        fig = plt.figure(figsize=(14, 7))
        fig.suptitle("Layer 1 (Conv1) Activations", fontsize=16, fontweight='bold')
        
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

        plt.show()

    def _plot_latent_difference(self, sim_latent, real_latent):
        latent_diff = np.abs(sim_latent - real_latent)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [1, 2]})
        
        im_data = np.vstack([sim_latent, real_latent])
        im = ax1.imshow(im_data, aspect='auto', cmap='magma')
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(["Sim Latent", "Real Latent"])
        ax1.set_title(f"Final Latent Representation (Size: {sim_latent.shape[0]})")
        fig.colorbar(im, ax=ax1, fraction=0.02, pad=0.04)

        ax2.bar(range(len(latent_diff)), latent_diff, color='teal', edgecolor='black', alpha=0.8)
        ax2.set_xlabel("Latent Feature Index")
        ax2.set_ylabel("Absolute Difference")
        ax2.set_title("Distribution Shift: | Sim_Latent - Real_Latent |")
        
        mean_diff = np.mean(latent_diff)
        ax2.axhline(mean_diff, color='red', linestyle='--', label=f'Mean Diff ({mean_diff:.2f})')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sim-to-Real Feature Comparator")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., sac_vr2)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference (cuda/cpu)")
    args = parser.parse_args()

    CALIB_PATH = "artifacts/duckie_calib_data.txt"
    SIM_IMAGE_PATH = "screenshots/sim/sim1.png"
    REAL_IMAGE_PATH = "screenshots/realbot/real1.png"

    comparator = Sim2RealComparator(
        model_name=args.model, 
        calib_path=CALIB_PATH,
        device=args.device,     
        grayscale=True     
    )
    
    comparator.compare(sim_img_path=SIM_IMAGE_PATH, real_img_path=REAL_IMAGE_PATH)
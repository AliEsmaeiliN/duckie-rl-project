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
    def __init__(self, model_path, calib_path, device="cpu", grayscale=True):
        self.device = torch.device(device)
        self.grayscale = grayscale
        
        # 1. Initialize a dummy environment to get the observation/action spaces
        # This is needed to instantiate the Actor properly
        dummy_env = DuckieOvalEnv.create_wrapped("dummy", grayscale=self.grayscale)
        self.actor = Actor(dummy_env).to(self.device)
        
        # 2. Load the trained weights
        checkpoint = torch.load(os.path.expanduser(model_path), map_location=self.device, weights_only=True)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor.eval()

        self.outputs = {}
        self._register_hooks()

        # 3. Load Real-world Camera Calibration Parameters from file
        self._load_calibration(calib_path)

    def _load_calibration(self, calib_path):
        """Loads camera matrix and distortion coefficients from the provided text file."""
        calib_path = os.path.expanduser(calib_path)
        if not os.path.exists(calib_path):
            raise FileNotFoundError(f"Calibration file not found at: {calib_path}")
            
        print(f"Loading calibration data from: {calib_path}")
        try:
            with open(calib_path, 'r') as f:
                # Duckietown calib files are usually YAML formatted
                calib_data = yaml.safe_load(f)
            
            # NOTE: Adjust these keys if your .txt file structure differs slightly.
            # Typical Duckietown format nests the matrix inside a 'data' key.
            cam_mat = calib_data.get('camera_matrix', {})
            if isinstance(cam_mat, dict) and 'data' in cam_mat:
                cam_mat = cam_mat['data']
                
            dist_coefs = calib_data.get('distortion_coefficients', calib_data.get('distortion_coefs', {}))
            if isinstance(dist_coefs, dict) and 'data' in dist_coefs:
                dist_coefs = dist_coefs['data']

            # Reshape camera matrix to 3x3
            self.camera_matrix = np.array(cam_mat, dtype=np.float32).reshape(3, 3)
            self.distortion_coefs = np.array(dist_coefs, dtype=np.float32)
            
            print("Calibration loaded successfully.")
            
        except Exception as e:
            print(f"Failed to parse calibration file. Ensure it is valid YAML/JSON. Error: {e}")
            print("Falling back to default UndistortWrapper parameters...")
            # Fallback parameters from original wrapper
            self.camera_matrix = np.array([
                [392.5531005859375, 0.0, 326.73844408192963],
                [0.0, 439.04815673828125, 220.5653813603385],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
            
            self.distortion_coefs = np.array([
                -0.9111617456077904, 0.603501770314888, 
                -0.014333851834234601, 0.010320245199077559, 0.0
            ], dtype=np.float32)

    def _get_hook(self, name):
        """Internal hook to capture intermediate layer outputs."""
        def hook(model, input, output):
            self.outputs[name] = output.detach()
        return hook

    def _register_hooks(self):
        """Hook into Conv1 and the final Latent linear layer of the ImpalaCNN."""
        encoder_seq = self.actor.encoder.main
        
        self.hook_conv1 = encoder_seq[0].register_forward_hook(self._get_hook('Conv1'))
        self.hook_latent = encoder_seq[-1].register_forward_hook(self._get_hook('Latent'))

    def preprocess_image(self, img_path, is_real_world=False):
        """
        Replicates the environment observation pipeline:
        Undistort (if real) -> Crop -> Resize -> Grayscale -> Frame Stack (x4)
        """
        img_path = os.path.expanduser(img_path)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1. Undistort (Only for real bot)
        if is_real_world:
            h, w = img.shape[:2]
            new_camera_mat, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.distortion_coefs, (w, h), 1, (w, h))
            img = cv2.undistort(img, self.camera_matrix, self.distortion_coefs, None, new_camera_mat)
            v_crop_frac = 0.4
            h_crop_frac = 0.2
            left = int(w * h_crop_frac)
            right = int(w * (1.0 - h_crop_frac))
            top = int(h * v_crop_frac)
            img = img[top:h, left:right]

        # --- CAPTURE PRE-CROP IMAGE HERE ---
        pre_crop_img = img.copy()

        # 2. Crop top 1/3
        h, w = img.shape[:2]
        top_boundary = int(h / 3)
        img = img[top_boundary:h, 0:w]

        # 3. Resize to 84x84
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)

        # 4. Grayscale
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # Add channel dimension (1, 84, 84)
            img = np.expand_dims(img, axis=0)
        else:
            # Transpose to CHW (3, 84, 84)
            img = img.transpose(2, 0, 1)

        # 5. Frame Stack (Duplicate the image 4 times to mimic static initialization)
        num_frames = 4
        stacked_img = np.tile(img, (num_frames, 1, 1))

        # 6. Convert to PyTorch Tensor with batch dimension (1, Channels*Stack, 84, 84)
        tensor = torch.Tensor(stacked_img).unsqueeze(0).to(self.device)
        return tensor, pre_crop_img

    def compare(self, sim_img_path, real_img_path):
        """Runs both images through the network and visualizes the differences."""
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

        # Plot the new alignment check first
        self._plot_pre_crop_views(sim_pre_crop, real_pre_crop)

        # Followed by the network analysis
        self._plot_conv_comparison(sim_conv1, real_conv1)
        self._plot_latent_difference(sim_latent, real_latent)

    def _plot_pre_crop_views(self, sim_img, real_img):
        """Visualizes the images right before the standard RL 1/3 top crop is applied."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Pre-Crop View Alignment: Simulation vs. Real Bot", fontsize=14, fontweight='bold')
        
        ax1.imshow(sim_img)
        ax1.set_title("Simulation (Raw View)")
        ax1.axis('off')
        
        ax2.imshow(real_img)
        ax2.set_title("Real Bot (After Undistort & Hardware Crop)")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()

    def _plot_conv_comparison(self, sim_fm, real_fm, num_filters=16):
        """Plots the first N filters of Conv1 for both Sim and Real side-by-side."""
        num_filters = min(num_filters, sim_fm.shape[0])
        fig, axes = plt.subplots(num_filters, 2, figsize=(6, 2 * num_filters))
        fig.suptitle("Layer 1 (Conv1) Activations: Simulation vs. Real World", fontsize=14)

        for i in range(num_filters):
            # Simulation Column
            axes[i, 0].imshow(sim_fm[i], cmap='viridis')
            axes[i, 0].axis('off')
            if i == 0: axes[i, 0].set_title("Simulation")

            # Real Column
            axes[i, 1].imshow(real_fm[i], cmap='viridis')
            axes[i, 1].axis('off')
            if i == 0: axes[i, 1].set_title("Real World")

        plt.tight_layout()
        plt.show()

    def _plot_latent_difference(self, sim_latent, real_latent):
        """Visualizes the Latent vector representations and their absolute differences."""
        latent_diff = np.abs(sim_latent - real_latent)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [1, 2]})
        
        # Plot 1: Heatmap of raw latent vectors
        im_data = np.vstack([sim_latent, real_latent])
        im = ax1.imshow(im_data, aspect='auto', cmap='magma')
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(["Sim Latent", "Real Latent"])
        ax1.set_title(f"Final Latent Representation (Size: {sim_latent.shape[0]})")
        fig.colorbar(im, ax=ax1, fraction=0.02, pad=0.04)

        # Plot 2: Bar chart showing the magnitude of difference per neuron
        ax2.bar(range(len(latent_diff)), latent_diff, color='tomato', edgecolor='black', alpha=0.8)
        ax2.set_xlabel("Latent Feature Index")
        ax2.set_ylabel("Absolute Difference")
        ax2.set_title("Distribution Shift: | Sim_Latent - Real_Latent |")
        
        # Add a threshold line to easily spot massive deviations
        mean_diff = np.mean(latent_diff)
        ax2.axhline(mean_diff, color='blue', linestyle='--', label=f'Mean Diff ({mean_diff:.2f})')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Pointing to the specific calibration and image paths requested
    MODEL_PATH = "../rl_models/sac_vr2.cleanrl_model"
    CALIB_PATH = "artifacts/duckie_calib_data.txt"
    SIM_IMAGE_PATH = "screenshots/sim/sim1.png"
    REAL_IMAGE_PATH = "screenshots/realbot/real1.png"

    comparator = Sim2RealComparator(
        model_path=MODEL_PATH, 
        calib_path=CALIB_PATH,
        device="cuda",     
        grayscale=True     
    )
    
    comparator.compare(sim_img_path=SIM_IMAGE_PATH, real_img_path=REAL_IMAGE_PATH)
import os
import numpy as np
from PIL import Image
import gymnasium as gym
from utils.env_lunch import EnvLunch
from utils.wrappers import CropResizeWrapper, ImgWrapper

def save_diagnostic(obs, name, folder="captures"):
    os.makedirs(folder, exist_ok=True)
    
    # Ensure it's a numpy array
    obs = np.array(obs)
    
    # 1. Handle Wrapper Transposition: (C, H, W) -> (H, W, C)
    if obs.ndim == 3 and obs.shape[0] in [1, 3, 4, 12]:
        obs = obs.transpose(1, 2, 0)
    
    # 2. Handle Grayscale 'keep_dim=True': (84, 84, 1) -> (84, 84)
    # PIL needs a 2D array for grayscale 'L' mode
    if obs.ndim == 3 and obs.shape[2] == 1:
        obs = obs.squeeze(-1)
        
    try:
        # Convert to uint8 (0-255) if it isn't already
        img = Image.fromarray(obs.astype(np.uint8))
        img.save(f"{folder}/{name}.png")
        print(f"Successfully saved: {name}.png (Shape: {obs.shape})")
    except Exception as e:
        print(f"Failed to save {name}. Final Shape: {obs.shape}. Error: {e}")

def capture_at_corner_curve():
    # 1. Setup Launcher with Bezier curves enabled
    launcher = EnvLunch(run_name="curve_test", grayscale=True, domain_rand=False)
    
    # 2. Build the environment chain
    raw_env = launcher._create_base_env(seed=1)
    #raw_env.unwrapped.draw_curve = True # Enables Bezier plotting
    
    crop_env = CropResizeWrapper(raw_env, shape=(84, 84)) #
    gray_env = gym.wrappers.GrayscaleObservation(crop_env, keep_dim=True) #

    # 3. Position: At the end of tile (3,0) facing tile (4,0) curve
    # Your tile_size is 0.585
    ts = 0.585
    # Position: X = ~3.8 tiles, Z = ~0.2 tiles (centered in right lane)
    # ts = 0.585
    # X: 3.7 tiles (near the end of the straight tile)
    # Z: 0.75 tiles (the center of the right-hand driving lane)
    curve_facing_pos = np.array([ts * 3.7, 0, ts * 0.75]) 
    curve_facing_angle = 0  # Facing East toward the curve
    raw_env.reset()
    raw_env.unwrapped.cur_pos = curve_facing_pos
    raw_env.unwrapped.cur_angle = curve_facing_angle
    
    # 4. Sequential Capture
    # Raw with Bezier Curves
    obs_raw = raw_env.unwrapped.render_obs()
    save_diagnostic(obs_raw, "01_raw_facing_curve")
    
    # Cropped & Resized (84x84)
    obs_crop = crop_env.observation(obs_raw)
    save_diagnostic(obs_crop, "02_cropped_curve")
    
    # Grayscale (Final RL Input)
    obs_gray = gray_env.observation(obs_crop)
    save_diagnostic(obs_gray, "03_grayscale_curve")
    
    print(f"Captured images facing curve at {curve_facing_pos}. Check 'curve_diagnostics/'.")
    raw_env.close()

if __name__ == "__main__":
    capture_at_corner_curve()
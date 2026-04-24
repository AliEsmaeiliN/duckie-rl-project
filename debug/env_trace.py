import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from debug.rl_env_debug import DuckieOvalEnv
from debug.agent import DuckiebotAgent

def trace_debug_step(model_path=None):
    
    env = DuckieOvalEnv.create_wrapped(
        run_name="manual_control",
        motion_blur=False, 
        grayscale=True,
        frame_stack=4,
        domain_rand=False,
        dynamics_rand=False,
        distortion=False
    )
    
    obs, info = env.reset()
    
    
    if model_path:
        # Dummy stack for initialization
        action = np.array([0.5, 0.2]) # [v, omega]
        print(f" [RL Intent] v: {action[0]}, omega: {action[1]}")
    else:
        # Case B: Constant Action
        action = np.array([0.4, 0.1]) 
        print(f" [Constant Intent] v: {action[0]}, omega: {action[1]}")

    # Trace inside the ActionWrapper
    # Your ActionWrapper scales v by 0.8: action_ = [action[0] * 0.8, action[1]]
    scaled_action = [action[0] * 0.8, action[1]]
    print(f" [ActionWrapper Output] Scaled v: {scaled_action[0]}")

    # Trace into Simulator Physics (Simplified PWM mapping)
    # Simulator.py: self.wheelVels = action * self.robot_speed
    sim = env.unwrapped
    wheel_vels = np.array(scaled_action) * sim.robot_speed
    print(f" [Simulator WheelVels] Left/Right: {wheel_vels}")

    # --- OBSERVATION TRACE ---
    print("\n--- Observation Pipeline ---")
    
    # 1. Raw Simulator Output
    raw_obs = env.unwrapped.render_obs()
    print(f" 1. Raw Simulator Obs: {raw_obs.shape}, Range: [{raw_obs.min()}, {raw_obs.max()}]")

    # 2. After Crop & Resize (CropResizeWrapper)
    # Based on your code: crops top 1/3 and resizes to 84x84
    img = Image.fromarray(raw_obs)
    w, h = img.size
    cropped = img.crop((0, int(h*(1/3)), w, h))
    resized = cropped.resize((84, 84), Image.BILINEAR)
    proc_obs = np.array(resized)
    print(f" 2. Post-Crop/Resize:  {proc_obs.shape}")

    # 3. After Grayscale (GrayscaleObservation)
    if launcher.grayscale:
        gray_obs = np.dot(proc_obs[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        print(f" 3. Post-Grayscale:    {gray_obs.shape}")

    # 4. Final CNN Input (ImgWrapper + FrameStack)
    # This is what the 'step' function returns
    next_obs, reward, done, truncated, info = env.step(action)
    print(f" 4. Final CNN Input (Stacked): {next_obs.shape}")
    
    # Visualization
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(raw_obs)
    ax[0].set_title("Simulator Raw")
    # Show last frame of stack
    if launcher.grayscale:
        ax[1].imshow(next_obs[-1], cmap='gray')
    else:
        ax[1].imshow(next_obs[-3:].transpose(1, 2, 0))
    ax[1].set_title("CNN Input (Final Frame)")
    plt.show()

if __name__ == "__main__":
    trace_debug_step()
import os
import torch
import numpy as np
from PIL import Image
from rl_env_debug import DuckieOvalEnv

def save_image(arr, name, folder="captures/observation_pipeline"):
    """Helper to save various array formats as images."""
    os.makedirs(folder, exist_ok=True)
    
    # Handle (C, H, W) vs (H, W, C)
    if len(arr.shape) == 3:
        if arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
            # It's likely CHW (after ImgWrapper or FrameStack)
            # We take the first 3 channels for a viewable RGB
            arr = arr[:3, :, :].transpose(1, 2, 0)
    
    if arr.dtype == np.float32:
        arr = (arr * 255).astype(np.uint8)
        
    img = Image.fromarray(arr.squeeze())
    img.save(f"{folder}/{name}.png")

def debug_step():
    
    run_name = "debug_run"
    env = DuckieOvalEnv.create_wrapped(
        run_name=run_name,
        capture_video=False,
        motion_blur=False,
        grayscale=True,
        frame_stack=4,
        domain_rand=False,
        distortion=True
    )

    obs, info = env.reset(seed=42)
    
    constant_action = np.array([0.4, 0.2], dtype=np.float32)

    print(f"{'Step':<6} | {'Reward':<10} | {'Action [v, w]':<15}")
    print("-" * 40)

    for i in range(50):
    
        next_obs, reward, done, truncated, misc = env.step(constant_action)
        
        print(f"{i:<6} | {reward:<10.4f} | {str(constant_action):<15}")

        current_obs = env.unwrapped.render_obs()
        if i % 20 == 0:
            save_image(current_obs, f"step_{int(i/20)}_0_Raw_Simulator")

        wrappers = []
        curr = env
        while hasattr(curr, 'env'):
            wrappers.append(curr)
            curr = curr.env
        if i % 20 == 0:
            for idx, wrapper in enumerate(reversed(wrappers)):
                wrapper_name = wrapper.__class__.__name__
                if hasattr(wrapper, 'observation'):
                    try:
                        current_obs = wrapper.observation(current_obs)
                        save_image(current_obs, f"step_{i}_{idx+1}_{wrapper_name}")
                    except Exception as e:
                        frames = [next_obs[t] for t in range(next_obs.shape[0])]
                        combined_stack = np.concatenate(frames, axis=1) 
                        save_image(combined_stack, f"step_{i}_{idx+1}_{wrapper_name}")
        if done:
            break

    env.close()
    print("\nDebug complete. Check the 'captures' folder for images.")

if __name__ == "__main__":
    debug_step()
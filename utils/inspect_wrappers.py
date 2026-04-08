import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from utils.env_lunch import EnvLunch
import cv2
import collections

def wrapper_pipeline(env, observation_raw, grayscale=True):
    
    print("\n--- Starting Wrapper-based Pipeline ---")

    raw_env = env
    
    obs_stages = {}

    obs_raw = observation_raw
    obs_stages["0. Raw Simulator (480x640x3)"] = obs_raw

    from utils.wrappers import ResizeWrapper
    env_resized = ResizeWrapper(raw_env, shape=(120, 160, 3)) # Ensure 120x160 base
    obs_resized = env_resized.observation(obs_raw)
    obs_stages["1. Resized to 120*160"] = obs_resized

    from utils.wrappers import CropResizeWrapper
    env_crop = CropResizeWrapper(env_resized, shape=(84, 84))
    obs_crop = env_crop.observation(obs_resized)
    obs_stages["2. Crop & Resize (84x84x3)"] = obs_crop

    if grayscale:
        env_gray = gym.wrappers.GrayscaleObservation(env_crop, keep_dim=True)
        obs_gray = env_gray.observation(obs_crop)
    else:
        obs_gray = obs_crop

    obs_stages["3. Grayscale (84x84x1)"] = obs_gray

    from utils.wrappers import ImgWrapper
    env_img = ImgWrapper(env_gray)
    obs_img = env_img.observation(obs_gray)
    obs_stages["4. Transposed (1x84x84)"] = obs_img

    env_final = luncher._apply_wrappers(raw_env)
    
    for _ in range(5): 
        obs_final, _, _, _, _ = env_final.step(np.array([1, 0.0]))
    
    # 3. Visualization logic
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle("Duckie-RL Perception Pipeline", fontsize=16)


    # Plot static stages
    stages = list(obs_stages.keys())
    for i, title in enumerate(stages):
        ax = plt.subplot(3, 3, i + 1)
        data = obs_stages[title]
        
        if data.shape[0] == 1: # CHW
            plt.imshow(data[0], cmap='gray')
        elif data.shape[-1] == 1: # HWC
            plt.imshow(data[..., 0], cmap='gray')
        else:
            plt.imshow(data)
        
        plt.title(title)
        plt.axis('off')

    for f in range(4):
        ax = plt.subplot(3, 4, 9 + f)
        plt.imshow(obs_final[f], cmap='gray')
        plt.title(f"Final Stack: Frame {f+1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show(block=False) # Allows the script to continue
    plt.pause(1)        

def manual_vision_pipeline(env, observation_raw, grayscale=True):
    """
    Manual recreation of the Duckie-RL vision pipeline using OpenCV.
    """
    print("\n--- Starting Manual OpenCV Pipeline ---")

    frame_stack = 4
    frames = collections.deque([np.zeros((1, 84, 84), dtype=np.uint8) for _ in range(frame_stack)], maxlen=frame_stack)

    raw_env = env
    obs_raw = observation_raw
    
    cv2.imshow("0. Raw Simulator", cv2.cvtColor(obs_raw, cv2.COLOR_RGB2BGR))
    
    img = cv2.resize(obs_raw, (160, 120), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("1. Resized Base", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    h, w = img.shape[:2]
    top_boundary = int(h * (1/4)) 
    img = img[top_boundary:h, 0:w]
    
    img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("2. Crop & Resized", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    if grayscale:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.imshow("3. Grayscale", img_gray)
        cv2.waitKey(0) 
    else:
        cv2.imshow("3. RGB Ready", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)

    # ---- test for frame stacking
    cv2.waitKey(0)
    for step in range(5):
        img = cv2.resize(obs_raw, (160, 120), interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]
        top_boundary = int(h * (1/4)) 
        img = img[top_boundary:h, 0:w]
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_LINEAR)
        
        if grayscale:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_processed = img_gray[np.newaxis, :, :]
        else:
            img_processed = img.transpose(2, 0, 1)

        frames.append(img_processed)
        stack_display = np.hstack([f[0] for f in frames])
        
        cv2.imshow("Manual Stack (Frames 1-4)", stack_display)
        print(f"Step {step} visualization active. Press any key for next step...")
        cv2.waitKey(500) # Auto-advance every 0.5s or press key

        obs_raw, _, _, _, _ = env.step(np.array([0.5, 0.5]))
            
    final_manual_stack = np.concatenate(list(frames), axis=0)
    print(f"Final Tensor Shape for RL: {final_manual_stack.shape}")
    cv2.waitKey(0)

    
if __name__ == "__main__":

    luncher = EnvLunch(
        run_name="inspection",
        grayscale=True,
        frame_stack=4,
        img_shape=(84, 84),
        domain_rand=False,
    )
    env = luncher._create_base_env(seed=42)
    obs_raw, _ = env.reset()

    wrapper_pipeline(env, obs_raw)
    cv2.waitKey(0)
    manual_vision_pipeline(env, obs_raw)
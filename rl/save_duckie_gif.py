import os
import torch
import numpy as np
import gymnasium as gym
import imageio
from .sac_continuous_action import Actor, make_env

def generate_gif():
    # --- CONFIG ---
    MODEL_PATH = "runs/oval_loop__sac_continuous_action__1__1771938634/models/sac_step_step_50000.cleanrl_model"
    OUTPUT_NAME = "duckiebot_eval.gif"
    MAX_STEPS = 200  # Number of frames to capture
    SEED = 42        # Changed seed to try and avoid "Invalid Pose"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- ENV SETUP ---
    # We use make_env()() to get a single environment instance
    # We don't need SyncVectorEnv here to keep frame extraction simple
    env_func = make_env(seed=SEED, idx=0, capture_video=False, run_name="gif_gen")
    env = env_func()
    
    # --- MODEL SETUP ---
    # Manually add the attributes the Actor class looks for
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space
    
    actor = Actor(env).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()

    # --- RECORDING ---
    frames = []
    obs, _ = env.reset(seed=SEED)
    
    print(f"Recording {MAX_STEPS} steps...")
    for i in range(MAX_STEPS):
        # 1. Capture the high-res frame from the simulator
        # This bypasses the window and captures the raw RGB array
        frame = env.unwrapped.render(mode="rgb_array")
        frames.append(frame)

        # 2. Get action from model
        with torch.no_grad():
            # Add batch and channel dimensions if needed
            obs_t = torch.Tensor(obs).unsqueeze(0).to(device)
            _, _, actions = actor.get_action(obs_t)
            action = actions.cpu().numpy()[0]

        # 3. Step
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 20 == 0:
            print(f"Step {i}/{MAX_STEPS} captured.")

        if terminated or truncated:
            print("Episode ended early.")
            break

    # --- SAVE ---
    print(f"Compiling GIF: {OUTPUT_NAME}...")
    imageio.mimsave(OUTPUT_NAME, frames, fps=30)
    print("Done! You can now download and view the GIF.")
    env.close()

if __name__ == "__main__":
    generate_gif()
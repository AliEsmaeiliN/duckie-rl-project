import torch
import numpy as np
import gymnasium as gym
import argparse
from sac_continuous_action import Actor, make_env

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SAC Agent in Duckietown")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the .cleanrl_model file")
    parser.add_argument("--env-id", type=str, default="oval_loop",
                        help="The name of the Duckietown map")
    parser.add_argument("--num-episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", default=False,
                        help="Whether to render the environment")
    parser.add_argument("--capture-video", type=bool, default=True,
                        help="Capture video of the evaluation episodes")
    return parser.parse_args()

def evaluate():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Recreate the environment exactly as it was during training
    # Note: make_env returns a thunk, so we call it and then wrap it if needed
    env_func = make_env(seed=args.num_episodes, idx=0, capture_video=args.capture_video, run_name="eval")
    env = env_func()
    
    # 2. Instantiate the Actor
    # We use env.single_observation_space because it's a VectorEnv in training, 
    # but here we might need to simulate that or adjust for a single env.
    # To keep it simple and compatible with your Actor class:
    class DummyEnv:
        def __init__(self, env):
            self.single_observation_space = env.observation_space
            self.single_action_space = env.action_space
    
    dummy = DummyEnv(env)
    actor = Actor(dummy).to(device)

    # 3. Load the weights
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()

    # 4. Evaluation Loop
    for episode in range(args.num_episodes):
        obs, info = env.reset()
        done = False
        episodic_reward = 0
        
        while not done:
            # Prepare observation: (C, H, W) -> (1, C, H, W)
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Use mean_action for deterministic evaluation
                _, _, action = actor.get_action(obs_tensor)
            
            action = action.cpu().numpy().reshape(-1)
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episodic_reward += reward
            
            if args.render:
                env.render()

        print(f"Episode {episode + 1}: Reward = {episodic_reward:.2f}")

    env.close()

if __name__ == "__main__":
    evaluate()
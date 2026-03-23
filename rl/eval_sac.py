import os 
import torch
import wandb
import numpy as np
import gymnasium as gym
import argparse
from sac_continuous_action import Actor
from utils.env_lunch import make_env


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SAC Agent in Duckietown")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the .cleanrl_model file")
    parser.add_argument("--env-id", type=str, default=None,
                        help="The name of the Duckietown map")
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", default=False,
                        help="Whether to render the environment")
    parser.add_argument("--capture-video", type=bool, default=True,
                        help="Capture video of the evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=1500,
                        help="Maximum number of steps for each episode" )
    parser.add_argument("--grayscale", type=bool, default=True,
                        help="Maximum number of steps for each episode" )
    parser.add_argument("--local", type=bool, default=False,
                        help="Whether the model path is the wandb artifact or local")
    return parser.parse_args()

def evaluate():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_func = make_env(seed=42, idx=0, capture_video=args.capture_video, run_name="eval", max_steps=args.max_steps, grayscale=args.grayscale)
    env = env_func()
    path_parts = args.model_path.split('/')
    run_name_short = path_parts[-1].split(':')[0] if not args.local else os.path.basename(args.model_path)
    if args.capture_video:
        video_folder = f"videos/{run_name_short}"
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)

        env = gym.wrappers.RecordVideo(
            env, 
            video_folder, 
            # it tells the wrapper to record if episode_id >= 0
            episode_trigger=lambda x: True 
        )
        print(f"Recording videos to {video_folder}")
    
    # We use env.single_observation_space because it's a VectorEnv in training, 
    # To keep it simple and compatible with Actor class:
    class DummyEnv:
        def __init__(self, env):
            self.single_observation_space = env.observation_space
            self.single_action_space = env.action_space
    
    dummy = DummyEnv(env)
    actor = Actor(dummy, grayscale=args.grayscale).to(device)

    # 3. Load the weights
    if args.local: 
        model_path = args.model_path
    else:
        print("Downloading Artifact")
        api = wandb.Api()
        artifact = api.artifact(args.model_path)
        artifact_dir = artifact.download()
        model_path = f"{artifact_dir}/sac_Final.cleanrl_model"
    
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()
    all_rewards = []
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

        all_rewards.append(episodic_reward)
        print(f"Episode {episode + 1}: Reward = {episodic_reward:.2f}")
    
    print(f"\n--- Final Evaluation Over {args.num_episodes} Episodes ---")
    print(f"Mean Reward: {np.mean(all_rewards):.2f}")
    print(f"Std Deviation: {np.std(all_rewards):.2f}")

    env.close()

if __name__ == "__main__":
    evaluate()
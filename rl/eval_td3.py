import os 
import torch
import wandb
import numpy as np
import gymnasium as gym
import argparse
from td3_continuous_action import Actor
from utils.env_lunch import EnvLunch


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TD3 Agent in Duckietown")
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

    if args.local: 
        model_path = args.model_path
    else:
        print("Downloading Artifact")
        api = wandb.Api()
        artifact = api.artifact(args.model_path)
        artifact_dir = artifact.download()
        model_path = f"{artifact_dir}/td3_Final.cleanrl_model"
    
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    env_id = args.env_id or checkpoint.get("env_id", "oval_loop")
    grayscale = args.grayscale if args.grayscale is not None else checkpoint.get("grayscale", True)

    # Handle randomization toggles (env_params)
    if "env_params" in checkpoint:
        print("Using the parameters inside the checkpoint")
        sim_params = checkpoint["env_params"]
    else:
        print("Could not find the metadata")
        sim_params = {
            "domain_rand": checkpoint.get("domain_rand", False),
            "distortion": checkpoint.get("distortion", False),
            "dynamics_rand": checkpoint.get("dynamics_rand", False),
            "camera_rand": checkpoint.get("camera_rand", False),
        }
    
    print(f"--- Metadata Extracted ---")
    print(f"Map: {env_id} | Grayscale: {grayscale}")
    print(f"Randomizations: {sim_params}")

    env_luncher = EnvLunch(
        run_name="eval",
        max_steps=3000,
        grayscale=args.grayscale,
        **sim_params
    )
    env_func = env_luncher.make_env_fn(
        seed=1, 
        idx=0,
        capture_video=True,
    )
    env = env_func()
    
    path_parts = args.model_path.split('/')
    run_name_short = path_parts[-1].split(':')[0] if not args.local else os.path.basename(args.model_path)
    if args.capture_video:
        video_folder = f"videos/TD3/{run_name_short}"
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)

        env = gym.wrappers.RecordVideo(
            env, 
            video_folder, 
            # it tells the wrapper to record if episode_id >= 0
            episode_trigger=lambda x: True 
        )
        print(f"Recording videos to {video_folder}")
    
    class DummyEnv:
        def __init__(self, env):
            self.single_observation_space = env.observation_space
            self.single_action_space = env.action_space
    
    dummy = DummyEnv(env)
    actor = Actor(dummy, grayscale=args.grayscale).to(device)

    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()


    all_rewards = []
    for episode in range(args.num_episodes):
        obs, info = env.reset()
        done = False
        episodic_reward = 0
        
        while not done:
            # Prepare observation: (C, H, W) -> (1, C, H, W)
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action = actor(obs_tensor)
            
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
import matplotlib.pyplot as plt
import sys
import os
import glob
import torch
import numpy as np
import wandb
from utils.rl_env import DuckieOvalEnv

def plot_model_input(s_obs, global_step):
    # Take the first environment's observation from the batch
    # s_obs shape is (Batch, 12, 120, 160)
    sample_obs = s_obs[0].cpu().numpy() 

    # Extract the first 3 channels (the most recent RGB frame)
    first_frame = sample_obs[0:3, :, :].transpose(1, 2, 0)

    plt.imshow(first_frame)
    plt.title(f"Input to Model - Step {global_step}")
    plt.show() 

def save_models(actor, qf1, qf2, step, run_name, args, env_params, suffix=""):
    
    model_dir = f"runs/{run_name}/models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    main_script = sys.argv[0].lower()
    if "td3" in main_script:
        algo_prefix = "td3"
    elif "sac" in main_script:
        algo_prefix = "sac"
    else:
        algo_prefix = "model" # Fallback

    label = suffix if suffix else "latest_step"
    model_path = f"{model_dir}/{algo_prefix}_{label}.cleanrl_model"

    torch.save({
        'actor_state_dict': actor.state_dict(),
        'qf1_state_dict': qf1.state_dict(),
        'qf2_state_dict': qf2.state_dict(),
        'global_step': step,
        'env_id': args.env_id,
        'run_notes': args.run_notes,
        'env_params': env_params,
    }, model_path)

    if wandb.run is not None:
        artifact_name = f"{run_name}_{label}"
        artifact = wandb.Artifact(name=artifact_name, type="model")
        artifact.add_file(model_path)      
        artifact.metadata = {"global_step": step, "suffix": suffix, "env_id": args.env_id, **env_params}
        
        wandb.log_artifact(artifact)
    
    print(f"Saved: {model_path} | Metadata: {args.env_id}, Grayscale={args.grayscale}")

def evaluate_policy(actor, args, device, algo_name, run_name = "run_name", num_episodes=10, **env_params):
    print(f"\n--- Starting Final Evaluation: {num_episodes} Episodes ---")
    actor.eval()

    custom_run_name = f"{algo_name}/{run_name}"
    
    # Create a separate evaluation environment
    eval_env = DuckieOvalEnv.create_wrapped(
        run_name=custom_run_name,
        motion_blur=args.motion_blur, 
        grayscale=True,
        frame_stack=4,
        capture_video = True,
        render_mode = "rgb_array",
        domain_rand=args.domain_rand,
        dynamics_rand=args.dynamics_rand,
        distortion=args.distortion
    )
    

    all_rewards = []
    all_lengths = []
    for ep in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        episodic_reward = 0
        episodic_length = 0
        
        while not done:
            with torch.no_grad():
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
                if hasattr(actor, "get_action"):
                    _, _, action = actor.get_action(obs_tensor) # Use mean_action for eval
                else:
                    action = actor(obs_tensor) #TD3 actor returns action directly
            
                action = action.cpu().numpy().reshape(-1)
            
            next_obs, reward, terminated, truncated, _ = eval_env.step(action)
            
            obs = next_obs
            episodic_reward += reward
            episodic_length += 1
            done = terminated or truncated

        all_rewards.append(episodic_reward)
        all_lengths.append(episodic_length)
        print(f"Eval Episode {ep+1}: Reward = {episodic_reward:.2f}")

    avg_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    print(f"Evaluation Average Reward: {avg_reward:.2f}, with Std: {std_reward}")
    eval_env.close()
    # Log to WandB
    if args.track:
        columns = ["episode", "reward", "length"]
        data = [[i + 1, all_rewards[i], all_lengths[i]] for i in range(len(all_rewards))]
        table = wandb.Table(data=data, columns=columns)
        metrics = {
            "final_eval/episode_table": table # Scatter data: Episode vs Reward
            }
        
        import time
        time.sleep(1)
        best_idx, worst_idx = np.argmax(all_rewards), np.argmin(all_rewards)

        print(f"best: {best_idx}, worst: {worst_idx}")

        def get_video_path(idx):
            return f"videos/{custom_run_name}/rl-video-episode-{idx}.mp4"

        best_path = get_video_path(best_idx)
        if os.path.exists(best_path):
            metrics[f"final_eval/best_video"] = wandb.Video(best_path, format="mp4", caption=f"Best Run (Reward: {all_rewards[best_idx]:.2f})")
    
        worst_path = get_video_path(worst_idx)
        if os.path.exists(worst_path) and worst_idx != best_idx:
            metrics[f"final_eval/worst_video"] = wandb.Video(worst_path, format="mp4", caption=f"Worst Run (Reward: {all_rewards[worst_idx]:.2f})")
        
        wandb.log(metrics)
        
    actor.train() # just in case
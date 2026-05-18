import matplotlib.pyplot as plt
import sys
import os
import glob
import torch
import numpy as np
import wandb
from utils.rl_env import DuckieOvalEnv

def plot_model_input(s_obs, global_step):
    sample_obs = s_obs[0].cpu().numpy() 
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

def evaluate_policy(eval_env, actor, args, device, is_interval=False, global_step=0, best_reward=-float('inf'), num_episodes=10):
    """
    Unified policy evaluation method for both intermediate training intervals and final verification.
    Returns: (avg_reward, std_reward, is_best)
    """
    eval_type = "Interval" if is_interval else "Final"
    print(f"\n--- Starting {eval_type} Evaluation [Step {global_step}]: {num_episodes} Episodes ---")
    actor.eval()

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

    avg_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    print(f"--- {eval_type} Evaluation Complete | Average Reward: {avg_reward:.2f} (Std: {std_reward:.2f}) ---")

    is_best = avg_reward > best_reward
    prefix = f"interval_eval" if is_interval else "final_eval"

    # Log to WandB
    if args.track:
        import time
        metrics = {
            f"{prefix}/avg_reward": avg_reward,
            f"{prefix}/std_reward": std_reward,
            f"{prefix}/avg_length": np.mean(all_lengths),
            "global_step": global_step
        }
        wandb.log(metrics)
        
    actor.train() 
    return avg_reward, std_reward, is_best

def log_pid_metrics(pid_info: dict, global_step: int, prefix: str = "pid") -> None:
    """
    Logs the most useful subset of pid_stabilizer info to W&B.
    """
    if pid_info is None:
        return
 
    wandb.log({
        # Core tracking quality
        f"{prefix}/omega_error":        abs(pid_info.get("e_omega", 0)),
        f"{prefix}/v_error":            abs(pid_info.get("e_v", 0)),
 
        # Jerk reduction (key stabilization metric)
        f"{prefix}/omega_jerk_raw":     pid_info.get("omega_jerk_raw", 0),
        f"{prefix}/omega_jerk_smooth":  pid_info.get("omega_jerk_smooth", 0),
        f"{prefix}/jerk_reduction_pct": pid_info.get("jerk_reduction_pct", 0),
 
        # Action comparison
        f"{prefix}/omega_rl":           pid_info.get("omega_rl", 0),
        f"{prefix}/omega_out":          pid_info.get("omega_out", 0),
        f"{prefix}/v_rl":               pid_info.get("v_rl", 0),
        f"{prefix}/v_out":              pid_info.get("v_out", 0),
 
        # PID internals (useful for tuning)
        f"{prefix}/steer_P":            pid_info.get("steer_P", 0),
        f"{prefix}/steer_I":            pid_info.get("steer_I", 0),
        f"{prefix}/steer_D":            pid_info.get("steer_D", 0),
        f"{prefix}/steer_integral":     pid_info.get("steer_integral", 0),
 
        "global_step": global_step,
    })

def evaluate_policy_interval(actor, qf1, qf2, global_step, best_reward, run_name, args, env_params, device, num_episodes=10):
    """
    Runs a standalone evaluation over a specified number of episodes,
    logs results to TensorBoard/W&B, and checkpoints the model if it's the best so far.
    """
    print(f"\n--- [Step {global_step}] Starting Periodic Evaluation: {num_episodes} Episodes ---")
    actor.eval()

    eval_env = DuckieOvalEnv.create_wrapped(
        run_name=f"sac_interval_eval_{run_name}",
        grayscale=args.grayscale,
        frame_stack=4,
        capture_video=False,
        render_mode="rgb_array",
        use_pid=args.pid,
        domain_rand=args.domain_rand,
        dynamics_rand=args.dynamics_rand,
        distortion=args.distortion,
        latency_rand=args.action_latency,
        direction=args.direction
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
                # Ensure deterministic execution via mean_action
                _, _, action = actor.get_action(obs_tensor) 
                action = action.cpu().numpy().reshape(-1)
            
            next_obs, reward, terminated, truncated, _ = eval_env.step(action)
            obs = next_obs
            episodic_reward += reward
            episodic_length += 1
            done = terminated or truncated

        all_rewards.append(episodic_reward)
        all_lengths.append(episodic_length)

    avg_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    print(f"[Step {global_step}] Interval Eval Average Reward: {avg_reward:.2f} (Std: {std_reward:.2f})")

    # Clean up evaluation window/context
    eval_env.close()

    # Log results to TensorBoard
    writer.add_scalar("eval/avg_reward", avg_reward, global_step)
    writer.add_scalar("eval/std_reward", std_reward, global_step)
    writer.add_scalar("eval/avg_length", np.mean(all_lengths), global_step)

    if args.track:
        import wandb
        wandb.log({
            "eval/avg_reward": avg_reward,
            "eval/std_reward": std_reward,
            "eval/avg_length": np.mean(all_lengths),
            "global_step": global_step
        })

    # Save tracking logic for finding the global maximum
    if avg_reward > best_reward:
        print(f" New Peak Performance Detected! {best_reward:.2f} ---> {avg_reward:.2f}. Saving model...")
        best_reward = avg_reward
        save_models(
            actor=actor,
            qf1=qf1,
            qf2=qf2,
            step=global_step,
            run_name=run_name,
            args=args,
            env_params=env_params,
            suffix=f"v{args.version}_BEST"
        )

    # Restore policy back to alternative operational mode
    actor.train()
    
    return best_reward

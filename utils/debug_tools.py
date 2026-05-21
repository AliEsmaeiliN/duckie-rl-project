import sys
import os
import torch
import numpy as np
import wandb
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



_eval_history = []
_best_trajectory_payload = {}

def get_speed_gradient_color(norm_speed):
    """
    Computes a sleek Neon Cyberpunk BGR color for a given normalized speed [0, 1].
    Transitions from Cyan (slow) -> Hot Magenta (medium) -> Vibrant Orange-Red (fast).
    """
    # Clip just in case
    norm_speed = np.clip(norm_speed, 0.0, 1.0)
    
    # 0.0: Neon Cyan (BGR: 255, 220, 0)
    # 0.5: Hot Pink/Magenta (BGR: 180, 0, 240)
    # 1.0: Bright Orange-Red (BGR: 0, 100, 255)
    if norm_speed <= 0.5:
        t = norm_speed / 0.5
        b = int((1.0 - t) * 255 + t * 180)
        g = int((1.0 - t) * 220 + t * 0)
        r = int((1.0 - t) * 0 + t * 240)
    else:
        t = (norm_speed - 0.5) / 0.5
        b = int((1.0 - t) * 180 + t * 0)
        g = int((1.0 - t) * 0 + t * 100)
        r = int((1.0 - t) * 240 + t * 255)
        
    return (b, g, r)

def log_distribution_plot(history, global_step=0, extra_payload=None):
    """
    Renders the Reward Distribution and Risk-Adjusted trend plot from memory
    and logs the figure to W&B.
    """
    steps = [h["step"] for h in history]
    avg_rewards = [h["avg_reward"] for h in history]
    std_rewards = [h["std_reward"] for h in history]
    risk_scores = [h["risk_adjusted_score"] for h in history]
    success_rates = [h["success_rate"] for h in history]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Left Axis: Reward Stability
    color = '#1f77b4'
    ax1.errorbar(steps, avg_rewards, yerr=std_rewards, fmt='o', color=color, 
                 ecolor='#a1c9f4', elinewidth=2, capsize=5, markersize=6, label='Mean Reward \u00b1 Std')
    ax1.plot(steps, risk_scores, 'D-', color='#ff7f0e', label='Risk-Adjusted Score')
    ax1.set_xlabel('Global Training Steps')
    ax1.set_ylabel('Episodic Return', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Right Axis: Success Rate
    ax2 = ax1.twinx()
    color2 = '#2ca02c'
    ax2.plot(steps, success_rates, 'o-.', color=color2, linewidth=2, label='Success Rate')
    ax2.set_ylabel('Success Rate (Completed/Total)', color=color2)
    ax2.set_ylim(-1, 105) # Success rate is 0 to 100
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('Evaluation: Reward Stability vs. Success Rate', fontsize=12, fontweight='bold', pad=15)
    
    # Combined Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True)
    
    ax1.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    log_dict = {"final_eval/performance_summary": wandb.Image(fig)}

    table = wandb.Table(columns=["step", "avg_reward", "std_reward", "risk_adjusted_score", "success_rate"])
    for h in history:
        table.add_data(h["step"], h["avg_reward"], h["std_reward"], h["risk_adjusted_score"], h["success_rate"])
    log_dict["final_eval/performance_data"] = table

    if extra_payload:
        log_dict.update(extra_payload)

    print(" Logging the final Evaluations Table and Plot ")
    wandb.log(log_dict, step=global_step)
    plt.close(fig)

def generate_trajectory(eval_env, actor, args, device, global_step, prefix="interval_eval"):
    """
    Runs two dedicated visualization episodes (one CW and one CCW) using the deterministic
    policy, generates overlay trajectory maps, and uploads interactive charts/tables to W&B.
    """
    if not (args.track and wandb.run is not None):
        return

    sim = eval_env.unwrapped
    original_direction = getattr(sim, "direction", "mixed")
    log_payload = {}
    
    for direction_key in ["CW", "CCW"]:
        print(f"    Running dedicated {direction_key} trajectory tracking episode...")
        sim.direction = direction_key
        
        traj_obs, _ = eval_env.reset()
        traj_done = False
        
        traj_x = []
        traj_z = []
        traj_v = []
        traj_omega = []
        traj_steps = []
        traj_speed = []
        step_count = 0

        traj_heading_err = []
        traj_cte = []
        traj_classical_omega = []
        
        while not traj_done:
            pos = sim.cur_pos  
            angle = sim.cur_angle

            traj_x.append(float(pos[0]))
            traj_z.append(float(pos[2]))
            traj_speed.append(float(sim.speed))
            traj_steps.append(step_count)

            try:
                lp = sim.get_lane_pos2(pos, angle)
                heading_err = lp.angle_rad
                cte = lp.dist
            except Exception:
                heading_err = 0.0
                cte = 0.0

            # --- Mock Classical PD Controller ---
            kp = 3.5  
            kd = 8.0 
            
            classical_omega = -kp * heading_err - kd * cte
            classical_omega = np.clip(classical_omega, -1.0, 1.0) # Match action space bounds
            
            traj_heading_err.append(heading_err)
            traj_cte.append(cte)
            traj_classical_omega.append(classical_omega)
            
            with torch.no_grad():
                obs_tensor = torch.Tensor(traj_obs).unsqueeze(0).to(device)
                if hasattr(actor, "get_action"):
                    _, _, action = actor.get_action(obs_tensor)
                else:
                    action = actor(obs_tensor)
                action = action.cpu().numpy().reshape(-1)
            
            traj_v.append(float(action[0]))
            traj_omega.append(float(action[1]))
            
            next_obs, reward, terminated, truncated, _ = eval_env.step(action)
            traj_obs = next_obs
            traj_done = terminated or truncated
            step_count += 1

        trajectory_image_logged = None
        try:
            top_down_map = sim._render_img(
                width=800,
                height=600,
                multi_fbo=sim.multi_fbo_human,
                final_fbo=sim.final_fbo_human,
                img_array=sim.img_array_human,
                top_down=True
            )
            
            a = (sim.grid_width * sim.road_tile_size) / 2.0
            b = (sim.grid_height * sim.road_tile_size) / 2.0
            H_to_fit = max(a, b) + 0.1
            aspect = 800.0 / 600.0

            z_min, z_max = b - H_to_fit, b + H_to_fit
            x_min, x_max = a - H_to_fit * aspect, a + H_to_fit * aspect

            overlay_img = top_down_map.copy()
            pixel_points = []
            for x, z in zip(traj_x, traj_z):
                px = int((x - x_min) / (x_max - x_min) * 800)
                py = int((z - z_min) / (z_max - z_min) * 600)
                px = np.clip(px, 0, 799)
                py = np.clip(py, 0, 599)
                pixel_points.append([px, py])

            if len(pixel_points) > 1:
                max_observed_speed = max(traj_speed) if len(traj_speed) > 0 else 1.2
                min_observed_speed = min(traj_speed) if len(traj_speed) > 0 else 0.0
                speed_range = max_observed_speed - min_observed_speed if max_observed_speed != min_observed_speed else 1.0

                for i in range(len(pixel_points) - 1):
                    p1 = pixel_points[i]
                    p2 = pixel_points[i+1]
                    current_speed = traj_speed[i]
                    
                    norm_speed = (current_speed - min_observed_speed) / speed_range if speed_range > 0 else 0.5
                    
                    color = get_speed_gradient_color(norm_speed)
                    cv2.line(overlay_img, tuple(p1), tuple(p2), color=color, thickness=1, lineType=cv2.LINE_AA)

                # Glowing Start Marker (Green inner, white border)
                cv2.circle(overlay_img, tuple(pixel_points[0]), 5, (0, 255, 0), -1, lineType=cv2.LINE_AA)
                cv2.circle(overlay_img, tuple(pixel_points[0]), 7, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                
                # Glowing End Marker (Red inner, white border)
                cv2.circle(overlay_img, tuple(pixel_points[-1]), 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(overlay_img, tuple(pixel_points[-1]), 7, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            
            trajectory_image_logged = wandb.Image(overlay_img, caption=f"2D Trajectory Map ({direction_key}) - Step {global_step}")
        except Exception as e:
            print(f"[debug_tools] Warning: Could not generate top-down map overlay for {direction_key}: {e}")

        #  Generate Matplotlib Waveform Comparison ---
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        ax1.plot(traj_steps, traj_v, label='Agent Throttle (v)', color='#1f77b4', linewidth=2)
        ax1.set_ylabel('Velocity')
        ax1.set_title(f'Action Waveforms vs Classical Baseline ({direction_key}) - Step {global_step}', fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.5)

        ax2.plot(traj_steps, traj_omega, label='RL Agent Steering (\u03c9)', color='#ff7f0e', linewidth=2)
        ax2.plot(traj_steps, traj_classical_omega, label='Classical PD Steering (\u03c9)', color='#2ca02c', linestyle='--', alpha=0.8)
        ax2.set_ylabel('Steering (\u03c9)')
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle='--', alpha=0.5)

        ax3.plot(traj_steps, traj_heading_err, label='Heading Error (rad)', color='#9467bd', linewidth=1.5)
        ax3.plot(traj_steps, traj_cte, label='Cross-Track Error (m)', color='#d62728', linewidth=1.5)
        ax3.set_ylabel('Error Margin')
        ax3.set_xlabel('Simulation Decision Steps')
        ax3.legend(loc='upper right')
        ax3.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        waveform_image_logged = wandb.Image(fig, caption=f"RL vs Classical Waveforms ({direction_key}) - Step {global_step}")
        plt.close(fig)


        log_payload[f"{prefix}/best_milestone_2d_trajectory_image_{direction_key}"] = trajectory_image_logged
        log_payload[f"{prefix}/best_milestone_action_waveforms_{direction_key}"] = waveform_image_logged
    
    wandb.log(log_payload, step=global_step)
    #  restore original environment state
    sim.direction = original_direction

    return log_payload

def evaluate_policy(eval_env, seed, actor, args, device, is_interval=False, global_step=0, best_reward=-float('inf'), num_episodes=10):
    """
    Unified policy evaluation method for both intermediate training intervals and final verification.
    Returns: (avg_reward, std_reward, is_best)
    """

    global _best_trajectory_payload

    if global_step >= args.total_timesteps:
        is_interval = False
        print("Final Evaluation test and Logging the Data ")

    eval_type = "Interval" if is_interval else "Final"
    print(f"\n--- Starting {eval_type} Evaluation [Step {global_step}]: {num_episodes} Episodes ---")
    actor.eval()

    all_rewards = []
    completed_episodes = 0
    eval_env.reset(seed=seed)
    for ep in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        episodic_reward = 0
        
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
            done = terminated or truncated

        all_rewards.append(episodic_reward)
        if truncated and not terminated:
            completed_episodes += 1

    avg_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    success_rate = completed_episodes / num_episodes * 100
    print(f"--- {eval_type} Evaluation Complete | Average Reward: {avg_reward:.2f} (Std: {std_reward:.2f}) | Success Rate: {success_rate:.2f} ---")

    beta = 0.5
    risk_adjusted_score = avg_reward - (beta * std_reward)
    print(f"    Risk-Adjusted Score: {risk_adjusted_score:.2f}")

    is_best = risk_adjusted_score > best_reward
    prefix = f"interval_eval" if is_interval else "final_eval"

    # Log to WandB
    if args.track:
        _eval_history.append({
            "step": int(global_step),
            "avg_reward": float(avg_reward),
            "std_reward": float(std_reward),
            "risk_adjusted_score": float(risk_adjusted_score),
            "success_rate": int(success_rate)
        })

        if is_best:
            print(f"Launching clean dual trajectory tracking...")
            _best_trajectory_payload = generate_trajectory(
                eval_env=eval_env,
                actor=actor,
                args=args,
                device=device,
                global_step=global_step,
                prefix="final_eval" # Ensure the keys match the final evaluation grouping
            )
        if not is_interval:
            log_distribution_plot(_eval_history, global_step=global_step, extra_payload=_best_trajectory_payload)
        
    actor.train() 
    return risk_adjusted_score, avg_reward, std_reward, is_best

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

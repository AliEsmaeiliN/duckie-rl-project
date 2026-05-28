import os
import sys 
import torch
import numpy as np
import wandb
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.wrappers.reward_wrappers import compute_unified_eval_reward

class DuckiebotEvaluator:
    """
    Modular evaluation suite for vision-based RL on the Duckiebot.
    Handles episodic evaluation, trajectory mapping, and WandB logging.
    """
    def __init__(self, eval_env, eval_seed, actor, args, device, prefix="eval_nominal"):
        self.eval_env = eval_env
        self.actor = actor
        self.args = args
        self.device = device
        self.seed = eval_seed
        self.prefix = prefix
        # State tracking encapsulated within the instance
        self.eval_history = []
        self.best_trajectory_payload = {}
    
    @staticmethod
    def get_speed_gradient_color(norm_speed):
        """
        Computes a sleek Neon Cyberpunk BGR color for a given normalized speed [0, 1].
        Transitions from Cyan (slow) -> Hot Magenta (medium) -> Vibrant Orange-Red (fast).
        """
        norm_speed = np.clip(norm_speed, 0.0, 1.0)
        
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

    def _log_distribution_plot(self, global_step, extra_payload=None):
        """
        Renders the Reward Distribution and Risk-Adjusted trend plot from memory
        and logs the figure to W&B.
        """
        steps = [h["step"] for h in self.eval_history]
        avg_rewards = [h["avg_reward"] for h in self.eval_history]
        std_rewards = [h["std_reward"] for h in self.eval_history]
        risk_scores = [h["risk_adjusted_score"] for h in self.eval_history]
        success_rates = [h["success_rate"] for h in self.eval_history]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = '#1f77b4'
        ax1.errorbar(steps, avg_rewards, yerr=std_rewards, fmt='o', color=color, 
                     ecolor='#a1c9f4', elinewidth=2, capsize=5, markersize=6, label='Mean Reward \u00b1 Std')
        ax1.plot(steps, risk_scores, 'D-', color='#ff7f0e', label='Risk-Adjusted Score')
        ax1.set_xlabel('Global Training Steps')
        ax1.set_ylabel('Episodic Return', color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color2 = '#2ca02c'
        ax2.plot(steps, success_rates, 'o-.', color=color2, linewidth=2, label='Success Rate')
        ax2.set_ylabel('Success Rate (Completed/Total)', color=color2)
        ax2.set_ylim(-1, 105) 
        ax2.tick_params(axis='y', labelcolor=color2)

        plt.title('Evaluation: Reward Stability vs. Success Rate', fontsize=12, fontweight='bold', pad=15)
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True)
        
        ax1.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()

        log_dict = {f"{self.prefix}/performance_summary": wandb.Image(fig)}

        table = wandb.Table(columns=["step", "avg_reward", "std_reward", "risk_adjusted_score", "success_rate"])
        for h in self.eval_history:
            table.add_data(h["step"], h["avg_reward"], h["std_reward"], h["risk_adjusted_score"], h["success_rate"])
        log_dict[f"{self.prefix}/performance_data"] = table

        if extra_payload:
            log_dict.update(extra_payload)

        print("Logging the final Evaluations Table and Plot.")
        wandb.log(log_dict, step=global_step)
        plt.close(fig)

    def generate_trajectory(self, global_step):
        """
        Runs dedicated visualization episodes, generates overlay trajectory maps, 
        and uploads interactive charts/tables to W&B.
        """
        if not (self.args.track and wandb.run is not None):
            return {}

        sim = self.eval_env.unwrapped
        original_direction = getattr(sim, "direction", "mixed")
        log_payload = {}
        
        for direction_key in ["CW", "CCW"]:
            attempt_count = 0
            max_attempts = 3

            while True:
                print(f"    Running dedicated {direction_key} trajectory > Attempt #{attempt_count + 1}...")
                sim.direction = direction_key
                current_seed = self.seed + attempt_count
                traj_obs, _ = self.eval_env.reset(seed=current_seed)
                traj_done = False
                
                traj_x, traj_z, traj_v, traj_omega = [], [], [], []
                traj_steps, traj_speed = [], []
                traj_heading_err, traj_cte = [], []
                traj_r_total, traj_r_speed, traj_r_lane, traj_r_heading, traj_r_jerk = [], [], [], [], []
                step_count = 0

                prev_action = np.zeros(2, dtype=np.float32)
                
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
                        heading_err, cte = 0.0, 0.0
                    
                    traj_heading_err.append(heading_err)
                    traj_cte.append(cte)
                    
                    with torch.no_grad():
                        obs_tensor = torch.Tensor(traj_obs).unsqueeze(0).to(self.device)
                        if hasattr(self.actor, "get_action"):
                            _, _, action = self.actor.get_action(obs_tensor)
                        else:
                            action = self.actor(obs_tensor)
                        action = action.cpu().numpy().reshape(-1)
                    
                    r_tot, r_sp, r_ln, r_hd, r_jk = compute_unified_eval_reward(
                        sim, current_action=action, prev_action=prev_action, return_components=True
                    )
                    traj_r_total.append(r_tot)
                    traj_r_speed.append(r_sp)
                    traj_r_lane.append(r_ln)
                    traj_r_heading.append(r_hd)
                    traj_r_jerk.append(r_jk)
                    
                    traj_v.append(float(action[0]))
                    traj_omega.append(float(action[1]))
                    
                    next_obs, _, terminated, truncated, _ = self.eval_env.step(action)
                    traj_obs = next_obs
                    traj_done = terminated or truncated
                    
                    prev_action = action.copy()
                    step_count += 1
                
                if truncated and not terminated:
                    print(f"      [Success] Completed full trajectory window without boundaries failure.")
                    break
                else:
                    attempt_count += 1
                    if attempt_count >= max_attempts:
                        print(f"      [Max Attempts Reached] Could not find a flawless completion in {max_attempts} attempts. Proceeding with latest trajectory data.")
                        break
                    print(f"      [Discarded] Agent crashed or exited lane boundaries. Retrying alternative path...")
                

            # Render 2D Top-Down Map
            try:
                top_down_map = sim._render_img(
                    width=800, height=600,
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
                    pixel_points.append([np.clip(px, 0, 799), np.clip(py, 0, 599)])

                if len(pixel_points) > 1:
                    max_observed_speed = max(traj_speed) if traj_speed else 1.2
                    min_observed_speed = min(traj_speed) if traj_speed else 0.0
                    speed_range = max_observed_speed - min_observed_speed if max_observed_speed != min_observed_speed else 1.0

                    for i in range(len(pixel_points) - 1):
                        current_speed = traj_speed[i]
                        norm_speed = (current_speed - min_observed_speed) / speed_range if speed_range > 0 else 0.5
                        color = self.get_speed_gradient_color(norm_speed)
                        cv2.line(overlay_img, tuple(pixel_points[i]), tuple(pixel_points[i+1]), color=color, thickness=1, lineType=cv2.LINE_AA)

                    # Glowing Start/End Markers
                    cv2.circle(overlay_img, tuple(pixel_points[0]), 5, (0, 255, 0), -1, lineType=cv2.LINE_AA)
                    cv2.circle(overlay_img, tuple(pixel_points[0]), 7, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                    cv2.circle(overlay_img, tuple(pixel_points[-1]), 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)
                    cv2.circle(overlay_img, tuple(pixel_points[-1]), 7, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                
                log_payload[f"{self.prefix}/best_milestone_2d_trajectory_image_{direction_key}"] = wandb.Image(
                    overlay_img, caption=f"2D Trajectory Map ({direction_key}) - Step {global_step}"
                )
            except Exception as e:
                print(f"[debug_tools] Warning: Could not generate top-down map overlay for {direction_key}: {e}")

            # Generate Matplotlib Waveform Comparison
            fig_act, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

            ax1.plot(traj_steps, traj_v, label='Agent Throttle (v)', color='#1f77b4', linewidth=2)
            ax1.set_ylabel('Velocity')
            ax1.set_title(f'Action Waveforms ({direction_key}) - Step {global_step}', fontweight='bold')
            ax1.legend(loc='upper right')
            ax1.grid(True, linestyle='--', alpha=0.5)

            ax2.plot(traj_steps, traj_omega, label='RL Agent Steering (\u03c9)', color='#ff7f0e', linewidth=2)
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
            log_payload[f"{self.prefix}/best_milestone_action_waveforms_{direction_key}"] = wandb.Image(
                fig_act, caption=f"RL vs Classical Waveforms ({direction_key}) - Step {global_step}"
            )
            plt.close(fig_act)

            fig_rew, ax_rew = plt.subplots(figsize=(11, 5))
            
            ax_rew.plot(traj_steps, traj_r_total, label='Weighted Total Reward', color='#2ca02c', linewidth=2.5)
            ax_rew.plot(traj_steps, traj_r_speed, label='Progress Component (30%)', color='#bcbd22', linestyle='--', alpha=0.8)
            ax_rew.plot(traj_steps, traj_r_lane, label='Lane Center Component (30%)', color='#e377c2', linestyle='--', alpha=0.8)
            ax_rew.plot(traj_steps, traj_r_heading, label='Heading Component (20%)', color='#17becf', linestyle='--', alpha=0.8)
            ax_rew.plot(traj_steps, traj_r_jerk, label='Lane Reward (20%)', color='#7f7f7f', linestyle=':', alpha=0.9)            

            ax_rew.set_title(f'Objective Evaluation Reward Matrix Analysis ({direction_key}) - Step {global_step}', fontweight='bold', pad=12)
            ax_rew.set_ylabel('Component Score Contributions')
            ax_rew.set_xlabel('Simulation Decision Steps')
            ax_rew.set_ylim(-1.05, 0.65)
            ax_rew.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=True)
            ax_rew.grid(True, linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            log_payload[f"{self.prefix}/best_milestone_reward_decompositions_{direction_key}"] = wandb.Image(
                fig_rew, caption=f"Evaluation Metric Breakdown Profiles ({direction_key}) - Step {global_step}"
            )
            plt.close(fig_rew)

        # Restore original environment state
        sim.direction = original_direction
        return log_payload

    def evaluate(self, is_interval=False, global_step=0, best_reward=-float('inf'), num_episodes=10):
        """
        Unified policy evaluation method.
        Returns: (risk_adjusted_score, avg_reward, std_reward, is_best)
        """
        if global_step >= self.args.total_timesteps:
            is_interval = False
            print("Final Evaluation test and Logging the Data")

        eval_type = "Interval" if is_interval else "Final"
        print(f"\n--- Starting {eval_type} Evaluation in {self.prefix} [Step {global_step}]: {num_episodes} Episodes ---")
        self.actor.eval()

        all_rewards = []
        completed_episodes = 0
        self.eval_env.reset(seed=self.seed)

        raw_sim = self.eval_env.unwrapped
        
        for _ in range(num_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episodic_reward = 0
            prev_action = np.zeros(2, dtype=np.float32)
            
            while not done:
                with torch.no_grad():
                    obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
                    if hasattr(self.actor, "get_action"):
                        _, _, action = self.actor.get_action(obs_tensor) 
                    else:
                        action = self.actor(obs_tensor) 
                
                    action = action.cpu().numpy().reshape(-1)

                next_obs, _, terminated, truncated, _ = self.eval_env.step(action)
                step_eval_reward = compute_unified_eval_reward(
                    raw_sim, current_action=action, prev_action=prev_action, return_components=False
                )
                
                obs = next_obs
                episodic_reward += step_eval_reward            
                done = terminated or truncated
                prev_action = action.copy()

            all_rewards.append(episodic_reward)
            if truncated and not terminated:
                completed_episodes += 1

        avg_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        success_rate = (completed_episodes / num_episodes) * 100
        print(f"--- {eval_type} Evaluation Complete | Average Reward: {avg_reward:.2f} (Std: {std_reward:.2f}) | Success Rate: {success_rate:.2f} ---")

        beta = 0.5
        risk_adjusted_score = avg_reward - (beta * std_reward)
        print(f"    Risk-Adjusted Score: {risk_adjusted_score:.2f}")

        is_best = risk_adjusted_score > best_reward

        # Log to WandB
        if self.args.track:
            self.eval_history.append({
                "step": int(global_step),
                "avg_reward": float(avg_reward),
                "std_reward": float(std_reward),
                "risk_adjusted_score": float(risk_adjusted_score),
                "success_rate": int(success_rate)
            })

            if is_best:
                print("Launching clean dual trajectory tracking...")
                self.best_trajectory_payload = self.generate_trajectory(global_step=global_step)
                wandb.log(self.best_trajectory_payload, step=global_step)
            if not is_interval:
                self._log_distribution_plot(global_step=global_step, extra_payload=self.best_trajectory_payload)
            
        self.actor.train() 
        return risk_adjusted_score, avg_reward, std_reward, is_best, success_rate

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

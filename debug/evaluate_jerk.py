import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from rl.cnn_architectures import ImpalaCNN as cnn_encoder
from utils.rl_env import DuckieOvalEnv
from models import SACActor as EvaluationActor

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAP_NAME = "oval_loop"
MAX_STEPS = 1500  # Safe upper bound safety cutoff

MODEL_WITH_JERK_PATH = os.path.expanduser('~/workspace/rl_models/sac_vr2.cleanrl_model')
MODEL_WITHOUT_JERK_PATH = os.path.expanduser('~/workspace/rl_models/sac_vr2_noJ.cleanrl_model')


def load_actor(model_path):
    """Safely extracts state dict from CleanRL checkpoint framework."""
    actor = EvaluationActor().to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if 'actor_state_dict' in checkpoint:
        actor.load_state_dict(checkpoint['actor_state_dict'])
    else:
        actor.load_state_dict(checkpoint)
    actor.eval()
    return actor

def run_evaluation_loop(actor, direction_string):
    """Runs a rollout until a complete loop (10 tiles passed) is achieved."""
    # FIX: Environment looks for exact uppercase "CW" or "CCW" strings in logic
    validated_direction = direction_string.upper()
    
    env = DuckieOvalEnv.create_wrapped(
        run_name="eval_script",
        capture_video=False,
        ema=True,
        motion_blur=False,
        grayscale=True,
        frame_stack=4,
        latency_rand=False,
        recovery_step=False,
        jerk_penalty=False,
        preprocessing=False,
        direction=validated_direction,
        spawn_mode="curriculum"
    )
    
    obs, _ = env.reset(seed=42)
    
    data = {
        "steps": [], "v": [], "omega": []
    }
    
    step = 0
    done = False
    
    while not done and step < MAX_STEPS:
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(DEVICE)
            action = actor(obs_tensor).cpu().numpy().reshape(-1)
            
        data["steps"].append(step)
        data["v"].append(action[0])
        data["omega"].append(action[1])
        
        obs, _, terminated, truncated, info = env.step(action)
        
        # Extract performance tracking parameters
        # Accessing nested environment dictionary: info -> 'Simulator' -> 'tiles_passed'
        
        tiles_passed = info.get('tiles_passed', 0)
        
        # Loop Termination Rule: Exit immediately upon complete loop tracking
        if tiles_passed >= 10:
            print(f"    [Loop Terminated] Finished full lap cleanly at step {step} ({tiles_passed} tiles).")
            terminated = True
            
        done = terminated or truncated
        step += 1
        
    env.close()
    return data

# --- Main Run Stack ---
if __name__ == "__main__":
    print("Loading models...")
    models = {
        "With Jerk Penalty": load_actor(MODEL_WITH_JERK_PATH),
        "Without Jerk Penalty": load_actor(MODEL_WITHOUT_JERK_PATH)
    }
    
    # Standard lowercase inputs, validated and cast to upper internally
    directions = ["cw", "ccw"]
    results = {}
    
    for model_label, actor_net in models.items():
        results[model_label] = {}
        for d in directions:
            print(f"Evaluating model variant [{model_label}] navigating profile [{d}]...")
            results[model_label][d] = run_evaluation_loop(actor_net, d)
            
    # --- Visualization Generation ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    model_keys = list(models.keys())
    
    for col_idx, model_label in enumerate(model_keys):
        for d, color, style in zip(directions, ['#1f77b4', '#d62728'], ['-', '--']):
            d_data = results[model_label][d]
            steps = d_data["steps"]
            
            # Row 0: Linear Velocity (v)
            axes[0, col_idx].plot(steps, d_data["v"], label=f"{d.upper()} Loop", color=color, linestyle=style, linewidth=2)
            axes[0, col_idx].set_ylim(0.0, 1.0)
            if model_label == "With Jerk Penalty":
                axes[0, col_idx].set_ylabel("Velocity ($v$)", fontsize=16)
            axes[0, col_idx].set_title(f"{model_label}", fontsize=16)
            axes[0, col_idx].grid(True, linestyle='--', alpha=0.4)
            
            # Row 1: Angular Velocity (omega)
            axes[1, col_idx].plot(steps, d_data["omega"], label=f"{d.upper()} Loop", color=color, linestyle=style, linewidth=2)
            axes[1, col_idx].set_ylim(-1.0, 1.0)
            if model_label == "With Jerk Penalty":
                axes[1, col_idx].set_ylabel("Steering ($\omega$)", fontsize=16)
            axes[1, col_idx].set_xlabel("Simulation Decision Steps", fontsize=12)
            #axes[1, col_idx].set_title(f"{model_label}")
            axes[1, col_idx].grid(True, linestyle='--', alpha=0.4)
            
        axes[0, col_idx].legend(loc="upper right")
        
    plt.suptitle(f"Action Comparison (Full Loop Completion Cutoff) | Map: {MAP_NAME}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_filename = "directional_overlay_comparison.png"
    plt.savefig(output_filename, dpi=200)
    print(f"\nEvaluation complete. Overlay data plot saved to: {output_filename}")
from dataclasses import dataclass
from typing import Optional

@dataclass
class SharedDuckieArgs:
    """Common Environment, Logging, and Base RL configurations"""
    # --- System & Execution ---
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    
    # --- Weights & Biases Logging ---
    track: bool = False
    wandb_project_name: str = "Duckie-RL-V2"
    wandb_entity: str = None
    run_notes: str = ""
    
    # --- Environment & Architecture ---
    env_id: str = "Oval-"
    num_envs: int = 1
    grayscale: bool = True
    capture_video: bool = False
    
    # --- Evaluation & Saving ---
    eval_interval: int = 20000
    start_evaluation: int = 1e5
    eval_model: bool = True
    save_model: bool = True
    save_interval: int = 50000
    version: int = 0
    
    # --- Shared Off-Policy RL Hyperparameters ---
    total_timesteps: int = 1000000
    buffer_size: int = int(5e4)
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    learning_starts: int = int(5e3)
    policy_frequency: int = 2
    
    # --- Duckietown specific arguments ---
    domain_rand: bool = False
    """texture/light randomization"""
    distortion: bool = False 
    """Simulates the fisheye lens"""
    dynamics_rand: bool = False
    """Simulates motor/trim imbalances"""
    camera_rand: bool = False 
    """Simulates mounting misalignments"""
    direction: str = "mixed"
    """Choosing the direction of the loop. CW, CCW or mixed"""
    curriculum_randomization: bool = True
    """Activating the randomizations gradually based on curriculum learning"""

    # --- Wrapper Configuration ---
    motion_blur: bool = False
    """Simulates the blur from the moving duckiebot"""
    action_latency: bool = False
    """Simulates the action latency from the duckiebot"""
    ema: bool = False
    """Use EMA action smoothing"""
    eval_ema: Optional[bool] = None
    """Override EMA action smoothing for evaluation. If None, inherits the training setting."""
    recovery: bool = False
    """Gives the robot 20 steps to recover"""
    jerk_penalty: bool = False
    """Adding the jerk penalty to the final reward"""
    preprocessing: bool = False
    """if toggled, applies the full sim-to-real visual enhancement stack (CLAHE, blur, contrast stretching)"""
    preprocessing_eval: Optional[bool] = None
    """Override preprocessing for evaluation. If None, inherits the training setting."""
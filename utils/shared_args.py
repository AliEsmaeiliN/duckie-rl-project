from dataclasses import dataclass
from typing import Optional

@dataclass
class SharedDuckieArgs:
    """Common Environment and Wrapper configurations for Duckietown RL"""
    
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
    recovery: bool = False
    """Gives the robot 20 steps to recover"""
    jerk_penalty: bool = False
    """Adding the jerk penalty to the final reward"""
    preprocessing: bool = False
    """if toggled, applies the full sim-to-real visual enhancement stack (CLAHE, blur, contrast stretching)"""
    preprocessing_eval: Optional[bool] = None
    """Override preprocessing for evaluation. If None, inherits the training setting."""
import os
import gymnasium as gym
import numpy as np
from gym_duckietown.simulator import Simulator
from wrappers_debug import (
    KinematicActionWrapper, ActionWrapper, ResizeWrapper, 
    CropResizeWrapper, ImgWrapper, DebugRewardWrapper, DtRewardWrapper,
    UndistortWrapper
)

class DuckieOvalEnv(Simulator):
    """
    A specialized Duckietown environment for Oval navigation.
    """
    def __init__(self, **kwargs):
        kwargs.setdefault('map_name', "oval_loop")
        kwargs.setdefault('camera_width', 640)
        kwargs.setdefault('camera_height', 480)
        kwargs.setdefault('accept_start_angle_deg', 4)
        kwargs.setdefault('full_transparency', True)
        kwargs.setdefault('max_steps', 10000)
        
        kwargs.setdefault('frame_skip', 4) 
        
        super().__init__(**kwargs)
        
        self.wheel_dist = 0.102 
        self.robot_radius = 0.0318
        self.motor_k = 27.0

    @classmethod
    def create_wrapped(cls, run_name, capture_video=False, motion_blur=False, grayscale=True, frame_stack=4, reward_type="adp", **kwargs):
        """
        Static method to build the fully wrapped stack.
        """
        env = cls(**kwargs)

        env = UndistortWrapper(env)

        # 1. Kinematics (v, w -> wl, wr)
        env = KinematicActionWrapper(env, wheel_dist=0.102, radius=0.0318, k=27.0)
        env = ActionWrapper(env)

        if capture_video:
            video_folder = f"videos/{run_name}"
            os.makedirs(video_folder, exist_ok=True)
            env = gym.wrappers.RecordVideo(env, video_folder, episode_trigger=lambda x: True)

        # 3. Vision Pipeline (Sim2Real Insurance)
        env = ResizeWrapper(env, shape=(120, 160, 3)) # Ensure 120x160 base
        env = CropResizeWrapper(env, shape=(84, 84))  # Crop sky, resize to 84x84
        
        if grayscale:
            env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
        
        env = ImgWrapper(env) # Transpose to CHW

        
        # 5. Reward System
        #env = DtRewardWrapper(env)
        env = DebugRewardWrapper(env, reward_type=reward_type)

        # 6. Temporal Stacking
        if frame_stack > 1:
            env = gym.wrappers.FrameStackObservation(env, stack_size=frame_stack)
            c = 1 if grayscale else 3
            final_channels = c * frame_stack
            new_obs_space = gym.spaces.Box(
                low=0, 
                high=255, 
                shape=(final_channels , 84, 84), 
                dtype=np.uint8
            )   
            env = gym.wrappers.TransformObservation(
                env, 
                lambda obs: np.array(obs).reshape(final_channels, 84, 84),
                observation_space=new_obs_space
            )

        return gym.wrappers.RecordEpisodeStatistics(env)

    def set_randomization(self, **kwargs):
        """
        Dynamically toggle randomization flags for Curriculum Learning.
        
        self.dynamics_rand    # Motor/Trim noise
        self.domain_rand      # Visual/Light noise
        self.distortion       # Fisheye effect
        self.camera_rand      # Camera mounting noise
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"Simulator config updated: {key} = {value}")
            else:
                print(f"Warning: Simulator has no attribute '{key}'")

            if getattr(self, 'distortion', False) and self.camera_model is None:
                from src.gym_duckietown.distortion import Distortion
                print("Initializing Distortion Model...")
                self.camera_model = Distortion(camera_rand=getattr(self, 'camera_rand', False))
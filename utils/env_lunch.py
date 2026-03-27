import os
import gymnasium as gym
import numpy as np
# Duckietown Specific
from gym_duckietown.envs import DuckietownEnv
from utils.wrappers import ImgWrapper, ActionWrapper, CropResizeWrapper, CustomRewardWrapper, MotionBlurWrapper

class EnvLunch:
    def __init__(self, 
                 run_name: str, 
                 max_steps: int = 1500, 
                 grayscale: bool = True, 
                 frame_stack: int = 4, 
                 img_shape: tuple = (84, 84),
                 **sim_to_real_kwargs):
        self.run_name = run_name
        self.max_steps = max_steps
        self.grayscale = grayscale
        self.frame_stack = frame_stack
        self.img_shape = img_shape
        self.sim_to_real_kwargs = sim_to_real_kwargs

    def _create_base_env(self, seed, render_mode=None):
        """Initializes the raw Duckietown simulator."""
        return DuckietownEnv(
            seed=seed,
            map_name="oval_loop",
            max_steps=self.max_steps,
            camera_width=160,
            camera_height=120,
            accept_start_angle_deg=4,
            full_transparency=True,
            render_mode=render_mode,
            frame_skip=3,
            **self.sim_to_real_kwargs
        )

    def _apply_wrappers(self, env, capture_video=False):
        """Sequentially applies Gymnasium wrappers."""
        env = MotionBlurWrapper(env)
        if capture_video:
            video_folder = f"videos/{self.run_name}"
            os.makedirs(video_folder, exist_ok=True)
            env = gym.wrappers.RecordVideo(env, video_folder, episode_trigger=lambda x: True)

        # Vision Preprocessing
        env = CropResizeWrapper(env, shape=self.img_shape)
        if self.grayscale:
            env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
        
        env = ImgWrapper(env) # CHW format
        
        # Dynamics & Rewards
        env = ActionWrapper(env)
        env = CustomRewardWrapper(env)

        # Temporal Stacking
        if self.frame_stack > 1:
            env = gym.wrappers.FrameStackObservation(env, stack_size=self.frame_stack)
            
            # Reshape for CNN input
            base_channels = 1 if self.grayscale else 3
            final_channels = base_channels * self.frame_stack
            
            new_obs_space = gym.spaces.Box(
                low=0, high=255, 
                shape=(final_channels, *self.img_shape), 
                dtype=np.uint8
            )
            
            env = gym.wrappers.TransformObservation(
                env, 
                lambda obs: np.array(obs).reshape(final_channels, *self.img_shape),
                observation_space=new_obs_space
            )

        return gym.wrappers.RecordEpisodeStatistics(env)

    def make_env_fn(self, seed, idx, capture_video=False):
        """Returns a 'thunk' function for VectorEnv integration."""
        def thunk():
            render_mode = "rgb_array" if (capture_video and idx == 0) else None
            env = self._create_base_env(seed, render_mode)
            env = self._apply_wrappers(env, capture_video)
            env.action_space.seed(seed)
            return env
        return thunk
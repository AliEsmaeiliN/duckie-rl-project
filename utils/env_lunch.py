import gymnasium as gym
import numpy as np
# Duckietown Specific
from gym_duckietown.envs import DuckietownEnv
from utils.wrappers import NormalizeWrapper, ImgWrapper, DtRewardWrapper, ActionWrapper, ResizeWrapper, CropResizeWrapper, CustomRewardWrapper

def create_dt_env(seed, max_steps, render_mode=None):
    env = DuckietownEnv(
            seed=seed,
            map_name="oval_loop", 
            max_steps=max_steps,
            camera_width=160,
            camera_height=120,
            
            # FOR SIM-TO-REAL
            domain_rand=False,        # for texture/light randomization
            #distortion=True,         # Simulates the fisheye lens
            #dynamics_rand=True,      # Simulates motor/trim imbalances
            #camera_rand=True,        # Simulates mounting misalignments
            
            accept_start_angle_deg=4, # Forces learning of recovery
            
            full_transparency=True,
            render_mode=render_mode,
            frame_skip=3             
        )
    
    return env

def apply_wrappers(env, run_name, capture_video=False, grayscale=True):

    if capture_video:
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

    # Crop and Resize first (from 120x160 to 84x84)
    env = CropResizeWrapper(env, shape=(84, 84))

    # Converting to Grayscale if needed
    if grayscale:
        env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)

    # To make the images from W*H*C into C*W*H
    env = ImgWrapper(env)
    env = ActionWrapper(env)
    env = CustomRewardWrapper(env)

    # Stack 4 frames
    stack_size=4
    env = gym.wrappers.FrameStackObservation(env, stack_size=stack_size)

    #Flatten the channels for the Encoder
    base_channels = 1 if grayscale else 3
    final_channels = base_channels * stack_size
    new_obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(final_channels, 84, 84), dtype=np.uint8)
    
    env = gym.wrappers.TransformObservation(
        env, 
        lambda obs: obs.reshape(final_channels, 84, 84),
        observation_space=new_obs_space  
    )   

    return env

def make_env(seed, idx, capture_video, run_name, max_steps = 1500, grayscale=True):
    def thunk():

        render_mode = "rgb_array" if (capture_video and idx == 0) else None

        env = create_dt_env(seed=seed, max_steps=max_steps, render_mode=render_mode)
        env = apply_wrappers(env, run_name, capture_video, grayscale)
        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

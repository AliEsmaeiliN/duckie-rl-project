import gymnasium as gym
import numpy as np
# Duckietown Specific
from gym_duckietown.envs import DuckietownEnv
from utils.wrappers import NormalizeWrapper, ImgWrapper, DtRewardWrapper, ActionWrapper, ResizeWrapper, CropResizeWrapper

def make_env(seed, idx, capture_video, run_name, max_steps_d = 1000):
    def thunk():
        render_mode = "rgb_array" if (capture_video and idx == 0) else None
        # 1. Initializing the Duckietown env
        env = DuckietownEnv(
            seed=123,  # random seed
            map_name="oval_loop",
            max_steps= max_steps_d,  # we don't want the gym to reset itself
            domain_rand=False,
            camera_width=160,
            camera_height=120,
            accept_start_angle_deg=4,  # start close to straight
            full_transparency=True,
            distortion=False,
            render_mode=render_mode,
            frame_skip = 3
        )
        print("Initialized environment")

        # 2. Record video if requested (CleanRL standard)   
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

        # 3. Crop and Resize first (from 120x160 to 84x84)
        env = CropResizeWrapper(env, shape=(84, 84))

        # 4. To make the images from W*H*C into C*W*H
        env = ImgWrapper(env)


        env = ActionWrapper(env)
        env = DtRewardWrapper(env)
        print("Initialized Wrappers")

        # 5. Stack 4 frames
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)
        #Flatten the 4x3 channels into 12 for the DQNEncoder
        new_obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(12, 84, 84), dtype=np.uint8)
        
        env = gym.wrappers.TransformObservation(
            env, 
            lambda obs: obs.reshape(12, 84, 84),
            observation_space=new_obs_space  
        )   
        # 6. Basic RL statistics for Tensorboard
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env.action_space.seed(seed)
        return env

    return thunk

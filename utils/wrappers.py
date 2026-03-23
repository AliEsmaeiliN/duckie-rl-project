import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image

from gym_duckietown.simulator import Simulator


class MotionBlurWrapper(Simulator):
    def __init__(self, env=None):
        Simulator.__init__(self)
        self.env = env
        self.frame_skip = 3
        self.env.delta_time = self.env.delta_time / self.frame_skip

    def step(self, action: np.ndarray):
        action = np.clip(action, -1, 1)
        # Actions could be a Python list
        action = np.array(action)
        motion_blur_window = []
        for _ in range(self.frame_skip):
            obs = self.env.render_obs()
            motion_blur_window.append(obs)
            self.env.update_physics(action)

        # Generate the current camera image

        obs = self.env.render_obs()
        motion_blur_window.append(obs)
        obs = np.average(motion_blur_window, axis=0, weights=[0.8, 0.15, 0.04, 0.01])

        misc = self.env.get_agent_info()

        d = self.env._compute_done_reward()
        misc["Simulator"]["msg"] = d.done_why

        return obs, d.reward, d.done, misc


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(120, 160, 3)):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=shape, 
            dtype=np.uint8
        )
        self.shape = shape # (120, 160, 3)
    def observation(self, observation):
        #from scipy.misc import imresize

        #return imresize(observation, self.shape)
        
        resized = Image.fromarray(observation).resize((self.shape[1], self.shape[0]))
        return np.array(resized)


class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super().__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)


class ImgWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class DtRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -10

        return reward


# this is needed because at max speed the duckie can't turn anymore
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        action_ = [action[0] * 0.8, action[1]]
        return action_

class CropResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        # Update the observation space to the new dimensions
        # Assuming RGB (3 channels)
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(self.shape[0], self.shape[1], 3), 
            dtype=np.uint8
        )

    def observation(self, obs):
        # 1. Convert to PIL for easy manipulation
        img = Image.fromarray(obs)
        
        width, height = img.size
        
        # 2. Crop: Keep the bottom 2/3
        # PIL crop box is (left, top, right, bottom)
        top_boundary = int(height * (1/3))
        img = img.crop((0, top_boundary, width, height))
        
        # 3. Resize to target shape (84x84)
        # Note: Image.resize takes (width, height)
        img = img.resize((self.shape[1], self.shape[0]), Image.BILINEAR)
        
        return np.array(img)
    
class DebugRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_action = np.array([0.0, 0.0])

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        sim = self.env.unwrapped
        try:
            lp = sim.get_lane_pos2(sim.cur_pos, sim.cur_angle)
            
            # Calculate your components
            r_speed = 2.0 * sim.speed
            k = 50
            r_align = np.exp(k * (lp.dot_dir - 1.0))
            r_dist  = -10.0 * np.abs(lp.dist)
            r_angle = -0.1 * np.abs(lp.angle_deg)
            
            action_diff = np.linalg.norm(action[0] - self.last_action[0])
            r_jerk = -0.5 * action_diff
            
            total = r_speed + r_align + r_dist + r_angle + r_jerk
            
            # 3. Print the breakdown (only every 15 steps to avoid flickering)
            msg = (
                f"\rDEBUG | Total: {total:6.2f} | Spd: {r_speed:4.1f} | Dist: {r_dist:5.1f} "
                f"| Aln: {r_align:4.1f} | Jrk: {r_jerk:4.1f} "
                f"| dir:{lp.dot_dir} | angle:{lp.angle_deg} | "
            )   
            print(msg, end="", flush=True)
                
        except Exception:
            # Handle 'NotInLane' or other issues during manual driving
            if sim.step_count % 15 == 0:
                print("DEBUG | NOT IN LANE")

        self.last_action = action.copy()
        return obs, reward, terminated, truncated, info
    
    
class CustomRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.jerk_penalty_coeff = - 0.5
        self.prev_action = np.zeros(2)

    def reward(self, reward):
        # Get internal simulator state for custom math
        sim = self.env.unwrapped 
        pos = sim.cur_pos
        angle = sim.cur_angle
        speed = sim.speed
        current_action = sim.last_action
        
        try:
            lp = sim.get_lane_pos2(pos, angle)
        except Exception:
            return -10.0 
            
        reward_speed = 2.5 * speed
        k = 2
        reward_alignment = np.exp(k * (lp.dot_dir - 1.0)) # tanh like behaviour to add a higher gradint near 1
        reward_distance = -10.0 * np.abs(lp.dist)
        reward_angle = -0.1 * np.abs(lp.angle_deg)
        
        action_diff = np.linalg.norm(current_action - self.prev_action) 
        reward_jerk = self.jerk_penalty_coeff * action_diff

        self.prev_action = current_action.copy()

        return reward_speed + reward_alignment + reward_distance + reward_angle + reward_jerk
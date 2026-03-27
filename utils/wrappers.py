import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image

from gym_duckietown.simulator import Simulator
from gym_duckietown.simulator import get_dir_vec


class MotionBlurWrapper(gym.Wrapper):
    def __init__(self, env=None, frame_skip=3):
        super().__init__(env)
        self.frame_skip = frame_skip
        self.env.unwrapped.delta_time = self.env.unwrapped.delta_time / (self.frame_skip + 1)
        
        
        self.weights = [0.01, 0.04, 0.15, 0.8]
    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        motion_blur_window = []

        for _ in range(self.frame_skip):
            obs = self.env.unwrapped.render_obs()
            motion_blur_window.append(obs)
            self.env.unwrapped.update_physics(action)


        obs = self.env.unwrapped.render_obs()
        motion_blur_window.append(obs)

        # axis=0 averages the pixel values across the time dimension
        blurred_obs = np.average(motion_blur_window, axis=0, weights=self.weights).astype(np.uint8)

        d_info = self.env.unwrapped._compute_done_reward(action)
        misc = self.env.unwrapped.get_agent_info()
        misc["Simulator"]["msg"] = d_info.done_why

        return blurred_obs, d_info.reward, d_info.done, False, misc


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
        img = Image.fromarray(obs)
        
        width, height = img.size
        
        # PIL crop box is (left, top, right, bottom)
        top_boundary = int(height * (1/3))
        img = img.crop((0, top_boundary, width, height))
        
        # target shape (84x84)
        img = img.resize((self.shape[1], self.shape[0]), Image.BILINEAR)
        
        return np.array(img)
    
class DebugRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_action = np.zeros(2)
    
    def get_predictive_direction(self, sim, pos, angle, look_ahead_dist=0.3):
        """
        Looks ahead in the current driving direction to see 
        what kind of turn is coming up.
        """
        # 1. Calculate a point ahead of the robot
        dir_vec = np.array([np.cos(angle), 0, -np.sin(angle)])
        predictive_pos = pos + dir_vec * look_ahead_dist
        
        # 2. Get the tile at that forward point
        coord = sim.get_grid_coords(predictive_pos)
        next_tile = sim._get_tile(*coord)
        #print(next_tile["kind"])
        if not next_tile or "curve" not in next_tile["kind"]:
            return "STRAIGHT"

        # 3. Apply your direction logic to the upcoming tile
        tile_dir_idx = next_tile["angle"]
        # We use the current angle to predict how we will enter it
        direction_idx = int(round(angle / (np.pi / 2)) + 1) % 4
        
        if (direction_idx + 2) % 4 == tile_dir_idx:
            return "INNER (CW)"
        elif direction_idx == tile_dir_idx:
            return "OUTER (CCW)"
        else:
            return "TRANSITION"
    

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        sim = self.env.unwrapped
        pos = sim.cur_pos
        angle = sim.cur_angle
        speed = sim.speed
        current_action = sim.last_action
        try:
            lp = sim.get_lane_pos2(sim.cur_pos, sim.cur_angle)
            
            coords = sim.get_grid_coords(pos) #
            tile = sim._get_tile(*coords) #
            tile_kind = tile["kind"] if tile else ""


            # Default
            dist_penalty_coeff = -10.0
            speed_coeff = 2.0

            #lane_dir = self.get_predictive_direction(sim, pos, angle)
            #print(lane_dir, end="\r", flush=True)
            direction = sim.episode_dir
            print(direction)

            lookahead_dist = 0.25 
            dir_vec = np.array([np.cos(angle), 0, -np.sin(angle)]) # Based on get_dir_vec
            lookahead_pos = pos + dir_vec * lookahead_dist
            
            look_coords = sim.get_grid_coords(lookahead_pos)
            look_tile = sim._get_tile(*look_coords)
            look_kind = look_tile["kind"] if look_tile else ""

            print(f"current: {tile_kind}, lookahead: {look_kind}")
            in_curve = "curve" in tile_kind
            approaching_curve = ("curve" in look_kind)
            in_danger_zone = (direction == "CW") and (in_curve or approaching_curve)
            print(in_curve)
            print(approaching_curve)
            print(in_danger_zone)

            if tile_kind == "curve_right":
                dist_penalty_coeff = -15.0 
                speed_coeff = 2.0
            elif tile_kind == "curve_left":
                dist_penalty_coeff = -10.0
                speed_coeff = 2.0

            r_speed = speed_coeff * speed
            k = 5
            r_align = np.exp(k * (lp.dot_dir - 1.0)) # tanh like behaviour to add a higher gradint near 1
            r_dist = dist_penalty_coeff * np.abs(lp.dist)
            r_angle = -0.1 * np.abs(lp.angle_deg)
            
            action_diff = np.linalg.norm(current_action - self.prev_action) 
            r_jerk = - 0.3 * action_diff

            self.prev_action = current_action.copy()

            total = r_angle + r_speed + r_align + r_jerk + r_dist
            msg = (
                f"\rDEBUG | Total: {total:6.2f} | Spd: {r_speed:4.1f} | Dist: {r_dist:5.1f} "
                f"| Aln: {r_align:4.1f} | Angle: {r_angle:4.1f}    | Jrk: {r_jerk:4.1f} "
                f"| dir:{lp.dot_dir} | angle:{lp.angle_deg} | "
            )   
            #print(msg, end="", flush=True)
            #print(lp.dist)
               
        except Exception as e:
            # Handle 'NotInLane' or other issues during manual driving
            if sim.step_count % 15 == 0:
                print(f"DEBUG | NOT IN LANE or Error: {e}")

        self.prev_action = action.copy()
        return obs, reward, terminated, truncated, info
    
    
class CustomRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
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
            
        # Asymmetric Logic
        coords = sim.get_grid_coords(pos) #
        tile = sim._get_tile(*coords) #
        tile_kind = tile["kind"] if tile else ""
        direction = sim.episode_dir

        # Lookahead Logic
        lookahead_dist = 0.25 
        dir_vec = np.array([np.cos(angle), 0, -np.sin(angle)]) # Based on get_dir_vec
        lookahead_pos = pos + dir_vec * lookahead_dist
        
        look_coords = sim.get_grid_coords(lookahead_pos)
        look_tile = sim._get_tile(*look_coords)
        look_kind = look_tile["kind"] if look_tile else ""

        in_curve = "curve" in tile_kind
        approaching_curve = "curve" in look_kind
        in_danger_zone = (direction == "CW") and (in_curve or approaching_curve)

        # Default
        dist_penalty_coeff = -8.0
        speed_coeff = 2.0
        jerk_coeff = -0.3
        k = 5.0

        if in_danger_zone:
            # Special "Stabilization" Values
            speed_coeff = 0.4      
            dist_penalty_coeff = -15.0      
            jerk_coeff = -1.5       
            k = 10.0                      

        
        reward_speed = speed_coeff * speed * lp.dot_dir
        reward_alignment = np.exp(k * (lp.dot_dir - 1.0)) # tanh like behaviour to add a higher gradint near 1
        reward_distance = dist_penalty_coeff * np.abs(lp.dist)
        reward_angle = -0.03 * np.abs(lp.angle_deg)
        
        action_diff = np.linalg.norm(current_action - self.prev_action) 
        reward_jerk = jerk_coeff * action_diff

        self.prev_action = current_action.copy()

        return reward_speed + reward_alignment + reward_distance + reward_angle + reward_jerk
import gymnasium as gym
import numpy as np

class DtRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -15.0

        return reward

class CustomRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_action = np.zeros(2)
        self.WRONG_LANE_LIMIT = - 0.12

    def reward(self, reward):

        if reward == -1000:
            return -20

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
        lookahead_dist = 0.1 
        dir_vec = np.array([np.cos(angle), 0, -np.sin(angle)]) # Based on get_dir_vec
        lookahead_pos = pos + dir_vec * lookahead_dist
        
        look_coords = sim.get_grid_coords(lookahead_pos)
        look_tile = sim._get_tile(*look_coords)
        look_kind = look_tile["kind"] if look_tile else ""

        in_curve = "curve" in tile_kind
        approaching_curve = "curve" in look_kind
        in_danger_zone = (direction == "CW") and (approaching_curve or in_curve)



        if in_danger_zone:
            # Special "Stabilization" Values
            speed_coeff = 1.0
            dist_coeff = -15.0
            jerk_coeff = -1.2
            target_offset = 0.05
            alignment_k = 5.0
        else:
            # "Race Mode" for straights
            speed_coeff = 2.5
            dist_coeff = -10.0
            jerk_coeff = - 2
            target_offset = 0.0
            alignment_k = 2.0
        
        if speed < 0.05:
            reward_speed = -1
        else:
            reward_speed = speed_coeff * speed * lp.dot_dir


        reward_alignment = 0.5 + np.exp(alignment_k * (lp.dot_dir - 1.0)) # tanh like behaviour to add a higher gradint near 1

        if lp.dist < self.WRONG_LANE_LIMIT:
            dist_coeff = -50

        reward_distance = dist_coeff * (lp.dist + target_offset) ** 2
            
        reward_angle = -0.03 * np.abs(lp.angle_deg)
        
        action_diff = np.linalg.norm(current_action - self.prev_action) 
        reward_jerk = jerk_coeff * action_diff
        reward_survival = 2

        self.prev_action = current_action.copy()
        total_reward = reward_speed + reward_alignment + reward_distance + reward_angle + reward_jerk + reward_survival
 
        return total_reward
    

class SimpleRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_action = np.zeros(2)

    def reward(self, reward):
        if reward == -1000:
            return -30

        # Get internal simulator state for custom math
        sim = self.env.unwrapped 
        pos = sim.cur_pos
        angle = sim.cur_angle
        speed = sim.speed
        current_action = sim.last_action
        lane_width = 0.1

        try:
            lp = sim.get_lane_pos2(pos, angle)
        except Exception:
            return -10.0 

        speed_coeff = 1.5
        dist_coeff = -15.0
        jerk_coeff = -1.0
        alignment_k = 2.0

        reward_speed = speed_coeff * speed
        reward_alignment = alignment_k * (lp.dot_dir ** 2) if lp.dot_dir > 0 else 4.0 * lp.dot_dir
        if np.abs(lp.dist) >= lane_width:
            dist_coeff = -40
        reward_distance = dist_coeff * np.abs(lp.dist)
        reward_angle = -0.03 * np.abs(lp.angle_deg)
        
        action_diff = np.linalg.norm(current_action - self.prev_action) 
        reward_jerk = jerk_coeff * action_diff

        self.prev_action = current_action.copy()
        print(f"{lp.dist}  distance \r", flush=True)

        return reward_speed + reward_alignment + reward_distance + reward_angle + reward_jerk
        

class AdaptiveRewardWrapper(gym.RewardWrapper):
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

class LanePositionReward(gym.RewardWrapper):
    def __init__(self, env, target_offset= -0.02):
        super().__init__(env)
        self.target_offset = target_offset 
        self.wrong_lane_limit = -0.2
        self.prev_action = np.zeros(2) 

    def reward(self, reward):

        if reward == -1000:
            return -10
        
        sim = self.env.unwrapped
        try:
            lp = sim.get_lane_pos2(sim.cur_pos, sim.cur_angle)
        except Exception:
            return -10.0 

        v = sim.speed
        reward_speed = 2.0 * v * lp.dot_dir
        
        if lp.dist < self.wrong_lane_limit:
            reward_distance = -10.0 
        else:
            reward_distance = -15.0 * (lp.dist - self.target_offset)**2
        
        reward_survival = 1.0 if v > 0.05 else -1.0
        
        current_action = sim.last_action
        reward_jerk = -2 * np.linalg.norm(current_action - self.prev_action)
        self.prev_action = current_action.copy()
        
        return reward_speed + reward_distance + reward_survival + reward_jerk
    
class PIDReward(gym.RewardWrapper):
    def __init__(self, env, target_offset= -0.03):
        super().__init__(env)
        self.target_offset = target_offset 
        self.wrong_lane_limit = -0.2

    def reward(self, reward):

        if reward == -1000:
            return -15
        
        sim = self.env.unwrapped
        try:
            lp = sim.get_lane_pos2(sim.cur_pos, sim.cur_angle)
        except Exception:
            return -10.0 

        v = sim.speed
        reward_speed = 2.0 * v * lp.dot_dir
        
        if lp.dist < self.wrong_lane_limit:
            reward_distance = -30.0 
        else:
            reward_distance = -15.0 * (lp.dist - self.target_offset)**2
        
        reward_survival = 1.0 if v > 0.05 else -1.0
        
        
        return reward_speed + reward_distance + reward_survival

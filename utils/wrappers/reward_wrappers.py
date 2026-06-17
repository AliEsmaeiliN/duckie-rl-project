import gymnasium as gym
import numpy as np

class DtRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -15.0

        return reward

class AdaptiveRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.WRONG_LANE_LIMIT = - 0.12

    def reward(self, reward):

        if reward == -1000:
            return -20

        # Get internal simulator state for custom math
        sim = self.env.unwrapped 
        pos = sim.cur_pos
        angle = sim.cur_angle
        speed = sim.speed
        
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
        
        reward_survival = 2

        total_reward = reward_speed + reward_alignment + reward_distance + reward_angle  + reward_survival
 
        return total_reward
    
    
class UnifiedReward(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.wrong_lane_limit = -0.2
        self.max_expected_angle = 30.0

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
        
        
        reward_distance = -15.0 * np.abs(lp.dist)
        
        reward_survival = 1.0 if v > 0.05 else -1.0
        
        return reward_speed + reward_distance + reward_survival
        
class UnifiedRewardv1(gym.RewardWrapper):
    def __init__(self, env, target_offset= -0.02):
        super().__init__(env)
        self.target_offset = 0 
        self.wrong_lane_limit = 0.2
        self.max_expected_angle = 30.0

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
        
        
        normalized = np.clip(np.abs(lp.dist - self.target_offset) / np.abs(self.wrong_lane_limit), 0.0, 1.0)
        reward_distance = -4 * normalized**2
        
        reward_survival = 1.0 if v > 0.05 else -1.0
        
        return reward_speed + reward_distance + reward_survival
    
class UnifiedRewardv2(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.deadzone = 0.02
        self.max_lane_deviation = 0.2

    def reward(self, reward):

        if reward == -1000:
            return -10
        
        sim = self.env.unwrapped
        try:
            lp = sim.get_lane_pos2(sim.cur_pos, sim.cur_angle)
        except Exception:
            return -10.0 

        v = sim.speed

        cross_track_error = np.abs(lp.dist)

        reward_progress = 1.5 * v * lp.dot_dir if v > 0.05 else -0.5
        
        
        if cross_track_error <= self.deadzone:
            reward_distance = 0.0
        else:
            effective_error = cross_track_error - self.deadzone
            normalized_error = np.clip(effective_error / self.max_lane_deviation, 0.0, 1.0)
            
            reward_distance = -10.0 * (1.0 - np.exp(-5.0 * (normalized_error ** 2)))

        reward_heading = -2.0 * (1.0 - lp.dot_dir)

        reward_survival = 1.0
        
        return reward_progress + reward_distance + reward_survival + reward_heading

class AdditiveJerkPenalty(gym.Wrapper):
    """
    Penalizes large changes between consecutive actions.
    Can be stacked on top of any existing RewardWrapper.
    """
    def __init__(self, env, v_jerk_coeff=-0.5, omega_jerk_coeff=-5):
        super().__init__(env)
        self.v_jerk_coeff = v_jerk_coeff
        self.omega_jerk_coeff = omega_jerk_coeff
        self.prev_action = np.zeros(env.action_space.shape)
        
    def step(self, action):
        
        v_curr, omega_curr = action
        v_prev, omega_prev = self.prev_action

        v_diff = np.abs(v_curr - v_prev)
        omega_diff = np.abs(omega_curr - omega_prev)

        jerk_penalty = (self.v_jerk_coeff * v_diff**2) + (self.omega_jerk_coeff * omega_diff**2)

        self.prev_action = action.copy()

        obs, reward, done, truncated, info = self.env.step(action)

        total_reward = reward + jerk_penalty

        return obs, total_reward, done, truncated, info
    
    def reset(self, **kwargs):

        self.prev_action = np.zeros(self.env.action_space.shape)
        return self.env.reset(**kwargs)
        

def compute_hybrid_eval_reward(sim, current_action, prev_action, return_components=False):
    """
    Hybrid Evaluation Matrix for Checkpoint Selection.
    Prioritizes Task Success (Progress + Survival) with a linear penalty landscape,
    using control smoothness strictly as a minor tie-breaker.
    """
    CATASTROPHIC_PENALTY = -2.0

    try:
        lp = sim.get_lane_pos2(sim.cur_pos, sim.cur_angle)
        cte         = lp.dist
        dot_dir     = lp.dot_dir
        angle_deg   = lp.angle_deg
    except Exception:
        if return_components:
            return CATASTROPHIC_PENALTY, 0.0, 0.0, 0.0, 0.0
        return CATASTROPHIC_PENALTY

    max_achievable_speed = sim.robot_speed * 0.8
    normalized_speed = np.clip(sim.speed / max_achievable_speed, 0.0, 1.0)
    r_progress = normalized_speed * max(0.0, dot_dir)

    max_deviation = 0.20
    r_lane = 1.0 - np.clip(np.abs(cte + 0.02) / max_deviation, 0.0, 1.0)

    max_expected_angle = 45.0 
    r_heading = 1.0 - np.clip(np.abs(angle_deg) / max_expected_angle, 0.0, 1.0)

    delta_omega = current_action[1] - prev_action[1]
    max_delta_omega = 2.0
    r_smoothness = -(delta_omega / max_delta_omega) ** 2

    w_progress   = 0.45
    w_lane       = 0.30
    w_heading    = 0.15
    w_smoothness = 0.10

    total_score = (
        (w_progress * r_progress) +
        (w_lane * r_lane) +
        (w_heading * r_heading) +
        (w_smoothness * r_smoothness)
    )

    if return_components:
        return total_score, r_progress, r_lane, r_heading, r_smoothness
    return total_score
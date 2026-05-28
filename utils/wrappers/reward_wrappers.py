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
    

class SimpleRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        if reward == -1000:
            return -30

        # Get internal simulator state for custom math
        sim = self.env.unwrapped 
        pos = sim.cur_pos
        angle = sim.cur_angle
        speed = sim.speed
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
        

        return reward_speed + reward_alignment + reward_distance + reward_angle
        

class AdaptiveRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

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
        

        return reward_speed + reward_alignment + reward_distance + reward_angle 

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
        
        
        return reward_speed + reward_distance + reward_survival 
    
class PIDReward(gym.RewardWrapper):
    def __init__(self, env, target_offset= -0.02):
        super().__init__(env)
        self.target_offset = target_offset 
        self.wrong_lane_limit = -0.2

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
            reward_distance = -8.0 
        else:
            reward_distance = -15.0 * (lp.dist - self.target_offset)**2
        
        reward_survival = 1.0 if v > 0.05 else -1.0
        
        
        return reward_speed + reward_distance + reward_survival
    
class SingleDirectionReward(gym.RewardWrapper):
    def __init__(self, env, target_offset= -0.02):
        super().__init__(env)
        self.target_offset = target_offset 
        self.wrong_lane_limit = -0.2

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
        
        
        reward_distance = -15.0 * (lp.dist - self.target_offset)**2
        
        reward_survival = 1.0 if v > 0.05 else -1.0
        
        
        return reward_speed + reward_distance + reward_survival 
    
class UnifiedReward(gym.RewardWrapper):
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
        
        
        normalized = np.clip(np.abs(lp.dist - self.target_offset) / np.abs(self.wrong_lane_limit), 0.0, 1.0)
        reward_distance = -10 * normalized**2
        
        reward_survival = 1.0 if v > 0.05 else -1.0
        
        return reward_speed + reward_distance + reward_survival
        
class UnifiedRewardv1(gym.RewardWrapper):
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
        
        
        normalized = np.clip(np.abs(lp.dist - self.target_offset) / np.abs(self.wrong_lane_limit), 0.0, 1.0)
        reward_distance = -10 * normalized**2
        
        reward_survival = 1.0 if v > 0.05 else -1.0
        
        return reward_speed + reward_distance + reward_survival
    
class UnifiedRewardv2(gym.RewardWrapper):
    def __init__(self, env, target_offset= -0.02):
        super().__init__(env)
        self.target_offset = target_offset 
        self.wrong_lane_limit = -0.15
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
        reward_speed = 3.0 * v * max(0.0, lp.dot_dir)
        
        dist_error = np.abs(lp.dist - self.target_offset)
        normalized = np.clip( dist_error / np.abs(self.wrong_lane_limit), 0.0, 1.0)
        reward_distance = -8 * normalized

        normalized_angle = np.clip(np.abs(lp.angle_deg) / self.max_expected_angle, 0.0, 1.0)
        reward_heading = -3.0 * normalized_angle
        
        lane_quality = (1.0 - np.clip(dist_error / np.abs(self.wrong_lane_limit), 0.0, 1.0))
        alignment    = (1.0 - normalized_angle)
        reward_lane  = 2.0 * lane_quality * alignment
        
        return reward_speed + reward_distance + reward_lane + reward_heading

class AdditiveJerkPenalty(gym.RewardWrapper):
    """
    Penalizes large changes between consecutive actions.
    Can be stacked on top of any existing RewardWrapper.
    """
    def __init__(self, env, jerk_coeff=-2):
        super().__init__(env)
        self.jerk_coeff = jerk_coeff
        self.prev_action = np.zeros(env.action_space.shape)
        
    def reward(self, reward):
        
        current_action = self.env.unwrapped.last_action 
        
        action_diff = np.linalg.norm(current_action - self.prev_action)
        jerk_penalty = self.jerk_coeff * action_diff

        self.prev_action = current_action.copy()

        return reward + jerk_penalty
        

def compute_unified_eval_reward(sim, current_action, prev_action, return_components=False):
    """
    Unified Evaluation Reward Matrix for Duckietown Oval Layout.
    Provides a strictly monotonic, normalized objective metric for policy ranking.
    """
    CATASTROPHIC_PENALTY = -2.0 
    
    try:
        lp = sim.get_lane_pos2(sim.cur_pos, sim.cur_angle)
        distance = lp.dist 
        dot_dir = lp.dot_dir
        angle_deg = lp.angle_deg
    except Exception:
        if return_components:
            return CATASTROPHIC_PENALTY, 0.0, 0.0, 0.0, 0.0
        return CATASTROPHIC_PENALTY

    max_deviation = 0.15
    target_offset = -0.02
    max_expected_angle = 30.0

    dist_error = np.abs(distance - target_offset)
    r_distance = -(dist_error / np.abs(max_deviation))

    max_achievable_speed = sim.robot_speed * 0.8
    normalized_speed = np.clip(sim.speed / max_achievable_speed, 0.0, 1.0)
    
    r_progress = normalized_speed * max(0.0, dot_dir)

    r_heading = -(np.abs(angle_deg) / max_expected_angle)

    lane_quality = max(0.0, 1.0 - (dist_error / np.abs(max_deviation)))
    alignment    = max(0.0, 1.0 - (np.abs(angle_deg) / max_expected_angle))
    r_lane_bonus = lane_quality * alignment


    w_progress, w_distance, w_heading, w_lane = 0.3, 0.3, 0.2, 0.2
    total_step_reward = ((w_progress * r_progress) + 
                        (w_distance * r_distance) + 
                        (w_heading * r_heading) + 
                        (w_lane * r_lane_bonus))
    
    total_step_reward = np.clip(total_step_reward, -2.0, 1.0)

    if return_components:
        return (total_step_reward, 
                w_progress * r_progress, 
                w_distance * r_distance, 
                w_heading * r_heading, 
                w_lane * r_lane_bonus)
    
    return total_step_reward


def compute_stability_eval_reward(sim, current_action, prev_action, return_components=False):
    """
    Evaluation Metric.
    """
    CATASTROPHIC_PENALTY = -5.0

    try:
        lp = sim.get_lane_pos2(sim.cur_pos, sim.cur_angle)
        dist         = lp.dist
        heading_err = lp.angle_rad
    except Exception:
        if return_components:
            return CATASTROPHIC_PENALTY, -1.0, -1.0, -1.0
        return CATASTROPHIC_PENALTY

    sigma_d   = 0.05   # 5cm lane center tolerance
    sigma_h   = 0.30   # ~17° heading tolerance
    cte = dist + 0.02
    r_lane     = np.exp(-(cte / sigma_d)**2)
    r_heading  = np.exp(-(heading_err / sigma_h)**2)
    r_trajectory = 0.6 * r_lane + 0.4 * r_heading 

    delta_omega     = current_action[1] - prev_action[1]
    max_delta_omega = 2.0
    
    r_jerk   = -(delta_omega / max_delta_omega)**2
    r_effort = -(current_action[1])**2
    r_smoothness = 0.7 * r_jerk + 0.3 * r_effort

    target_speed = 0.4
    sigma_v      = 0.15
    r_velocity   = np.exp(-((sim.speed - target_speed) / sigma_v)**2)

    w_trajectory = 0.5
    w_smoothness = 0.3
    w_velocity   = 0.2

    total_score = (
        w_trajectory * r_trajectory +
        w_smoothness * r_smoothness +
        w_velocity   * r_velocity
    )
    total_score = np.clip(total_score, -2.0, 1.0)

    if return_components:
        return total_score, r_trajectory, r_smoothness, r_velocity
    return total_score

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
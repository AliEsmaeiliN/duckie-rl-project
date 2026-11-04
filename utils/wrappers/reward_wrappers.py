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
            target_offset = 0.05
            alignment_k = 5.0
        else:
            # "Race Mode" for straights
            speed_coeff = 2.5
            dist_coeff = -10.0
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
    
class UnifiedRewardv1(gym.RewardWrapper):
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
            
            reward_distance = -5.0 * (1.0 - np.exp(-5.0 * (normalized_error ** 2)))

        reward_heading = -2.0 * (1.0 - lp.dot_dir)

        reward_survival = 1.0
        
        return reward_progress + reward_distance + reward_survival + reward_heading

class UnifiedRewardv2(gym.RewardWrapper):
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

class UnifiedRewardV3(gym.RewardWrapper):
    """
    Handles: progress, cross-track, and heading.
    """
    def __init__(self, env,
                 k_speed=1.5,
                 k_dist=2.0,
                 k_heading=4.0,
                 deadzone=0.02,
                 max_dev=0.18,
                 max_expected_angle=40.0):
        super().__init__(env)
        self.k_speed   = k_speed
        self.k_dist    = k_dist
        self.k_heading = k_heading
        self.deadzone  = deadzone
        self.max_dev   = max_dev
        self.max_expected_angle = max_expected_angle

    def reward(self, reward):
        if reward == -1000:
            return -10.0

        sim = self.env.unwrapped
        try:
            lp = sim.get_lane_pos2(sim.cur_pos, sim.cur_angle)
        except Exception:
            return -10.0

        v = sim.speed

        r_speed = self.k_speed * v * lp.dot_dir if v > 0.05 else -0.5

        eff_err  = max(abs(lp.dist) - self.deadzone, 0.0)
        norm_err = np.clip(eff_err / self.max_dev, 0.0, 1.0)
        r_dist   = -self.k_dist * (1.0 - np.exp(-4.0 * norm_err ** 2))

        norm_angle = np.clip(abs(lp.angle_deg) / self.max_expected_angle, 0.0, 1.0)
        r_heading  = -self.k_heading * norm_angle ** 2

        r_survival = 1.0 if v > 0.05 else 0.0

        return r_speed + r_dist + r_heading + r_survival


class UnifiedLineReward(gym.Wrapper):
    """
    UnifiedReward plus a graded penalty for the robot body crossing the lane lines.

    On tight right turns the wheels can clip the white line while the forward camera
    still sees only road, so termination arrives without visual warning. This term
    instead charges a cost that grows with how far the body's footprint is past the
    line, starting `margin` before the paint, so the agent gets a dense, Markov signal
    (it depends only on the current pose) and learns to keep clearance. Termination
    itself is unchanged; use this instead of RecoveryTrainingWrapper.

    Line positions come from the road texture: on a 0.585 m tile the white line's
    inner edge sits 0.114 m right of the lane center, the yellow line's 0.104 m left.

    Keep k_white + k_yellow below the ~1.5/step that centered driving earns: with gamma=0.99,
    any routine state worth less than about -0.1/step makes the -10 termination the better
    deal and the agent learns to crash early (return flat near zero).
    """
    WHITE_EDGE = 0.114
    YELLOW_EDGE = 0.104

    def __init__(self, env, k_white=1.0, k_yellow=0.5, margin=0.0, line_scale=0.04):
        super().__init__(env)
        self.wrong_lane_limit = 0.2
        self.k_white = k_white
        self.k_yellow = k_yellow
        self.margin = margin          # start charging this far (m) before the paint
        self.line_scale = line_scale  # excess (m) at which the penalty saturates
        self._polylines = {}          # (i, j, curve index) -> sampled lane points and tangents

    def reset(self, **kwargs):
        self._polylines.clear()  # the map may change between episodes (multimap)
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward, white_excess, yellow_excess = self._reward(reward)
        info["line_excess_white"] = white_excess
        info["line_excess_yellow"] = yellow_excess
        return obs, reward, terminated, truncated, info

    def _footprint_excess(self, sim):
        """
        Max distance (m) the robot is past the white / yellow boundary, measured at the points
        termination checks (_valid_pose): both wheels and the front. Rectangle corners would
        swing 2-3 cm further out with any heading error or on tight curves and penalize normal
        centered driving.
        """
        from gym_duckietown.simulator import ROBOT_LENGTH, ROBOT_WIDTH, _actual_center, get_dir_vec, get_right_vec

        angle = sim.cur_angle
        center = _actual_center(sim.cur_pos, angle)
        f = 0.5 * ROBOT_LENGTH * get_dir_vec(angle)
        r = 0.5 * ROBOT_WIDTH * get_right_vec(angle)
        corners = (center - r, center + r, center + f)

        white_limit = self.WHITE_EDGE - self.margin
        yellow_limit = self.YELLOW_EDGE - self.margin
        dir_vec = 2.0 * f / ROBOT_LENGTH
        white, yellow = 0.0, 0.0
        for p in corners:
            d = self._signed_lane_dist(sim, p, angle, dir_vec)
            if d is None:
                # Corner is over a non-drivable tile: the full line is already crossed
                return self.line_scale, yellow
            white = max(white, d - white_limit)
            yellow = max(yellow, -d - yellow_limit)
        return white, yellow

    def _signed_lane_dist(self, sim, p, angle, dir_vec):
        """
        Signed distance from p to the lane curve sim.closest_curve_point would pick, using a cached
        dense sampling of the curve (the recursive Bezier search is ~10x slower). Same sign as
        get_lane_pos2: positive is RIGHT of the lane, toward the white line, despite that
        function's comment. None when p is not on a drivable tile.
        """
        i, j = sim.get_grid_coords(p)
        tile = sim._get_tile(i, j)
        if tile is None or not tile["drivable"]:
            return None
        if tile["kind"] == "4way":  # direction-dependent curve choice, use the simulator's lookup
            point, tangent = sim.closest_curve_point(p, angle)
            return None if point is None else float(np.dot(p - point, np.cross(tangent, [0.0, 1.0, 0.0])))

        curves = tile["curves"]
        idx = int(np.argmax((curves[:, -1, :] - curves[:, 0, :]) @ dir_vec))
        key = (i, j, idx)
        if key not in self._polylines:
            from gym_duckietown.graphics import bezier_point, bezier_tangent
            ts = np.linspace(0.0, 1.0, 64)
            pts = np.array([bezier_point(curves[idx], t) for t in ts])
            normals = np.array([np.cross(bezier_tangent(curves[idx], t), [0.0, 1.0, 0.0]) for t in ts])
            self._polylines[key] = (pts, normals)
        pts, normals = self._polylines[key]
        k = int(np.argmin(((pts - p) ** 2).sum(axis=1)))
        return float(np.dot(p - pts[k], normals[k]))

    def _reward(self, reward):
        if reward == -1000:
            return -10.0, 0.0, 0.0

        sim = self.env.unwrapped
        try:
            lp = sim.get_lane_pos2(sim.cur_pos, sim.cur_angle)
        except Exception:
            return -10.0, 0.0, 0.0

        # --- UnifiedReward terms ---
        v = sim.speed
        reward_speed = 2.0 * v * lp.dot_dir
        normalized = np.clip(np.abs(lp.dist) / self.wrong_lane_limit, 0.0, 1.0)
        reward_distance = -4 * normalized**2
        reward_survival = 1.0 if v > 0.05 else -1.0

        # --- Line clearance ---
        white, yellow = self._footprint_excess(sim)
        reward_line = (-self.k_white * np.clip(white / self.line_scale, 0.0, 1.0)
                       - self.k_yellow * np.clip(yellow / self.line_scale, 0.0, 1.0))

        return (reward_speed + reward_distance + reward_survival + reward_line,
                max(white, 0.0), max(yellow, 0.0))

class AdditiveJerkPenalty(gym.Wrapper):
    """
    Penalizes large changes between consecutive actions.
    Can be stacked on top of any existing RewardWrapper.
    """
    def __init__(self, env, v_jerk_coeff=-0.5, omega_jerk_coeff=-6):
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
    r_lane = 1.0 - np.clip(np.abs(cte) / max_deviation, 0.0, 1.0)

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
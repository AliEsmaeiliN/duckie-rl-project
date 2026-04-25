import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image
import numpy as np 


class TemporalWrapper(gym.Wrapper):
    def __init__(self, env=None, frame_skip=3, motion_blur=True):
        super().__init__(env)
        self.frame_skip = frame_skip
        self.motion_blur = motion_blur
        self.unwrapped.delta_time = self.unwrapped.delta_time / (self.frame_skip + 1)
        
        self.weights = [0.01, 0.04, 0.15, 0.8]  
        
    def step(self, action: np.ndarray):
        action = np.clip(action, -1, 1)
        motion_blur_window = []
        processed_action = self.env.action(action)
        if hasattr(self.env, 'action'):
            processed_action = self.env.action(action)

        for _ in range(self.frame_skip + 1):
            obs = self.unwrapped.render_obs()
            motion_blur_window.append(obs)

            self.unwrapped.update_physics(processed_action)
            
        if not self.motion_blur:
            processed_obs = motion_blur_window[-1]
        else:
            current_weights = self.weights[:len(motion_blur_window)]
            if np.sum(current_weights) == 0:
                processed_obs = motion_blur_window[-1]
            else:
                processed_obs = np.average(
                    motion_blur_window, 
                    axis=0, 
                    weights=current_weights
                ).astype(np.uint8)


        d_info = self.unwrapped._compute_done_reward(processed_action)

        return processed_obs, d_info.reward, d_info.done, False, self.unwrapped.get_agent_info()


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
            reward = -15.0

        return reward


# this is needed because at max speed the duckie can't turn anymore
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        action_ = np.array([action[0] * 0.8, action[1]], dtype=np.float32)
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
    
class CustomRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_action = np.zeros(2)

    def reward(self, reward):

        if reward <= -15.0:
            return reward

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
        in_danger_zone = (direction == "CW") and approaching_curve



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
            jerk_coeff = -0.5
            target_offset = 0.0
            alignment_k = 2.0
        
        reward_speed = speed_coeff * speed * lp.dot_dir
        reward_alignment = np.exp(alignment_k * (lp.dot_dir - 1.0)) # tanh like behaviour to add a higher gradint near 1
        reward_distance = dist_coeff * (lp.dist - target_offset)**2
        reward_angle = -0.03 * np.abs(lp.angle_deg)
        
        action_diff = np.linalg.norm(current_action - self.prev_action) 
        reward_jerk = jerk_coeff * action_diff

        self.prev_action = current_action.copy()

        return reward_speed + reward_alignment + reward_distance + reward_angle + reward_jerk
    

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

class KinematicActionWrapper(gym.ActionWrapper):
    def __init__(self, env, gain=1.0, trim=0.0, wheel_dist=0.102, radius=0.0318, k=27.0, limit=1.0):
        super().__init__(env)
        self.gain = gain
        self.trim = trim
        self.radius = radius
        self.k = k
        self.limit = limit
        self.wheel_dist = wheel_dist

    def action(self, action):
        # Action is [v, omega] from the RL Agent
        vel, angle = action

        # Adjust motor constants by gain and trim
        k_r_inv = (self.gain + self.trim) / self.k
        k_l_inv = (self.gain - self.trim) / self.k

        # Calculate angular velocities for wheels
        omega_r = (vel + 0.5 * angle * self.wheel_dist) / self.radius
        omega_l = (vel - 0.5 * angle * self.wheel_dist) / self.radius

        # Convert to duty cycle (PWM)
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # Apply physical limits (max motor power)
        u_r_limited = np.clip(u_r, -self.limit, self.limit)
        u_l_limited = np.clip(u_l, -self.limit, self.limit)

        return np.array([u_l_limited, u_r_limited], dtype=np.float32)
    
class UndistortWrapper(gym.ObservationWrapper):
    """
    To Undo the Fish eye transformation - undistorts the image with plumbbob distortion
    Using the default configuration parameters on the duckietown/Software repo
    https://github.com/duckietown/Software/blob/master18/catkin_ws/src/
    ...05-teleop/pi_camera/include/pi_camera/camera_info.py
    """

    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)


        # Set a variable in the unwrapped env so images don't get distorted
        self.env.unwrapped.undistort = False

        # K - Intrinsic camera matrix for the raw (distorted) images.
        camera_matrix = [
            311.5665681454016, 0.0, 335.2914198639567,
            0.0, 308.44370374450375, 235.41469946322758,
            0.0, 0.0, 1.0
        ]
        self.camera_matrix = np.reshape(camera_matrix, (3, 3))

        # distortion parameters - (k1, k2, t1, t2, k3)
        distortion_coefs = [
            -0.2605492415446591, 0.05392162341209542, 
            0.0011529115993347476, -0.004728714280095291, 0.0
        ]
        self.distortion_coefs = np.reshape(distortion_coefs, (1, 5))

        # R - Rectification matrix - stereo cameras only, so identity
        self.rectification_matrix = np.eye(3)

        # P - Projection Matrix - specifies the intrinsic (camera) matrix
        #  of the processed (rectified) image
        projection_matrix = [
            223.22879028320312, 0.0, 327.2591191800675, 0.0,
            0.0, 247.4501953125, 233.82550662924768, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        self.projection_matrix = np.reshape(projection_matrix, (3, 4))

        # Initialize mappings

        # Used for rectification
        self.mapx = None
        self.mapy = None

    def observation(self, observation):
        if self.env.unwrapped.distortion:
            return self._undistort(observation)
        return observation

    def _undistort(self, observation):
        if self.mapx is None:
            # Not initialized - initialize all the transformations we'll need
            self.mapx = np.zeros(observation.shape)
            self.mapy = np.zeros(observation.shape)

            H, W, _ = observation.shape

            # Initialize self.mapx and self.mapy (updated)
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                self.camera_matrix,
                self.distortion_coefs,
                self.rectification_matrix,
                self.projection_matrix,
                (W, H),
                cv2.CV_32FC1,
            )

        return cv2.remap(observation, self.mapx, self.mapy, cv2.INTER_NEAREST)
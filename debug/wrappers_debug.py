import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image
import numpy as np 
import cv2
import collections

def save_debug_image(obs, name, folder="captures/observation_pipeline"):
    """Internal helper to save observations for debugging."""
    
    arr = np.array(obs)
    
    if len(arr.shape) == 3 and arr.shape[0] in [1, 3, 4]:
        if arr.shape[0] == 4:
            arr = arr[-1, :, :]
        else:
            arr = arr.transpose(1, 2, 0)
            
    if arr.dtype == np.float32:
        arr = (arr * 255).astype(np.uint8)
        
    img = Image.fromarray(arr.squeeze())
    img.save(f"{folder}/{name}.png")

class RewardCompute():
    def __init__(self):
        self.WRONG_LANE_LIMIT = -0.20 # Meters (standard lane width is ~0.2-0.3)

    def compute(self, name, **kwargs):
        """Registry to call the correct reward function by string name."""
        funcs = {
            "adl": self.adl_reward,
            "simple": self.simple_reward,
            "adp": self.adp_reward,
            "dbg": self.dbg_reward
        }
        if name not in funcs:
            raise ValueError(f"Reward type '{name}' not recognized.")
        return funcs[name](**kwargs)

    def adl_reward(self, speed, distance, heading, angle, danger_zone, current_action, previous_action):
        if distance < self.WRONG_LANE_LIMIT:
            return -20.0, 0, 0, 0, 0 # Return tuple to maintain consistency

        if danger_zone:
            speed_coeff, dist_coeff, jerk_coeff, target_offset, alignment_k = 1.0, -15.0, -1.2, 0.05, 5.0
        else:
            speed_coeff, dist_coeff, jerk_coeff, target_offset, alignment_k = 2.5, -10.0, -2.5, 0.0, 2.0
        
        reward_speed = -3.0 if speed < 0.1 else speed_coeff * speed * heading
        reward_alignment = np.exp(alignment_k * (heading - 1.0))
        reward_distance = -40.0 * (distance ** 2) if distance < 0 else dist_coeff * (distance - target_offset) ** 2
        reward_angle = -0.03 * np.abs(angle)
        reward_jerk = jerk_coeff * np.linalg.norm(current_action - previous_action)

        return reward_speed, reward_distance, reward_alignment, reward_angle, reward_jerk
    
    def simple_reward(self, speed, distance, heading, angle, danger_zone, current_action, previous_action):
        speed_coeff, dist_coeff, jerk_coeff, alignment_k = 1.5, -15.0, -1.0, 2.0
        
        reward_speed = speed_coeff * speed
        reward_alignment = alignment_k * (heading ** 2) if heading > 0 else 4.0 * heading
        if np.abs(distance) >= np.abs(self.WRONG_LANE_LIMIT) / 2:
            dist_coeff = -40
            
        reward_distance = dist_coeff * np.abs(distance)
        reward_angle = -0.03 * np.abs(angle)
        reward_jerk = jerk_coeff * np.linalg.norm(current_action - previous_action)

        return reward_speed, reward_distance, reward_alignment, reward_angle, reward_jerk

    def adp_reward(self, speed, distance, heading, angle, danger_zone, current_action, previous_action):
        dist_penalty_coeff, speed_coeff, jerk_coeff, k = -8.0, 2.0, -0.3, 5.0

        if danger_zone:
            speed_coeff, dist_penalty_coeff, jerk_coeff, k = 0.4, -15.0, -1.5, 10.0                      
        
        reward_speed = speed_coeff * speed * heading
        reward_alignment = np.exp(k * (heading - 1.0))
        reward_distance = dist_penalty_coeff * np.abs(distance)
        reward_angle = -0.03 * np.abs(angle)
        reward_jerk = jerk_coeff * np.linalg.norm(current_action - previous_action)

        return reward_speed, reward_distance, reward_alignment, reward_angle, reward_jerk
    
    def dbg_reward(self, speed, distance, heading, angle, danger_zone, current_action, previous_action):

        if danger_zone:
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
            reward_speed = speed_coeff * speed * heading

        reward_alignment = np.exp(alignment_k * (heading - 1.0)) # tanh like behaviour to add a higher gradint near 1

        if distance < self.WRONG_LANE_LIMIT / 2 :
            dist_coeff = -30

        reward_distance = 0.5 + dist_coeff * (distance - target_offset) ** 2
        reward_angle = -0.03 * np.abs(angle)
        action_diff = np.linalg.norm(current_action - previous_action)
        reward_jerk = jerk_coeff * action_diff 

        return reward_speed, reward_distance, reward_alignment, reward_angle, reward_jerk

        


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
        step = self.unwrapped.step_count
        action_ = np.array([action[0] * 0.8, action[1]], dtype=np.float32)
            
        #print(f"Action wrapper at step {step} : {action_}")
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
    
class DebugRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, reward_type="adl"):
        super().__init__(env)
        self.reward_type = reward_type
        self.reward_compute = RewardCompute()
        self.prev_action = np.zeros(2)
        self.latest_reward_components = {}

    def reward(self, reward):
        # Handle termination/crash rewards from the simulator directly
        if reward <= -15.0: 
            return reward

        sim = self.env.unwrapped 
        pos = sim.cur_pos
        angle = sim.cur_angle
        
        try:
            lp = sim.get_lane_pos2(pos, angle)
        except Exception:
            return -10.0 
            
        # Danger Zone Logic (Asymmetric behavior for Clockwise turns)
        coords = sim.get_grid_coords(pos)
        tile = sim._get_tile(*coords)
        tile_kind = tile["kind"] if tile else ""
        
        # Lookahead Logic to detect upcoming curves
        lookahead_dist = 0.25 
        dir_vec = np.array([np.cos(angle), 0, -np.sin(angle)])
        lookahead_pos = pos + dir_vec * lookahead_dist
        look_tile = sim._get_tile(*sim.get_grid_coords(lookahead_pos))
        look_kind = look_tile["kind"] if look_tile else ""

        in_danger_zone = (sim.episode_dir == "CW") and ("curve" in look_kind)

        # Compute components
        comps = self.reward_compute.compute(
            name=self.reward_type,
            speed=sim.speed,
            distance=lp.dist,
            heading=lp.dot_dir,
            angle=lp.angle_deg,
            danger_zone=in_danger_zone,
            current_action=sim.last_action,
            previous_action=self.prev_action
        )

        if isinstance(comps, tuple):
            r_speed, r_dist, r_align, r_angle, r_jerk = comps
            total_reward = sum(comps)
            
            self.latest_reward_components = {
                "speed": r_speed,
                "distance": r_dist,
                "alignment": r_align,
                "angle": r_angle,
                "jerk": r_jerk,
                "raw_dist": lp.dist,
                "raw_heading": lp.dot_dir
            }
        else:
            total_reward = comps

        self.prev_action = sim.last_action.copy()
        return total_reward

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
        step = self.unwrapped.step_count

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
            
        #print(f"Kinematic wrapper at step {step} : {[u_l_limited, u_r_limited]}")

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
    
class ActionLatencyWrapper(gym.Wrapper):
    def __init__(self, env, min_latency=1, max_latency=4):
        """
        min_latency/max_latency: steps to delay an action.
        If the simulator runs at 30Hz with frame_skip=4, 
        1 step delay is approx 33ms.
        """
        super().__init__(env)
        self.min_latency = min_latency
        self.max_latency = max_latency
        self.current_latency = min_latency
        self.action_buffer = collections.deque()
        
    def reset(self, **kwargs):
        self.current_latency = np.random.randint(self.min_latency, self.max_latency + 1)
        self.action_buffer = collections.deque(
        [np.zeros(self.action_space.shape)] * self.current_latency
        )
        return self.env.reset(**kwargs)

    def step(self, action):
        self.action_buffer.append(action)   # Push current intent to back
        exec_action = self.action_buffer.popleft() # Pop oldest intent from front
        return self.env.step(exec_action)
    
class RecoveryTrainingWrapper(gym.Wrapper):
    """
    Intercepts termination when the agent goes out of bounds.
    Keeps the episode alive for `max_recovery_steps` to teach the agent to recover.
    """
    def __init__(self, env, max_recovery_steps=30, ood_penalty=-10.0):
        super().__init__(env)
        self.recovery_steps = 0
        self.max_recovery_steps = max_recovery_steps
        self.ood_penalty = ood_penalty

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        is_ood = done and (reward <= -1000 or info.get("Simulator", {}).get("done_code") == "invalid-pose")

        if is_ood or self.recovery_steps > 0:
            self.recovery_steps += 1
            
            reward = self.ood_penalty
            
            if self.recovery_steps < self.max_recovery_steps:
                done = False
                
                try:
                    sim = self.unwrapped
                    lp = sim.get_lane_pos2(sim.cur_pos, sim.cur_angle)
                    if abs(lp.dist) < 0.18: 
                        self.recovery_steps = 0
                except Exception:
                    pass 
            else:
                # Agent failed to recover in time
                done = True
        else:
            self.recovery_steps = 0
            
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        self.recovery_steps = 0
        return self.env.reset(**kwargs)
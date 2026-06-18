import gymnasium as gym
import numpy as np
import collections

class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        action_ = np.array([action[0] * 0.8, action[1]], dtype=np.float32)
        return action_
    
class KinematicActionWrapper(gym.ActionWrapper):
    def __init__(self, env, gain=1.0, trim=0.0, wheel_dist=0.102, radius=0.0318, k=27.0, limit=1.0, v_scale=0.8, omega_scale=3):
        super().__init__(env)
        self.gain = gain
        self.trim = trim
        self.radius = radius
        self.k = k
        self.limit = limit
        self.wheel_dist = wheel_dist
        self.v_scale = v_scale #adapted from dt action wrapper to scale the speed for turning
        self.omega_scale = omega_scale
        self.k_r_inv = (gain + trim) / self.k
        self.k_l_inv = (gain - trim) / self.k

    def action(self, action):
        # Action is [v, omega] from the RL Agent
        vel, omega = action[0] * self.v_scale,  (action[1] **3) * self.omega_scale


        # Calculate angular velocities for wheels
        omega_r = (vel + 0.5 * omega * self.wheel_dist) / self.radius
        omega_l = (vel - 0.5 * omega * self.wheel_dist) / self.radius

        # Convert to duty cycle (PWM)
        u_r = omega_r * self.k_r_inv
        u_l = omega_l * self.k_l_inv

        # Apply physical limits (max motor power)
        #u_r_limited = np.clip(u_r, -self.limit, self.limit)
        #u_l_limited = np.clip(u_l, -self.limit, self.limit)

        if np.abs(u_r) > self.limit or np.abs(u_l) > self.limit:
            excess = max(np.abs(u_r), np.abs(u_l))
            scale  = self.limit / excess          # uniform scale preserves steering ratio
            u_r   *= scale
            u_l   *= scale

        return np.array([u_l, u_r], dtype=np.float32)
    
class ActionLatencyWrapper(gym.Wrapper):
    def __init__(self, env, min_latency=0, max_latency=1, jitter_prob=0.05):
        """
        At 30Hz + frame_skip=4: 1 step = 133ms.
        Real bot pipeline ~40ms → 0-1 steps covers real lag.
        jitter_prob: 5% chance of CPU/ROS spike repeating last action.
        Updated to adapt to the constant lag within a run introduced with dynamics randomization.
        """
        super().__init__(env)
        self.min_latency = min_latency
        self.max_latency = max_latency
        self.jitter_prob = jitter_prob
        self.action_buffer = collections.deque()
        self.last_action = np.zeros(env.action_space.shape)

    def reset(self, **kwargs):
        self.current_latency = np.random.randint(
            self.min_latency, self.max_latency + 1
        )
        self.action_buffer = collections.deque(
            [np.zeros(self.action_space.shape)] * self.current_latency
        )
        self.last_action = np.zeros(self.action_space.shape)
        return self.env.reset(**kwargs)

    def step(self, action):
        self.action_buffer.append(action)

        if np.random.rand() < self.jitter_prob:
            # CPU spike / ROS delay — hold last action, drain buffer normally
            self.action_buffer.popleft()
            exec_action = self.last_action
        else:
            exec_action = self.action_buffer.popleft()
            self.last_action = exec_action

        return self.env.step(exec_action)
    
class ActionSmoothingWrapper(gym.Wrapper):
    """
    Applies EMA smoothing to actions at the policy level (Slow Loop).
    Matches the agent.py logic on the physical Duckiebot.
    """
    def __init__(self, env, alpha=0.8):
        super().__init__(env)
        self.alpha = alpha
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)

    def reset(self, **kwargs):
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        return self.env.reset(**kwargs)

    def step(self, action):
        smoothed = (self.alpha * action) + ((1.0 - self.alpha) * self.prev_action)
        
        self.prev_action = smoothed.copy()
        
        return self.env.step(smoothed)


class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        action_ = np.array([action[0] * 0.8, action[1]], dtype=np.float32)
        return action_
    
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
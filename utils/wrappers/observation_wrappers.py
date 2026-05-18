import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2

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
        return cv2.resize(
            observation, 
            (self.shape[1], self.shape[0]), 
            interpolation=cv2.INTER_AREA 
        )


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

class CropResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape 
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(self.shape[0], self.shape[1], 3), 
            dtype=np.uint8
        )

    def observation(self, obs):
        
        h, w = obs.shape[:2]
        top_boundary = int(h / 3)
        cropped = obs[top_boundary:h, 0:w]
        
        return cv2.resize(
            cropped, 
            (self.shape[1], self.shape[0]), 
            interpolation=cv2.INTER_AREA
        )

class GrayscaleWrapper(gym.ObservationWrapper):
    """
    Optimized Grayscale Wrapper for Duckietown.
    Converts RGB (H, W, 3) to Grayscale (1, H, W).
    Using OpenCV for maximum speed to minimize control loop latency.
    """
    def __init__(self, env):
        super().__init__(env)
        h, w, _ = self.observation_space.shape
        
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(1, h, w), 
            dtype=np.uint8
        )

    def observation(self, obs):
        
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return np.expand_dims(gray, axis=0)
    
class GaussianBlurWrapper(gym.ObservationWrapper):
    """Matches the Gaussian blur applied on the real bot."""
    def __init__(self, env):
        super().__init__(env)
    def observation(self, obs):
        return cv2.GaussianBlur(obs, (3, 3), 0)
    
class CLAHEWrapper(gym.ObservationWrapper):
    """Simulates real-world lighting variation via CLAHE."""
    def __init__(self, env, clip_limit=2.0):
        super().__init__(env)
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    
    def observation(self, obs):
        yuv = cv2.cvtColor(obs, cv2.COLOR_RGB2YUV)
        yuv[:, :, 0] = self.clahe.apply(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

class  ContrastStretchingWrapper(gym.ObservationWrapper):
    """Dynamically expands the contrast."""
    def __init__(self, env):
        super().__init__(env)
    def observation(self, observation):
        return cv2.normalize(observation, None, 0, 255, cv2.NORM_MINMAX)
    

class FastKinematicBlurWrapper(gym.ObservationWrapper):
    """
    Applies dynamic directional motion blur based on robot's
    physical velocity and angular velocity.
    Updated and faster version of previous Temporal wrapper.
    """
    def __init__(self, env, max_blur_size=7, v_scale=1.0, omega_scale=1.5):
        super().__init__(env)
        self.max_blur_size = max_blur_size
        self.v_scale = v_scale
        self.omega_scale = omega_scale
        # Derived from actual max physical values: v~1.2 m/s, omega~1.0
        self.max_magnitude = np.sqrt((1.2 * v_scale)**2 + (1.0 * omega_scale)**2)

    def observation(self, obs):
        sim = self.unwrapped
        v     = float(sim.speed) if hasattr(sim, 'speed') else 0.0
        omega = float(sim.last_action[1]) if sim.last_action is not None else 0.0

        blur_y = v * self.v_scale
        blur_x = omega * self.omega_scale
        magnitude = np.sqrt(blur_x**2 + blur_y**2)
        magnitude_norm = np.clip(magnitude / self.max_magnitude, 0.0, 1.0)

        if magnitude_norm < 0.05:
            return obs

        kernel_size = max(3, int(magnitude_norm * self.max_blur_size))
        if kernel_size % 2 == 0:
            kernel_size += 1

        angle  = np.arctan2(blur_y, blur_x)
        center = kernel_size // 2

        x1 = int(np.clip(center - np.cos(angle) * center, 0, kernel_size - 1))
        y1 = int(np.clip(center - np.sin(angle) * center, 0, kernel_size - 1))
        x2 = int(np.clip(center + np.cos(angle) * center, 0, kernel_size - 1))
        y2 = int(np.clip(center + np.sin(angle) * center, 0, kernel_size - 1))

        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        cv2.line(kernel, (x1, y1), (x2, y2), 1.0, thickness=1)

        kernel_sum = kernel.sum()
        if kernel_sum == 0:
            return obs
        kernel /= kernel_sum

        if obs.ndim == 3 and obs.shape[0] in (1, 3):
            obs_hw = obs.transpose(1, 2, 0)
            blurred = cv2.filter2D(obs_hw, -1, kernel)
            if blurred.ndim == 2:
                blurred = np.expand_dims(blurred, axis=-1)
            return blurred.transpose(2, 0, 1).astype(obs.dtype)

        return cv2.filter2D(obs, -1, kernel).astype(obs.dtype)
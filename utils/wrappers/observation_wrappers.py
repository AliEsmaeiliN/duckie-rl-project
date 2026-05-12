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
    
class GrayScaleWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        # Change observation space to 1 channel while keeping H, W
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(obs_shape[0], obs_shape[1], 1), 
            dtype=np.uint8
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return gray[:, :, np.newaxis]
    
class UndistortWrapper(gym.ObservationWrapper):
    """
    Undoes the fisheye transformation using plumb_bob distortion.
    Uses the exact calibration parameters from the physical Duckiebot (duckie1nav).
    """
    def __init__(self, env=None):
        super().__init__(env)
        
        # Access the unwrapped environment to check if distortion is enabled
        self.env.unwrapped.undistort = False

        # K - Intrinsic camera matrix (Raw distorted image mapping)
        # Represents focal lengths (fx, fy) and optical centers (cx, cy)
        self.camera_matrix = np.array([
            [560.2421673869314, 0.0, 318.8173154671802],
            [0.0, 564.9631346359404, 235.6814689313625],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # D - Distortion coefficients (k1, k2, p1, p2, k3)
        # k1 = -0.911 indicates severe barrel (fisheye) distortion
        self.distortion_coefs = np.array([
            -0.9111617456077904, 0.603501770314888, 
            -0.014333851834234601, 0.010320245199077559, 0.0
        ], dtype=np.float32)

        # R - Rectification matrix (Identity for a monocular setup)
        self.rectification_matrix = np.eye(3, dtype=np.float32)

        # P - Projection Matrix (Optimal new camera matrix after undistortion)
        # Notice focal lengths drop to ~392/439 to account for cropped curved edges
        self.projection_matrix = np.array([
            [392.5531005859375, 0.0, 326.73844408192963, 0.0],
            [0.0, 439.04815673828125, 220.5653813603385, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ], dtype=np.float32)

        # Caching the remapping grids
        self.mapx = None
        self.mapy = None

    def observation(self, observation):
        # Only apply math if the simulator is actually generating distorted images
        if getattr(self.env.unwrapped, 'distortion', False):
            return self._undistort(observation)
        return observation

    def _undistort(self, observation):
        # Calculate the mapping matrices ONCE on the first frame
        if self.mapx is None:
            h, w = observation.shape[:2]
            
            # Sanity check: Ensure simulation resolution matches calibration (640x480)
            # If the simulator outputs a different shape, this math will warp incorrectly.
            if w != 640 or h != 480:
                # Dynamically adjust the camera matrices to match the scale
                scale_x = w / 640.0
                scale_y = h / 480.0
                
                scaled_camera_matrix = self.camera_matrix.copy()
                scaled_camera_matrix[0, :] *= scale_x
                scaled_camera_matrix[1, :] *= scale_y
                
                scaled_proj_matrix = self.projection_matrix.copy()
                scaled_proj_matrix[0, :] *= scale_x
                scaled_proj_matrix[1, :] *= scale_y
            else:
                scaled_camera_matrix = self.camera_matrix
                scaled_proj_matrix = self.projection_matrix

            self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                scaled_camera_matrix,
                self.distortion_coefs,
                self.rectification_matrix,
                scaled_proj_matrix,
                (w, h),
                cv2.CV_32FC1,
            )

        return cv2.remap(observation, self.mapx, self.mapy, cv2.INTER_LINEAR)

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
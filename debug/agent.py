import torch
import numpy as np
import collections
import cv2
import yaml
from PIL import Image

from models import SACActor, TD3Actor

class DuckiebotAgent:
    def __init__(self, model_path, algo_type="sac", grayscale=True, frame_stack=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.algo_type = algo_type.lower()
        self.grayscale = grayscale
        self.frame_stack = frame_stack
        
        print(f"Loading {self.algo_type.upper()} model from {model_path}...")
        self.frames = collections.deque(maxlen=frame_stack)
        
        if self.algo_type == "sac":
            self.actor = SACActor(grayscale=self.grayscale).to(self.device)
        elif self.algo_type == "td3":
            self.actor = TD3Actor(grayscale=self.grayscale).to(self.device)
        else:
            raise ValueError(f"Unknown algo type: {self.algo_type}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor.eval()

        self.c = 1 if grayscale else 3
        self.frames = collections.deque(maxlen=frame_stack)

        self.veh = "duckie1nav" # get from env
        # self.map_x, self.map_y = self._load_calibration()

    def _load_calibration(self):
        """Loads intrinsic parameters and prepares cv2 maps."""
        calib_path = f"/data/config/calibrations/camera_intrinsic/{self.veh}.yaml"
        
        with open(calib_path, 'r') as f:
            calib_data = yaml.safe_load(f)
            
        intrinsics = np.array(calib_data['camera_matrix']['data']).reshape(3, 3)
        distortion = np.array(calib_data['distortion_coefficients']['data'])
        img_width = calib_data['image_width']
        img_height = calib_data['image_height']

        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            intrinsics, distortion, (img_width, img_height), 0, (img_width, img_height)
        )
        map_x, map_y = cv2.initUndistortRectifyMap(
            intrinsics, distortion, None, new_camera_matrix, (img_width, img_height), cv2.CV_32FC1
        )
        return map_x, map_y

    def preprocess(self, obs_bgr):
        img_rgb = cv2.cvtColor(obs_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
        
        width, height = img.size
        top_boundary = int(height * (1/3))
        img = img.crop((0, top_boundary, width, height))
        
        # 3. Resize to 84x84
        img = img.resize((84, 84), Image.BILINEAR)
        final_np = np.array(img) # Now (84, 84, 3) RGB
        
        if self.grayscale:
            gray = cv2.cvtColor(final_np, cv2.COLOR_RGB2GRAY)
            return gray[np.newaxis, :, :] # (1, 84, 84)
        
        # (3, 84, 84)
        return final_np.transpose(2, 0, 1)
    
    def preprocess_cv(self, obs_rgb):
        """
        Replicates the Sim2Real vision pipeline: 
        """
        
        rectified_img = cv2.remap(obs_rgb, self.map_x, self.map_y, cv2.INTER_LINEAR)

        # ResizeWrapper
        img = cv2.resize(rectified_img, (160, 120), interpolation=cv2.INTER_LINEAR)
        
        # CropResizeWrapper
        h, w = img.shape[:2]
        top_boundary = int(h / 3)
        img = img[top_boundary:h, 0:w]
        
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.grayscale:
            img_processed = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            img_processed = img_processed[np.newaxis, :, :]
        else:
            img_processed = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            img_processed = img_processed.transpose(2, 0, 1)
            
        return img_processed

    def get_action(self, processed_stack_input, is_stacked = False):
        """
        Handles deterministic inference for the physical robot.
        """

        # (C*Stack, 84, 84)
        stacked_input = np.concatenate(list(self.frames), axis=0)
        input_tensor = torch.FloatTensor(stacked_input).unsqueeze(0).to(self.device)

        if is_stacked:
            input_tensor = torch.FloatTensor(processed_stack_input).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.algo_type == "sac":
                # use mean_action
                _, _, action = self.actor.get_action(input_tensor)
            else:
                action = self.actor(input_tensor)
        
        return action.cpu().numpy().reshape(-1)

    def postprocess_kinematics(self, action):
        """
        Translates [v, omega] to physical Wheel Commands [u_l, u_r].
        Replicates ActionWrapper and KinematicActionWrapper.
        """
        v, omega = action[0] * 0.8,  action[1]
        
        # DB21J physical constants
        radius, wheel_dist, k, trim = 0.0318, 0.102, 27.0, -0.1
        
        # Kinematic equations
        u_r = np.clip(((v + 0.5 * omega * wheel_dist) / radius) * (1.0 + trim) / k, -1.0, 1.0)
        u_l = np.clip(((v - 0.5 * omega * wheel_dist) / radius) * (1.0 - trim) / k, -1.0, 1.0)
        
        return [u_l, u_r]
    
    def update_buffer(self, processed_frame):
        """Appends the last frame to the stack"""

        if len(self.frames) == 0:
            for _ in range(self.frame_stack):
                self.frames.append(processed_frame)
        else:
            self.frames.append(processed_frame)
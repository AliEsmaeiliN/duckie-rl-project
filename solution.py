import torch
import numpy as np
import collections
from PIL import Image
import cv2

from rl.sac_continuous_action import Actor as SACActor
from rl.td3_continuous_action import Actor as TD3Actor

class DuckiebotAgent:
    def __init__(self, model_path, algo_type="sac", grayscale=True, frame_stack=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.algo_type = algo_type.lower()
        self.grayscale = grayscale
        self.frame_stack = frame_stack
        
        self.frames = collections.deque(maxlen=frame_stack)
        
        if self.algo_type == "sac":
            self.actor = SACActor(None).to(self.device)
        elif self.algo_type == "td3":
            self.actor = TD3Actor(None).to(self.device)
        else:
            raise ValueError("Could not detect the agent type to upload the actor")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor.eval()

    def preprocess(self, obs_rgb):
        """
        Replicates the Sim2Real vision pipeline: 
        Resize(120,160) -> Crop(bottom 3/4) -> Resize(84,84) -> Gray -> Stack
        """
        
        # ResizeWrapper
        img = cv2.resize(obs_rgb, (160, 120), interpolation=cv2.INTER_LINEAR)
        
        # CropResizeWrapper
        h, w = img.shape[:2]
        top_boundary = int(h * 0.25)
        img = img[top_boundary:h, 0:w]
        
        # 3. Resize to 84x84
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_LINEAR)
        
        # 4. Grayscale & Transpose (ImgWrapper logic)
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # Add channel dim: (84, 84) -> (1, 84, 84)
            img = np.expand_dims(img, axis=0)
        else:
            # RGB Transpose: (84, 84, 3) -> (3, 84, 84)
            img = img.transpose(2, 0, 1)
            
        # 5. Temporal Stacking
        if len(self.frames) == 0:
            for _ in range(self.frame_stack):
                self.frames.append(img)
        else:
            self.frames.append(img)
            
        # 6. Final Tensor Construction
        # Concatenate along the channel axis (axis 0)
        stacked_obs = np.concatenate(list(self.frames), axis=0)
        
        obs_tensor = torch.FloatTensor(stacked_obs).unsqueeze(0).to(self.device)
        return obs_tensor

    def act(self, obs_tensor):
        """
        Handles deterministic inference for the physical robot.
        """
        with torch.no_grad():
            if self.algo_type == "sac":
                # use mean_action
                _, _, action = self.actor.get_action(obs_tensor)
            else:
                action = self.actor(obs_tensor)
        
        return action.cpu().numpy()[0]

    def postprocess_kinematics(self, action):
        """
        Translates [v, omega] to physical Wheel Commands [u_l, u_r].
        Replicates ActionWrapper and KinematicActionWrapper.
        """
        v, omega = action[0] * 0.8, action[1]
        
        # DB21J physical constants
        radius, wheel_dist, k = 0.0318, 0.102, 27.0
        
        # Kinematic equations
        u_r = np.clip(((v + 0.5 * omega * wheel_dist) / radius) / k, -1.0, 1.0)
        u_l = np.clip(((v - 0.5 * omega * wheel_dist) / radius) / k, -1.0, 1.0)
        
        return [u_l, u_r]

if __name__ == "__main__":
    MODEL_PATH = "models/sac_v1/sac_latest_step.cleanrl_model"
    agent = DuckiebotAgent(MODEL_PATH, algo_type="sac")

    print(f"Agent initialized using {agent.algo_type} on {agent.device}")

    # This part is handled by the Duckietown ROS wrapper in reality
    def on_camera_frame(raw_frame):
        obs_tensor = agent.preprocess(raw_frame)
        raw_action = agent.act(obs_tensor)
        wheel_cmd = agent.postprocess_kinematics(raw_action)
        return wheel_cmd
import torch
import numpy as np
import collections
import cv2

from .models import SACActor, TD3Actor

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

        checkpoint = torch.load(model_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor.eval()

        self.c = 1 if grayscale else 3
        self.frames = collections.deque(maxlen=frame_stack)

    def preprocess(self, obs_rgb):
        """
        Replicates the Sim2Real vision pipeline: 
        """
        
        # ResizeWrapper
        img = cv2.resize(obs_rgb, (160, 120), interpolation=cv2.INTER_LINEAR)
        
        # CropResizeWrapper
        h, w = img.shape[:2]
        top_boundary = int(h * 0.25)
        img = img[top_boundary:h, 0:w]
        
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_LINEAR)
        
        if self.grayscale:
            img_processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_processed = img_processed[np.newaxis, :, :]
        else:
            img_processed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_processed = img_processed.transpose(2, 0, 1)
            
        return img_processed.astype(np.float32)

    def get_action(self, obs_tensor):
        """
        Handles deterministic inference for the physical robot.
        """
        processed_frame = self.preprocess(obs_tensor)

        if len(self.frames) == 0:
            for _ in range(self.frame_stack):
                self.frames.append(processed_frame)
        else:
            self.frames.append(processed_frame)
            
        # Stack along channel dimension: (C*Stack, 84, 84)
        stacked_input = np.concatenate(list(self.frames), axis=0)
        input_tensor = torch.FloatTensor(stacked_input).unsqueeze(0).to(self.device)

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

    
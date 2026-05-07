import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image
import numpy as np 
import cv2
import collections

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
        processed_action = action

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
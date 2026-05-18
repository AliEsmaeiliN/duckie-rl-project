import gymnasium as gym
import numpy as np

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
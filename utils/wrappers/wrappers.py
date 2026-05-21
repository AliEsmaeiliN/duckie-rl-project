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
        self.reset_threshold = 0.07   # Must get back near centerline to clear mode
        self.in_recovery_mode = False

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        sim_code = info.get("Simulator", {}).get("done_code")
        is_sim_ood = done and (sim_code == "invalid-pose")
        
        #is_ood = done and (reward <= -1000 or info.get("Simulator", {}).get("done_code") == "invalid-pose")
        if is_sim_ood:
            self.in_recovery_mode  = True

        if self.in_recovery_mode:
            print(f"recovery step: {self.recovery_steps}")
            self.recovery_steps += 1
            
            reward = self.ood_penalty
            
            if self.recovery_steps < self.max_recovery_steps:
                done = False
                
                try:
                    sim = self.unwrapped
                    lp = sim.get_lane_pos2(sim.cur_pos, sim.cur_angle)
                    if abs(lp.dist) < self.reset_threshold: 
                        print("Recovered")
                        self.recovery_steps = 0
                        self.in_recovery_mode = False
                except Exception:
                    pass 
            else:
                # Agent failed to recover in time
                done = True
                self.in_recovery_mode = False
        else:
            self.recovery_steps = 0
            
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        self.recovery_steps = 0
        self.in_recovery_mode = False
        return self.env.reset(**kwargs)
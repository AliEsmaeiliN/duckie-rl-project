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
        if is_sim_ood and self.max_recovery_steps > 0:
            self.in_recovery_mode  = True

        if self.in_recovery_mode:
            #print(f"recovery step: {self.recovery_steps}")
            self.recovery_steps += 1
            
            reward = self.ood_penalty
            
            if self.recovery_steps < self.max_recovery_steps:
                done = False
                
                try:
                    sim = self.unwrapped
                    lp = sim.get_lane_pos2(sim.cur_pos, sim.cur_angle)
                    if abs(lp.dist) < self.reset_threshold: 
                        #print("Recovered")
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
    
class TileTrackingWrapper(gym.Wrapper):
    """
    Tracks the number of tiles traversed and loops completed during an episode.
    Injects these metrics into the `info` dictionary for evaluation logging.
    """
    def __init__(self, env):
        super().__init__(env)
        self.current_tile = None
        self.start_tile = None
        self.tiles_passed = 0
        self.laps_completed = 0
        self.visited_tiles = set()
        self.track_lenght = 10

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        sim = self.unwrapped
        self.current_tile = sim.get_grid_coords(sim.cur_pos)
        self.start_tile = self.current_tile
        
        self.tiles_passed = 0
        self.laps_completed = 0
        self.visited_tiles = {self.current_tile}
        
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        sim = self.unwrapped
        
        new_tile = sim.get_grid_coords(sim.cur_pos)
        
        if new_tile != self.current_tile:
            self.current_tile = new_tile
            self.tiles_passed += 1
            self.visited_tiles.add(new_tile)
            
            if new_tile == self.start_tile and len(self.visited_tiles) > 7:
                self.laps_completed += 1
                self.visited_tiles = {self.start_tile}
        
        info['tiles_passed'] = self.tiles_passed
        info['laps_completed'] = self.laps_completed
        
        return obs, reward, done, truncated, info
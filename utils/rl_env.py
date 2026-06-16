import os
import gymnasium as gym
import numpy as np
from gym_duckietown.simulator import Simulator
from utils.wrappers.wrappers import *
from utils.wrappers.observation_wrappers import *
from utils.wrappers.action_wrappers import *
from utils.wrappers.reward_wrappers import *
from utils.wrappers.reward_wrappers import AdditiveJerkPenalty


class DuckieOvalEnv(Simulator):
    """
    A specialized Duckietown environment for Oval navigation.
    """
    def __init__(self, **kwargs):
        kwargs.setdefault('map_name', "oval_loop")
        kwargs.setdefault('camera_width', 640)
        kwargs.setdefault('camera_height', 480)
        kwargs.setdefault('accept_start_angle_deg', 20)
        kwargs.setdefault('full_transparency', True)
        kwargs.setdefault('max_steps', 3000)
        kwargs.setdefault('frame_skip', 3)
        kwargs.setdefault('spawn_mode', 'curriculum')
        kwargs.setdefault('spawn_difficulty', 0.0)
        
        super().__init__(**kwargs)
        
        self.wheel_dist = 0.102 
        self.robot_radius = 0.0318
        self.motor_k = 27.0

    @classmethod
    def create_wrapped(cls, run_name, reward_type="unified_v1", capture_video=False, 
                        ema=False, motion_blur=False, grayscale=True, 
                        frame_stack=4, latency_rand=False, recovery_step=False,
                        jerk_penalty=False, preprocessing=False, is_eval=True, **kwargs
                    ):
        """
        Static method to build the fully wrapped stack.
        """
        env = cls(**kwargs)

        if is_eval:
            env = TileTrackingWrapper(env)

        env = KinematicActionWrapper(env, wheel_dist=0.102, radius=0.0318, k=27.0)


        if ema:
            env = ActionSmoothingWrapper(env)

        if latency_rand:
            env = ActionLatencyWrapper(env)

        #env = DirectionLockWrapper(env)

        if capture_video:
            video_folder = f"videos/{run_name}"
            os.makedirs(video_folder, exist_ok=True)
            env = gym.wrappers.RecordVideo(env, video_folder, episode_trigger=lambda x: True)
        
        if motion_blur:
            env = FastKinematicBlurWrapper(env)

        if preprocessing:
            env = CLAHEWrapper(env)
            env = GaussianBlurWrapper(env)
        
        env = CropResizeWrapper(env, shape=(84, 84))

        
        if grayscale:
            env = GrayscaleWrapper(env)
            if preprocessing:
                env = ContrastStretchingWrapper(env)
        else:
            env = ImgWrapper(env) # Transpose to CHW

        reward_registry = {
            "unified": UnifiedReward,
            "adaptive": AdaptiveRewardWrapper,
            "unified_v2": UnifiedRewardv2,
            "unified_v1": UnifiedRewardv1,
        }

        if reward_type not in reward_registry:
            raise ValueError(f"Unknown reward_type requested: {reward_type}. Choose from {list(reward_registry.keys())}")
        else:
            print(f"Using {reward_registry[reward_type].__name__} For the Reward Function")
            
        env = reward_registry[reward_type](env)

        if jerk_penalty:
            env = AdditiveJerkPenalty(env)
        
        if recovery_step:
            env = RecoveryTrainingWrapper(env, max_recovery_steps=20, ood_penalty=-10.0)

        if frame_stack > 1:
            env = gym.wrappers.FrameStackObservation(env, stack_size=frame_stack)
            c = 1 if grayscale else 3
            final_channels = c * frame_stack
            new_obs_space = gym.spaces.Box(
                low=0, 
                high=255, 
                shape=(final_channels , 84, 84), 
                dtype=np.uint8
            )   
            env = gym.wrappers.TransformObservation(
                env, 
                lambda obs: np.array(obs).reshape(final_channels, 84, 84),
                observation_space=new_obs_space
            )

        return gym.wrappers.RecordEpisodeStatistics(env)

    def set_curriculum(self, max_recovery_steps=None, **rand_kwargs):
        """
        Unified method called during training milestones to update both
        the recovery wrapper settings and internal simulation randomizations.
        """
        sim = self.unwrapped
        for key, value in rand_kwargs.items():
            if hasattr(sim, key):
                setattr(sim, key, value)
                print(f"Simulator config updated: {key} = {value}")
            else:
                print(f"Warning: Simulator has no attribute '{key}'")

        if max_recovery_steps is not None:
            curr_env = self
            found = False
            while hasattr(curr_env, 'env'):
                if isinstance(curr_env, RecoveryTrainingWrapper):
                    curr_env.max_recovery_steps = max_recovery_steps
                    print(f"[{self.map_name}] Recovery Wrapper updated: max_recovery_steps = {max_recovery_steps}")
                    found = True
                    break
                curr_env = curr_env.env
            
            if not found:
                print("Warning: RecoveryTrainingWrapper was not found in the environment stack!")
    
    def set_spawn_config(self, mode: str = None, difficulty: float = None):
        """Dynamically update spawn strategy during training."""
        if mode is not None:
            self.spawn_mode = mode
        if difficulty is not None:
            self.spawn_difficulty = np.clip(difficulty, 0.0, 1.0)
        print(f"Spawn Config Updated: Mode={self.spawn_mode}, Difficulty={self.spawn_difficulty}")

def update_curriculum_stage(envs, global_step, args):
    """
    Unified switchboard function managed within the environment module 
    to handle spatial margins, FSM recovery steps, and physical randomizations.
    """
    total_timesteps = args.total_timesteps

    if global_step == 0 and args.recovery:
        envs.call("set_curriculum", max_recovery_steps=20)        

    elif global_step == 300000:
        print(f"\n[Curriculum] Step {global_step}: Visual Domain Randomization ON.")
        envs.call("set_curriculum", domain_rand=args.domain_rand)

    elif global_step == 450000:
        print(f"\n[Curriculum] Step {global_step}: Activating Camera & Dynamics Randomization.")
        envs.call("set_curriculum", camera_rand=args.camera_rand, dynamics_rand=args.dynamics_rand)

    elif global_step == 600000 and args.recovery: 
        print(f"\n[Curriculum] Step {global_step}: Tightening recovery window to 10 steps.")
        envs.call("set_curriculum", max_recovery_steps=10)

    elif global_step == int(0.7 * total_timesteps):
        if args.recovery:
            print(f"\n[Curriculum] Step {global_step}: Safety Horizon closed (0 steps). Policy running under absolute constraints.")
            envs.call("set_curriculum", max_recovery_steps=0)
        

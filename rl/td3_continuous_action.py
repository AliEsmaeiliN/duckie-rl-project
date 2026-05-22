# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import random
import time
from dataclasses import dataclass
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer

# CNN Architucture 
from rl.cnn_architectures import ImpalaCNN as cnn_encoder

# Utilities
from utils.rl_env import DuckieOvalEnv
from utils.debug_tools import save_models, DuckiebotEvaluator

# Target the specific logger used in the simulator
import logging
duckietown_logger = logging.getLogger("gym-duckietown")
duckietown_logger.setLevel(logging.WARNING)

# Disable error checking for maximum training throughput
import pyglet
pyglet.options['debug_gl'] = False


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "Duckie-RL-V2"
    """the wandb's project name"""
    wandb_group: str = "TD3"
    """The algorithm"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    eval_interval: int = 20000
    """the interval to evaluate the Actor periodically"""
    run_notes: str = ""
    """for wandb tracking notes"""
    save_interval: int = 50000
    """the interval to save the Actor periodically"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    grayscale: bool = True
    """whether to convert the observation to grayscale"""
    version: int = 0
    """the version of the model, default zero is for the test"""
    start_evaluation: int = 700000
    """timestamp to start the internal evaluation (usually after finalizing the randomizations)"""


    # Algorithm specific arguments
    env_id: str = "Oval_td3"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(5e4)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""

    #Duckietown specific arguments
    domain_rand: bool = False
    """texture/light randomization"""
    distortion: bool = False 
    """Simulates the fisheye lens"""
    dynamics_rand: bool = False
    """Simulates motor/trim imbalances"""
    camera_rand: bool = False 
    """Simulates mounting misalignments"""
    motion_blur: bool = False
    """Simulates the blur from the moving duckiebot"""
    action_latency: bool = False
    """Simulates the action latency from the duckiebot"""
    ema: bool = False
    """Use EMA action smoothing"""
    direction: str = "mixed"
    """Choosing the direction of the loop. CW, CCW or mixed"""
    curriculum_randomization: bool = True
    """Acivating the randomizations gradually based on curriculum learning"""
    recovery: bool = False
    """Gives the robot 20 steps to recover"""
    jerk_penalty: bool = False
    """Adding the jerk penalty to the final reward"""

def make_env(seed, idx, run_name, capture_video=False, action_smoothing=False, motion_blur=False, latency_rand=False, jerk_penalty=False, **env_kwargs):
    def thunk():
        render_mode = "rgb_array" if (capture_video and idx == 0) else None
        env = DuckieOvalEnv.create_wrapped(
            run_name=run_name,
            ema=action_smoothing,
            latency_rand=latency_rand,
            render_mode=render_mode,
            motion_blur=motion_blur,
            seed=seed,
            jerk_penalty=jerk_penalty,
            direction=args.direction,
            **env_kwargs
        )
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, feature_dim=256):
        super().__init__()
        in_channels = 4 if args.grayscale else 12

        self.encoder = cnn_encoder(
            in_channels=in_channels,
            obs_shape=env.single_observation_space.shape,
            feature_dim=feature_dim
        )
        
        action_dim = np.prod(env.single_action_space.shape)

        self.fc1 = nn.Linear(feature_dim + action_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        visual_features = self.encoder(x)
        combined = torch.cat([visual_features, a], 1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env, feature_dim=256 , grayscale=True):
        super().__init__()
        in_channels = 4 if grayscale else 12

        self.encoder = cnn_encoder(
            in_channels=in_channels,
            obs_shape=env.single_observation_space.shape,
            feature_dim=feature_dim
        )

        #self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        #self.fc2 = nn.Linear(256, 256)
        
        self.fc_mu = nn.Linear(feature_dim, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        visual_features = self.encoder(x)
        mu = self.fc_mu(visual_features)

        v = torch.sigmoid(mu[:, 0:1]) 
        omega = torch.tanh(mu[:, 1:2])

        x = torch.cat([v, omega], dim=-1)
        return x * self.action_scale + self.action_bias


if __name__ == "__main__":

    args = tyro.cli(Args)
    input_mode = "" if args.grayscale else "_RGB"
    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_name = f"td3__{args.env_id}{input_mode}__v{args.version}__{args.seed}__{timestamp}"
    if args.track:
        import wandb
        active_tags = [args.env_id]
        active_tags.append("Grayscale" if args.grayscale else "RGB")
        active_tags.append(args.direction)
        if args.domain_rand: active_tags.append("DomainRand")
        if args.dynamics_rand: active_tags.append("DynamicsRand")
        if args.camera_rand: active_tags.append("CameraRand")
        if args.curriculum_randomization: active_tags.append("Crcm Rand")
        if args.ema: active_tags.append("EMA")
        if args.action_latency: active_tags.append("ActionLatency")
        if args.recovery: active_tags.append("Recovery")
        if args.jerk_penalty: active_tags.append("JerkPenalty")





        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            group=args.wandb_group,
            tags=active_tags,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )
        reward_logic = wandb.Artifact('rl-logic-files', type='code')
        reward_logic.add_file('utils/wrappers/reward_wrappers.py') 
        reward_logic.add_file('utils/rl_env.py')
        reward_logic.add_file('rl/td3_continuous_action.py')
        
        run.log_artifact(reward_logic)

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_params = {
        "domain_rand": args.domain_rand,
        "dynamics_rand": args.dynamics_rand,
        "camera_rand": args.camera_rand,
    }
    
    base_cfg = {key: False for key in env_params}
    robust_cfg = {key: True for key in env_params}
        
    active_env_params = {} if args.curriculum_randomization else env_params.copy()

    # LIGHTWEIGHT UNIFIED EVALUATION ENVIRONMENTS
    best_eval_reward = -float('inf')
    eval_env_seed = args.seed + 100

    eval_env_imperfect = make_env(seed=eval_env_seed, idx=0, run_name=f"{run_name}_eval", action_smoothing=True, motion_blur=True, latency_rand=True, jerk_penalty=args.jerk_penalty, **robust_cfg)()
    eval_env_perfect = make_env(seed=eval_env_seed, idx=0, run_name=f"{run_name}_eval2", jerk_penalty=args.jerk_penalty, **base_cfg)() 
    eval_env_perfect.unwrapped.set_randomization(margin_factor=0.1)
    eval_env_imperfect.unwrapped.set_randomization(margin_factor=0.1)
    eval_env_imperfect.unwrapped.set_spawn_config(mode="curriculum", difficulty=1)
    
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.seed + i, i, run_name, recovery_step=args.recovery, action_smoothing=args.ema, motion_blur=args.motion_blur, latency_rand=args.action_latency, jerk_penalty=args.jerk_penalty, **active_env_params) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf2 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    qf2_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    evaluator1 = DuckiebotEvaluator(eval_env_perfect, eval_env_seed, actor, args, device, prefix="eval_perfect")
    evaluator2 = DuckiebotEvaluator(eval_env_imperfect, eval_env_seed, actor, args, device, prefix="eval_imperfect")
    
    envs.single_observation_space.dtype = np.uint8
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps + 1):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            raw_actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            raw_actions[:, 0] = np.abs(raw_actions[:, 0])
            actions = raw_actions
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Curriculum spawn
        if any(terminations) or any(truncations):
            new_difficulty = min(1.0, global_step / (0.6 * args.total_timesteps))
            # This sets the attribute for ALL parallel sub-environments
            envs.set_attr("spawn_difficulty", new_difficulty)
            writer.add_scalar("charts/spawn_difficulty", new_difficulty, global_step)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "episode" in infos:
            for i in range(envs.num_envs):
                # Using the mask '_episode' to see which sub-env actually finished
                if "_episode" in infos and infos["_episode"][i]:
                    print(f"global_step={global_step}, episodic_return={infos['episode']['r'][i]}")
                    writer.add_scalar("charts/episodic_return", infos['episode']['r'][i], global_step)
                    writer.add_scalar("charts/episodic_length", infos['episode']['l'][i], global_step)  

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        if "final_observation" in infos:
            for idx, has_final_obs in enumerate(infos.get("_final_observation", [])):
                if has_final_obs:
                    real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                clipped_noise = (torch.randn_like(data.actions, device=device) * args.policy_noise).clamp(
                    -args.noise_clip, args.noise_clip
                ) * target_actor.action_scale

                next_state_actions = (target_actor(data.next_observations) + clipped_noise).clamp(
                    envs.single_action_space.low[0], envs.single_action_space.high[0]
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                    
            if args.curriculum_randomization:
                if global_step == 3e5:
                    print("Curriculum Step 1: Activating Domain Randomization")
                    envs.call("set_randomization", domain_rand=args.domain_rand)
                elif global_step == 4.5e5:
                    print("Curriculum Step 2: Activating Camera Randomization")
                    envs.call("set_randomization", camera_rand=args.camera_rand)
                elif global_step == 6e5:
                    print("Curriculum Step 2: Activating Dynamics Randomization")
                    envs.call("set_randomization", dynamics_rand=args.dynamics_rand)
            
            if global_step % args.eval_interval == 0 and global_step >= args.start_evaluation:
                score1, _, _, is_best = evaluator1.evaluate(
                    is_interval=True,
                    global_step=global_step,
                    best_reward=best_eval_reward,
                    num_episodes=10
                )
                score2, _, _, _ = evaluator2.evaluate(
                    is_interval=True,
                    global_step=global_step,
                    best_reward=best_eval_reward,
                    num_episodes=10
                )
                
                writer.add_scalar("charts/risk_adjusted_score_perfect", score1, global_step)
                writer.add_scalar("charts/risk_adjusted_score_imperfect", score2, global_step)

                if is_best:
                    best_eval_reward = score1
                    print(f" New Peak Performance Milestone! Saving weights...")
                    save_models(
                        actor=actor, qf1=qf1, qf2=qf2, 
                        step=global_step, run_name=run_name, 
                        args=args, env_params=env_params, 
                        suffix=f"v{args.version}_BEST"
                    )


    if args.save_model:
        save_models(actor, qf1, qf2, global_step, run_name, args, env_params, suffix=f"v{args.version}_Final")

    eval_env_perfect.close()
    eval_env_imperfect.close()
    envs.close()
    writer.close()

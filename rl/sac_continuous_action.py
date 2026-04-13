# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import random
import time
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
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
from utils.debug_tools import save_models, evaluate_policy

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
    wandb_project_name: str = "Duckie-RL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "SAC"
    """The algorithm"""
    run_notes: str = ""
    """for wandb tracking notes"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_interval: int = 100000
    """the interval to save the Actor periodically"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    grayscale: bool = True
    """whether to convert the observation to grayscale"""
    eval_model: bool = True
    """whether to evaluate the saved model at the end of training"""

    # Algorithm specific arguments
    env_id: str = "Oval-"
    """the environment id of the task"""
    total_timesteps: int = 1000001
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(5e4) # image input: we can not have too many 1e6 ... Currently the best performing speed wise is 5e4 (1 env , SyncVectorEnv)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256    #256 before ... Currently the best performing speed wise is 256 (1 env , SyncVectorEnv)
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4 #1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

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

def make_env(seed, idx, run_name, capture_video=False, motion_blur=False, **env_kwargs):
    def thunk():
        render_mode = "rgb_array" if (capture_video and idx == 0) else None
        env = DuckieOvalEnv.create_wrapped(
            run_name=run_name,
            motion_blur=motion_blur,
            render_mode=render_mode,
            seed=seed, 
            **env_kwargs
        )
        env.action_space.seed(seed)
        return env

    return thunk

# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, feature_dim=256):
        super().__init__()

        self.channels = 4 if args.grayscale else 12
        # Independent Visual Encoder
        self.encoder = cnn_encoder(
            in_channels=self.channels,
            obs_shape=env.single_observation_space.shape,
            feature_dim=feature_dim
        )

        # The input size is feature_dim (visuals) + action_dim (robot commands)
        action_dim = np.prod(env.single_action_space.shape)

        self.fc1 = nn.Linear(feature_dim + action_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_q = nn.Linear(256, 1)

    def forward(self, x, a):
        # x: Image observations (Batch, 12, 120, 160)
        # a: Actions (Batch, 2)
        
        # Extract features from the images
        visual_features = self.encoder(x)
        
        # Concatenate visual features with the action vector
        # [Batch, 256] + [Batch, 2] -> [Batch, 258]
        combined_input = torch.cat([visual_features, a], dim=1)
        
        # Standard MLP layers to estimate Q-value
        x = F.relu(self.fc1(combined_input))
        x = F.relu(self.fc2(x))
        q_value = self.fc_q(x)
        return q_value


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, grayscale=True):
        super().__init__()

        self.channels = 4 if grayscale else 12
        # Modified Encoder
        self.encoder = cnn_encoder(
            in_channels=self.channels,
            obs_shape=env.single_observation_space.shape,
            feature_dim=256
        )
        


        #self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        #self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
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
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.encoder(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        v_t = torch.sigmoid(x_t[:,0:1])
        omega_t = torch.tanh(x_t[:,1:2])
        y_t = torch.cat([v_t, omega_t], dim=-1)

        #y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        # --- LOG PROB CORRECTION (Jacobian) ---
        log_prob = normal.log_prob(x_t)

        # Sigmoid correction: log(d/dx sigmoid) = log(sigmoid * (1 - sigmoid))
        log_prob[:, 0:1] -= torch.log(v_t * (1.0 - v_t) + 1e-6)
        
        # Tanh correction: log(d/dx tanh) = log(1 - tanh^2)
        log_prob[:, 1:2] -= torch.log(1.0 - omega_t.pow(2) + 1e-6)
        
        log_prob = log_prob.sum(1, keepdim=True)
        
        # Mean for evaluation (Deterministic mode)
        mean_v = torch.sigmoid(mean[:, 0:1])
        mean_omega = torch.tanh(mean[:, 1:2])
        mean_action = torch.cat([mean_v, mean_omega], dim=-1) * self.action_scale + self.action_bias

        # Enforcing Action Bound
        #log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        #log_prob = log_prob.sum(1, keepdim=True)
        #mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


if __name__ == "__main__":

    args = tyro.cli(Args)
    input_mode = "" if args.grayscale else "_RGB"
    run_name = f"sac__{args.env_id}{input_mode}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        active_tags = [args.env_id]
        active_tags.append("Grayscale" if args.grayscale else "RGB")
        if args.domain_rand: active_tags.append("DomainRand")
        if args.dynamics_rand: active_tags.append("DynamicsRand")
        if args.camera_rand: active_tags.append("CameraRand")
        if args.distortion: active_tags.append("Distortion")
        if args.motion_blur: active_tags.append("MotionBlur")

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
        reward_logic.add_file('utils/wrappers.py') 
        reward_logic.add_file('utils/env_lunch.py')
        try:
            reward_logic.add_file('job_sac.sh')
        except (ValueError, FileNotFoundError) as e:
            print(f"Warning: Could not find job file for artifact logging: {e}")
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
        "distortion": args.distortion,
        "dynamics_rand": args.dynamics_rand,
        "camera_rand": args.camera_rand,
    }

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.seed + i, i, run_name, args.capture_video, args.motion_blur) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs, grayscale=args.grayscale).to(device)
    qf1 = SoftQNetwork(envs, feature_dim=256).to(device)
    qf2 = SoftQNetwork(envs, feature_dim=256).to(device)
    qf1_target = SoftQNetwork(envs, feature_dim=256).to(device)
    qf2_target = SoftQNetwork(envs, feature_dim=256).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    #envs.single_observation_space.dtype = np.float32  previous version
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
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            obs_tensor = torch.Tensor(obs).to(device)
            actions, _, _ = actor.get_action(obs_tensor)
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

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
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
        
            s_obs = data.observations.to(device, non_blocking=True)
            s_next_obs = data.next_observations.to(device, non_blocking=True)

            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(s_next_obs)
                qf1_next_target = qf1_target(s_next_obs, next_state_actions)
                qf2_next_target = qf2_target(s_next_obs, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(s_obs, data.actions).view(-1)
            qf2_a_values = qf2(s_obs, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(s_obs)
                    qf1_pi = qf1(s_obs, pi)
                    qf2_pi = qf2(s_obs, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(s_obs)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
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
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)


    if args.save_model:
        save_models(actor, qf1, qf2, global_step, run_name, args, env_params, suffix="Final")
    if args.eval_model:
        evaluate_policy(
            actor=actor,
            args=args,
            device=device,
            algo_name="SAC",
            num_episodes=10,
            run_name=run_name,
            **env_params
        )
    
    envs.close()
    writer.close()

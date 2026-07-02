import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import random
import time
from typing import Optional
from dataclasses import dataclass
from datetime import datetime

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
from utils.rl_env import DuckieOvalEnv, update_curriculum_stage
from utils.debug_tools import save_models, DuckiebotEvaluator
from utils.shared_args import SharedDuckieArgs

# Target the specific logger used in the simulator
import logging
duckietown_logger = logging.getLogger("gym-duckietown")
duckietown_logger.setLevel(logging.WARNING)

# Disable error checking for maximum training throughput
import pyglet
pyglet.options['debug_gl'] = False


@dataclass
class Args(SharedDuckieArgs):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    wandb_group: str = "SAC"
    """The algorithm"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4 #1e-3
    """the learning rate of the Q network network optimizer"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""


def make_env(seed, idx, run_name, capture_video=False, action_smoothing=False, motion_blur=False, direction_lock=False,
             latency_rand=False, jerk_penalty=False, recovery_step=False, preprocessing=False, is_eval=False, **env_kwargs):
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
            recovery_step=recovery_step,
            preprocessing=preprocessing,
            is_eval=is_eval,
            direction_lock=direction_lock,
            
            direction=args.direction,
            reward_type=args.reward_type,
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
    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_name = f"sac__{args.env_id}{input_mode}__{args.reward_type}__{args.seed}__{timestamp}"
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
        reward_logic.add_file('rl/sac_continuous_action.py')
        
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
        "camera_rand": args.camera_rand
    }

    base_cfg = {key: False for key in env_params}
    robust_cfg = {key: True for key in env_params}

    eval_preprocess = args.preprocessing if args.preprocessing_eval is None else args.preprocessing_eval
    eval_ema = args.ema if args.eval_ema is None else args.eval_ema

    active_env_params = {} if args.curriculum_randomization else env_params.copy()

    # LIGHTWEIGHT UNIFIED EVALUATION ENVIRONMENTS
    best_eval_reward = -float('inf')
    eval_env_seed = args.seed + 100

    eval_env_imperfect = make_env(
        seed=eval_env_seed, 
        idx=0, 
        run_name=f"{run_name}_eval", 
        action_smoothing=eval_ema, 
        motion_blur=True, 
        latency_rand=True, 
        preprocessing=eval_preprocess,
        is_eval=True,
        direction_lock=True,
        **robust_cfg
    )()
    
    eval_env_perfect = make_env(
        seed=eval_env_seed, 
        idx=0, 
        run_name=f"{run_name}_eval2", 
        preprocessing=eval_preprocess,
        is_eval=True,
        direction_lock=True,
        **base_cfg
    )()

    eval_env_perfect.unwrapped.set_curriculum(margin_factor=0.1)
    eval_env_imperfect.unwrapped.set_curriculum(margin_factor=0.1)
    eval_env_imperfect.unwrapped.set_spawn_config(mode="curriculum", difficulty=1)
    
    envs = gym.vector.SyncVectorEnv([
        make_env(
            args.seed + i, i, run_name,
            recovery_step=args.recovery, 
            action_smoothing=args.ema, 
            motion_blur=args.motion_blur, 
            latency_rand=args.action_latency, 
            jerk_penalty=args.jerk_penalty, 
            preprocessing=args.preprocessing, 
            **active_env_params
        ) for i in range(args.num_envs)
    ])
   
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

    evaluator_p = DuckiebotEvaluator(eval_env_perfect, eval_env_seed, actor, args, device, prefix="eval_perfect")
    evaluator2_imp = DuckiebotEvaluator(eval_env_imperfect, eval_env_seed, actor, args, device, prefix="eval_imperfect")

    randomization_unlocked = False


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
    for global_step in range(args.total_timesteps + 1):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            raw_actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            raw_actions[:, 0] = np.abs(raw_actions[:, 0])
            actions = raw_actions
        else:
            obs_tensor = torch.Tensor(obs).to(device)
            actions, _, _ = actor.get_action(obs_tensor)
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Curriculum spawn
        if any(terminations) or any(truncations):
            new_difficulty = min(1.0, global_step / (0.6 * args.total_timesteps))
            envs.set_attr("spawn_difficulty", new_difficulty)
            writer.add_scalar("charts/spawn_difficulty", new_difficulty, global_step)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "episode" in infos:
            for i in range(envs.num_envs):
                # Using the mask '_episode' to see which sub-env actually finished
                if "_episode" in infos and infos["_episode"][i]:
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
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
            if args.curriculum_randomization:
                update_curriculum_stage(envs=envs, global_step=global_step, args=args)
            
            if global_step % args.eval_interval == 0 and global_step >= args.start_evaluation:
                score1, _, _, is_best_p, _ = evaluator_p.evaluate(
                    is_interval=True,
                    global_step=global_step,
                    best_reward=best_eval_reward,
                    num_episodes=10
                )
                score2, _, _, is_best_imp, success_rate = evaluator2_imp.evaluate(
                    is_interval=True,
                    global_step=global_step,
                    best_reward=best_eval_reward,
                    num_episodes=10
                )
                if not randomization_unlocked and score1 >= 450.0 and args.curriculum_randomization: 
                    print(f"\n[Performance Gate] Peak Mastery Detected (Score: {score1:.2f}/500)!")
                    print("Unlocking all Visual, Camera, and Dynamics Randomizations simultaneously.")
                    
                    envs.call("set_curriculum", 
                            domain_rand=args.domain_rand, 
                            camera_rand=args.camera_rand, 
                            dynamics_rand=args.dynamics_rand)
                    
                    randomization_unlocked = True
                
                writer.add_scalar("charts/risk_adjusted_score_perfect", score1, global_step)
                writer.add_scalar("charts/risk_adjusted_score_imperfect", score2, global_step)

                if is_best_imp and (success_rate >= 80.0):
                    best_eval_reward = score2
                    print(f" New Peak Performance Milestone! Saving weights...")
                    save_models(
                        actor=actor, qf1=qf1, qf2=qf2, 
                        step=global_step, run_name=run_name, 
                        args=args, env_params=env_params, 
                        suffix=f"vr{args.reward_type}s{args.seed}_BEST"
                    )


    if args.save_model:
        save_models(actor, qf1, qf2, global_step, run_name, args, env_params, suffix=f"vr{args.reward_type}s{args.seed}_Final")
    
    eval_env_perfect.close()
    eval_env_imperfect.close()
    envs.close()
    writer.close()

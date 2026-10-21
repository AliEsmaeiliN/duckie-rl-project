# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import random
import time
from typing import Optional
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

from cleanrl_utils.frame_buffer import FrameStackReplayBuffer

# CNN Architucture 
from rl.cnn_architectures import ImpalaCNN as cnn_encoder

# Utilities
from utils.rl_env import DuckieOvalEnv, update_curriculum_stage, make_vector_env
from utils.debug_tools import save_models, DuckiebotEvaluator
from utils.shared_args import SharedDuckieArgs, pick_device

# Target the specific logger used in the simulator
import logging
duckietown_logger = logging.getLogger("gym-duckietown")
duckietown_logger.setLevel(logging.WARNING)

# pyglet options (headless, GL error checking off) are set in gym_duckietown/__init__.py:
# they only take effect before pyglet.gl is imported, which happens on the import above

@dataclass
class Args(SharedDuckieArgs):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    wandb_group: str = "TD3"
    """The algorithm"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""

def make_env(seed, idx, run_name, capture_video=False, action_smoothing=False, motion_blur=False,
             latency_rand=False, jerk_penalty=False, recovery_step=False, preprocessing=False, is_eval=False, **env_kwargs):
    # Read here, not in thunk: the thunk is sent to the env worker processes
    direction, reward_type = args.direction, args.reward_type

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
            
            direction=direction,
            reward_type=reward_type,
            **env_kwargs
        )
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QHead(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim + action_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, features, a):
        x = torch.cat([features, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class QNetwork(nn.Module):
    """
    Twin Q-networks sharing one visual encoder (DrQ-v2 style). The encoder is trained by
    the critic loss only; the actor reads its features detached.
    """
    def __init__(self, env, feature_dim=256, grayscale=True):
        super().__init__()
        in_channels = 4 if grayscale else 12
        self.encoder = cnn_encoder(
            in_channels=in_channels,
            obs_shape=env.single_observation_space.shape,
            feature_dim=feature_dim
        )
        action_dim = np.prod(env.single_action_space.shape)
        self.q1 = QHead(feature_dim, action_dim)
        self.q2 = QHead(feature_dim, action_dim)

    def q_from_features(self, features, a):
        return self.q1(features, a), self.q2(features, a)

    def forward(self, x, a):
        return self.q_from_features(self.encoder(x), a)


class Actor(nn.Module):
    """
    Policy head on the critic's encoder. `encoder` is the critic's module, so the saved
    actor state dict has the same layout as before (encoder + fc_mu) and deployment code
    loads it unchanged. Only head_parameters() go to the actor optimizer.
    """
    def __init__(self, env, encoder=None, feature_dim=256, grayscale=True):
        super().__init__()
        # Without a critic encoder (e.g. loading a checkpoint for inspection) build a private one
        self.encoder = encoder if encoder is not None else cnn_encoder(
            in_channels=4 if grayscale else 12,
            obs_shape=env.single_observation_space.shape,
            feature_dim=feature_dim
        )

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

    def head_parameters(self):
        return list(self.fc_mu.parameters())

    def forward(self, x):
        return self.action_from_features(self.encoder(x))

    def action_from_features(self, features):
        mu = self.fc_mu(features)

        v = torch.sigmoid(mu[:, 0:1]) 
        omega = torch.tanh(mu[:, 1:2])

        x = torch.cat([v, omega], dim=-1)
        return x * self.action_scale + self.action_bias


if __name__ == "__main__":

    args = tyro.cli(Args)
    input_mode = "" if args.grayscale else "_RGB"
    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_name = f"td3__{args.env_id}{input_mode}__{args.reward_type}__{args.seed}__{timestamp}"
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

    device = pick_device(args.cuda)
    print(f"Using device: {device}")
    if args.track:
        import platform
        wandb.config.update({"device": str(device), "host": platform.node()})

    # env setup
    env_params = {
        "domain_rand": args.domain_rand,
        "dynamics_rand": args.dynamics_rand,
        "camera_rand": args.camera_rand,
    }
    
    base_cfg = {key: False for key in env_params}
    robust_cfg = {key: True for key in env_params}
        
    active_env_params = {} if args.curriculum_randomization else env_params.copy()

    eval_preprocess = args.preprocessing if args.preprocessing_eval is None else args.preprocessing_eval
    eval_ema = args.ema if args.eval_ema is None else args.eval_ema

    # Training envs first: async workers are spawned before this process opens a GL context
    envs = make_vector_env([
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
    ], async_envs=args.async_envs)

    # LIGHTWEIGHT UNIFIED EVALUATION ENVIRONMENTS
    best_eval_reward = -float('inf')
    eval_env_seed = args.seed + 100

    eval_env_imperfect = make_env(
        seed=eval_env_seed, 
        idx=0, 
        run_name=f"{run_name}_eval_im", 
        action_smoothing=eval_ema, 
        motion_blur=True, 
        latency_rand=True, 
        preprocessing=eval_preprocess,
        is_eval=True,
        **robust_cfg
    )()
    
    eval_env_perfect = make_env(
        seed=eval_env_seed, 
        idx=0, 
        run_name=f"{run_name}_eval_p", 
        preprocessing=eval_preprocess,
        is_eval=True,
        **base_cfg
    )()

    eval_env_perfect.unwrapped.set_curriculum(margin_factor=0.1)
    eval_env_imperfect.unwrapped.set_curriculum(margin_factor=0.1)
    eval_env_imperfect.unwrapped.set_spawn_config(mode="curriculum", difficulty=1)

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    qf = QNetwork(envs, grayscale=args.grayscale).to(device)
    qf_target = QNetwork(envs, grayscale=args.grayscale).to(device)
    qf_target.load_state_dict(qf.state_dict())
    actor = Actor(envs, encoder=qf.encoder).to(device)
    # The target actor's head reads the target critic's encoder
    target_actor = Actor(envs, encoder=qf_target.encoder).to(device)
    target_actor.fc_mu.load_state_dict(actor.fc_mu.state_dict())
    q_optimizer = optim.Adam(list(qf.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(actor.head_parameters(), lr=args.learning_rate)

    evaluator_p = DuckiebotEvaluator(eval_env_perfect, eval_env_seed, actor, args, device, prefix="eval_perfect")
    evaluator_imp = DuckiebotEvaluator(eval_env_imperfect, eval_env_seed, actor, args, device, prefix="eval_imperfect")
    randomization_unlocked = False
    
    buffer_device = device if args.buffer_device == "auto" else torch.device(args.buffer_device)
    rb = FrameStackReplayBuffer(
        args.buffer_size,
        envs.single_observation_space.shape,
        int(np.prod(envs.single_action_space.shape)),
        frame_channels=1 if args.grayscale else 3,
        device=buffer_device,
    )
    online_params = actor.head_parameters() + list(qf.parameters())
    target_params = target_actor.head_parameters() + list(qf_target.parameters())
    action_low, action_high = float(envs.single_action_space.low[0]), float(envs.single_action_space.high[0])

    def update(n_updates):
        """One TD3 gradient step. Returns the metrics as tensors, read only when logged."""
        data = rb.sample(args.batch_size)
        s_obs = data.observations.to(device, non_blocking=True)
        s_next_obs = data.next_observations.to(device, non_blocking=True)
        s_actions = data.actions.to(device, non_blocking=True)

        with torch.no_grad():
            clipped_noise = (torch.randn_like(s_actions, device=device) * args.policy_noise).clamp(
                -args.noise_clip, args.noise_clip
            ) * target_actor.action_scale

            # One target-encoder pass for both the target actor and the target critics
            next_features = qf_target.encoder(s_next_obs)
            next_state_actions = (target_actor.action_from_features(next_features) + clipped_noise).clamp(
                action_low, action_high
            )
            qf1_next_target, qf2_next_target = qf_target.q_from_features(next_features, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            next_q_value = data.rewards.to(device) + (1 - data.dones.to(device)) * args.gamma * (min_qf_next_target).view(-1)

        features = qf.encoder(s_obs)
        qf1_a_values, qf2_a_values = qf.q_from_features(features, s_actions)
        qf1_a_values, qf2_a_values = qf1_a_values.view(-1), qf2_a_values.view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # optimize the model
        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()

        metrics = {"qf1_values": qf1_a_values.mean(), "qf2_values": qf2_a_values.mean(),
                   "qf1_loss": qf1_loss, "qf2_loss": qf2_loss, "qf_loss": qf_loss / 2.0}

        if n_updates % args.policy_frequency == 0:
            features = features.detach()
            actor_loss = -qf.q1(features, actor.action_from_features(features)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            metrics["actor_loss"] = actor_loss

            # update the target network
            # Polyak update as one batched op instead of a Python loop over parameters
            with torch.no_grad():
                torch._foreach_lerp_(target_params, online_params, args.tau)
        return metrics

    start_time = time.time()
    # Per-window timing: SPS over the last 100 steps and where the time goes
    perf_window_start, perf_window_step, perf_wait_time, perf_update_time = time.perf_counter(), 0, 0.0, 0.0
    n_updates = 0
    metrics = {}

    def every(interval):
        # True once per `interval` transitions, global_step advances by num_envs per iteration
        return global_step % interval < args.num_envs

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    # Envs whose episode ended on the previous step: their next step only resets them
    # (gymnasium's next-step autoreset) and is not a transition to store
    autoreset = np.zeros(args.num_envs, dtype=bool)
    for iteration in range(args.total_timesteps // args.num_envs + 1):
        global_step = iteration * args.num_envs
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            raw_actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            raw_actions[:, 0] = np.abs(raw_actions[:, 0])
            actions = raw_actions
        else:
            with torch.no_grad():
                actions = actor(torch.from_numpy(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
            actions = actions.cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        # The envs step in their worker processes while the networks update on the buffer
        envs.step_async(actions)

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            t_update = time.perf_counter()
            for _ in range(args.gradient_steps):
                metrics = update(n_updates)
                n_updates += 1
            perf_update_time += time.perf_counter() - t_update

        t_wait = time.perf_counter()
        next_obs, rewards, terminations, truncations, infos = envs.step_wait()
        perf_wait_time += time.perf_counter() - t_wait

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
                    writer.add_scalar("charts/episodic_return", infos['episode']['r'][i], global_step)
                    writer.add_scalar("charts/episodic_length", infos['episode']['l'][i], global_step)  

        # TRY NOT TO MODIFY: save data to reply buffer. With next-step autoreset, next_obs of
        # a finished episode is its real last observation, so no final_observation handling
        rb.add(obs, next_obs, actions, rewards, terminations, valid=~autoreset)
        autoreset = np.logical_or(terminations, truncations)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        if global_step > args.learning_starts:
            if every(100) and metrics:
                for key, value in metrics.items():
                    writer.add_scalar(f"losses/{key}", value.item(), global_step)
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                    
            if args.curriculum_randomization:
                update_curriculum_stage(envs=envs, global_step=global_step, args=args)
            
            if every(args.eval_interval) and global_step >= args.start_evaluation:
                score1, _, _, is_best_p, _ = evaluator_p.evaluate(
                    is_interval=True,
                    global_step=global_step,
                    best_reward=best_eval_reward,
                    num_episodes=10
                )
                score2, _, _, is_best_imp, success_rate = evaluator_imp.evaluate(
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
                        actor=actor, critic=qf, 
                        step=global_step, run_name=run_name, 
                        args=args, env_params=env_params, 
                        suffix=f"vr{args.reward_type}s{args.seed}_BEST"
                    )

        if global_step > 0 and every(100):
            window = time.perf_counter() - perf_window_start
            n = global_step - perf_window_step
            writer.add_scalar("perf/SPS_recent", n / window, global_step)
            # Time blocked waiting for the envs, i.e. stepping not hidden behind the update
            writer.add_scalar("perf/env_wait_ms", 1000 * perf_wait_time / n, global_step)
            writer.add_scalar("perf/update_ms", 1000 * perf_update_time / n, global_step)
            perf_window_start, perf_window_step = time.perf_counter(), global_step
            perf_wait_time, perf_update_time = 0.0, 0.0

    if args.save_model:
        save_models(actor, critic=qf, step=global_step, run_name=run_name, args=args, env_params=env_params, suffix=f"vr{args.reward_type}s{args.seed}_Final")

    eval_env_perfect.close()
    eval_env_imperfect.close()
    envs.close()
    writer.close()

from typing import NamedTuple

import numpy as np
import torch


class FrameSamples(NamedTuple):
    observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class FrameStackReplayBuffer:
    """
    Replay buffer for frame-stacked image observations that lives on the training device.

    The next observation of a transition is the current stack shifted by one frame plus the
    newest frame, so only that newest frame is stored next to the stack: 5 frames per
    transition instead of 8. Sampling then happens on the device, with no host-to-device copy
    of the batch. Observations stay uint8, the encoders normalize them.
    """

    def __init__(self, capacity: int, obs_shape, action_dim: int, frame_channels: int, device):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.c = frame_channels
        frame_shape = (frame_channels, *obs_shape[1:])
        self.obs = torch.zeros((self.capacity, *obs_shape), dtype=torch.uint8, device=self.device)
        self.next_frame = torch.zeros((self.capacity, *frame_shape), dtype=torch.uint8, device=self.device)
        self.actions = torch.zeros((self.capacity, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros(self.capacity, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(self.capacity, dtype=torch.float32, device=self.device)
        self.pos = 0
        self.size = 0
        # The frame-stack layout is checked on the first adds, it holds for the whole run
        self._checks_left = 1000

    def add(self, obs, next_obs, actions, rewards, terminations, valid):
        """Adds the rows of a vector-env step for which `valid` is True."""
        valid = np.asarray(valid, dtype=bool)
        if not valid.any():
            return
        obs, next_obs = obs[valid], next_obs[valid]
        if self._checks_left > 0:
            self._checks_left -= 1
            assert np.array_equal(next_obs[:, : -self.c], obs[:, self.c :]), (
                "next_obs is not obs shifted by one frame: is FrameStackObservation the "
                "last observation transform, with oldest frames first?"
            )
        n = len(obs)
        idx = (self.pos + np.arange(n)) % self.capacity
        idx_t = torch.from_numpy(idx).to(self.device)
        self.obs[idx_t] = torch.from_numpy(obs).to(self.device)
        self.next_frame[idx_t] = torch.from_numpy(np.ascontiguousarray(next_obs[:, -self.c :])).to(self.device)
        self.actions[idx_t] = torch.as_tensor(np.asarray(actions)[valid], dtype=torch.float32).to(self.device)
        self.rewards[idx_t] = torch.as_tensor(np.asarray(rewards)[valid], dtype=torch.float32).to(self.device)
        self.dones[idx_t] = torch.as_tensor(np.asarray(terminations)[valid], dtype=torch.float32).to(self.device)
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int) -> FrameSamples:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        obs = self.obs[idx]
        next_obs = torch.cat([obs[:, self.c :], self.next_frame[idx]], dim=1)
        return FrameSamples(obs, next_obs, self.actions[idx], self.rewards[idx], self.dones[idx])

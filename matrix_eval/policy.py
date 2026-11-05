"""
Standalone policy wrapper for Duckiematrix evaluation.

Rebuilds the trained actor (SAC or TD3) from a .cleanrl_model checkpoint and
reproduces the training observation/action pipeline from utils/rl_env.py
without importing gym_duckietown, so it runs in the separate `duckie-matrix` env.

Observation pipeline (training order, see DuckieOvalEnv.create_wrapped):
    RGB 640x480 -> [CLAHE -> GaussianBlur] -> crop top 1/3 -> resize res x res (INTER_AREA)
    -> grayscale (1,H,W) [-> contrast stretch]  |  or RGB transposed to (3,H,W)
    -> stack last 4 frames on the channel axis (oldest first)

Action pipeline (KinematicActionWrapper, optional ActionSmoothingWrapper before it):
    [v, omega] -> v*v_scale, omega*omega_scale -> wheel PWM, uniformly rescaled to |u| <= 1
"""
import collections
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SIM_CAMERA_SIZE = (640, 480)  # (w, h) the gym_duckietown camera used during training


class ImpalaCNN(nn.Module):
    """Same layer layout as rl/cnn_architectures.ImpalaCNN, so state dict keys match."""

    def __init__(self, in_channels: int, feature_dim: int, flat_dim: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, 16, 8, stride=4), nn.LeakyReLU(),
            nn.Conv2d(16, 32, 4, stride=2), nn.LeakyReLU(), nn.Flatten(),
            nn.Linear(flat_dim, feature_dim),
        )

    def forward(self, obs):
        x = obs.float() / 255.0 - 0.5
        h = self.main(x)
        return F.layer_norm(h, (h.size(-1),))


class Actor(nn.Module):
    """Deterministic actor: SAC uses the squashed mean, TD3 the squashed mu."""

    def __init__(self, algo: str, in_channels: int, feature_dim: int, flat_dim: int, action_dim: int = 2):
        super().__init__()
        self.algo = algo
        self.encoder = ImpalaCNN(in_channels, feature_dim, flat_dim)
        if algo == "sac":
            self.fc_mean = nn.Linear(feature_dim, action_dim)
            self.fc_logstd = nn.Linear(feature_dim, action_dim)
        else:
            self.fc_mu = nn.Linear(feature_dim, action_dim)
        self.register_buffer("action_scale", torch.ones(action_dim))
        self.register_buffer("action_bias", torch.zeros(action_dim))

    def forward(self, x):
        h = self.encoder(x)
        out = self.fc_mean(h) if self.algo == "sac" else self.fc_mu(h)
        a = torch.cat([torch.sigmoid(out[:, 0:1]), torch.tanh(out[:, 1:2])], dim=-1)
        return a * self.action_scale + self.action_bias


def load_camera_calibration(path: str):
    """(K, D, (w, h)) from a Duckiematrix map's cameras.yaml (a map dir or the file itself)."""
    import yaml
    if os.path.isdir(path):
        path = os.path.join(path, "cameras.yaml")
    with open(path) as f:
        cams = yaml.safe_load(f)["cameras"]
    cam = next(iter(cams.values()))
    K = np.array(cam["camera_matrix"], dtype=np.float64).reshape(3, 3)
    D = np.array(cam["distortion_parameters"], dtype=np.float64)
    return K, D, (int(cam["width"]), int(cam["height"]))


@dataclass
class PolicySpec:
    algo: str
    in_channels: int
    grayscale: bool
    frame_stack: int
    resolution: int
    meta: dict


def _infer_spec(state: dict, frame_stack: int) -> PolicySpec:
    algo = "sac" if "fc_mean.weight" in state else "td3"
    in_channels = state["encoder.main.0.weight"].shape[1]
    per_frame = in_channels // frame_stack
    if per_frame not in (1, 3) or in_channels % frame_stack:
        raise ValueError(f"Cannot split {in_channels} input channels into {frame_stack} frames")
    flat_dim = state["encoder.main.5.weight"].shape[1]
    # conv(8, s4) then conv(4, s2): 84 -> 20 -> 9 (32*81), 42 -> 9 -> 3 (32*9)
    side = int(round(np.sqrt(flat_dim / 32)))
    resolution = {9: 84, 3: 42}.get(side)
    if resolution is None:
        raise ValueError(f"Unrecognised encoder flat dim {flat_dim}")
    return PolicySpec(algo, in_channels, per_frame == 1, frame_stack, resolution, {})


class MatrixPolicy:
    def __init__(self, model_path: str, frame_stack: int = 4, preprocessing: bool = False,
                 ema: bool = False, ema_alpha: float = 0.8, v_scale: float = 0.8,
                 omega_scale: float = 3.0, gain: float = 1.0, trim: float = 0.0,
                 wheel_dist: float = 0.102, radius: float = 0.0318, motor_k: float = 27.0,
                 rectify_calib: Optional[str] = None, device: str = "cpu"):
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        state = ckpt["actor_state_dict"]
        self.spec = _infer_spec(state, frame_stack)
        self.spec.meta = {k: v for k, v in ckpt.items() if not k.endswith("state_dict")}
        flat_dim = state["encoder.main.5.weight"].shape[1]
        feature_dim = state["encoder.main.5.weight"].shape[0]

        self.device = torch.device(device)
        self.actor = Actor(self.spec.algo, self.spec.in_channels, feature_dim, flat_dim).to(self.device)
        self.actor.load_state_dict(state)
        self.actor.eval()

        self.name = os.path.splitext(os.path.basename(model_path))[0]
        self.preprocessing = preprocessing
        self.ema, self.ema_alpha = ema, ema_alpha
        self.v_scale, self.omega_scale = v_scale, omega_scale
        self.wheel_dist, self.radius = wheel_dist, radius
        self.k_r_inv = (gain + trim) / motor_k
        self.k_l_inv = (gain - trim) / motor_k
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._calib = None  # (K, D) of the rendered camera, when rectifying
        if rectify_calib:
            K, D, self._calib_wh = load_camera_calibration(rectify_calib)
            self._calib = (K, D)
        self._rect_maps = None
        self.last_view: Optional[np.ndarray] = None  # camera frame as fed to the crop (BGR)
        self.reset()

    def reset(self):
        self.frames = collections.deque(maxlen=self.spec.frame_stack)
        self.prev_action = np.zeros(2, dtype=np.float32)

    # ---- observation ----
    def rectify(self, bgr: np.ndarray) -> np.ndarray:
        """Undistort like the real-robot pipeline (debug/agent.py: alpha=0, same output size)."""
        h, w = bgr.shape[:2]
        if self._rect_maps is None or self._rect_maps[0] != (w, h):
            K, D = self._calib
            sx, sy = w / self._calib_wh[0], h / self._calib_wh[1]
            K = K * np.array([[sx], [sy], [1.0]])
            newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0, (w, h))
            mx, my = cv2.initUndistortRectifyMap(K, D, None, newK, (w, h), cv2.CV_32FC1)
            self._rect_maps = ((w, h), mx, my)
        return cv2.remap(bgr, self._rect_maps[1], self._rect_maps[2], cv2.INTER_LINEAR)

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        if self._calib is not None:
            bgr = self.rectify(bgr)
        self.last_view = bgr
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if (rgb.shape[1], rgb.shape[0]) != SIM_CAMERA_SIZE:
            rgb = cv2.resize(rgb, SIM_CAMERA_SIZE, interpolation=cv2.INTER_AREA)
        if self.preprocessing:
            yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
            yuv[:, :, 0] = self._clahe.apply(yuv[:, :, 0])
            rgb = cv2.GaussianBlur(cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB), (3, 3), 0)
        h = rgb.shape[0]
        res = self.spec.resolution
        img = cv2.resize(rgb[int(h / 3):h], (res, res), interpolation=cv2.INTER_AREA)
        if self.spec.grayscale:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if self.preprocessing:
                gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            return gray[np.newaxis]
        return img.transpose(2, 0, 1)

    def observe(self, bgr: np.ndarray) -> np.ndarray:
        frame = self.preprocess(bgr)
        if not self.frames:  # FrameStackObservation pads with the reset frame
            self.frames.extend([frame] * self.spec.frame_stack)
        else:
            self.frames.append(frame)
        return np.concatenate(self.frames, axis=0)

    # ---- action ----
    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        action = self.actor(x).cpu().numpy().reshape(-1).astype(np.float32)
        if self.ema:
            action = self.ema_alpha * action + (1.0 - self.ema_alpha) * self.prev_action
            self.prev_action = action.copy()
        return action

    def to_pwm(self, action: np.ndarray):
        vel, omega = action[0] * self.v_scale, action[1] * self.omega_scale
        u_r = (vel + 0.5 * omega * self.wheel_dist) / self.radius * self.k_r_inv
        u_l = (vel - 0.5 * omega * self.wheel_dist) / self.radius * self.k_l_inv
        excess = max(abs(u_r), abs(u_l))
        if excess > 1.0:
            u_r, u_l = u_r / excess, u_l / excess
        return float(u_l), float(u_r)

    def describe(self) -> str:
        s = self.spec
        return (f"{self.name}: {s.algo.upper()} | {'gray' if s.grayscale else 'rgb'} x{s.frame_stack} "
                f"{'| rectified ' if self._calib is not None else ''}"
                f"@ {s.resolution}px | step={s.meta.get('global_step')} env_id={s.meta.get('env_id')}")

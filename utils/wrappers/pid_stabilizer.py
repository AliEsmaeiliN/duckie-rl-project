from collections import deque
from dataclasses import dataclass
import gymnasium as gym
import numpy as np


@dataclass
class StabilizerConfig:
    ema_alpha_v: float = 0.8
    ema_alpha_omega: float = 0.65

    kp_v: float = 0.8
    kp_omega: float = 0.9

    ki_v: float = 0.0
    ki_omega: float = 0.0

    kd_v: float = 0.0
    kd_omega: float = 0.0

    v_min: float = -1.0
    v_max: float = 1.0
    omega_min: float = -1.0
    omega_max: float = 1.0

    integral_max: float = 0.5

    # frame_skip=4 at ~30Hz base
    dt: float = 0.033 * 4 


class PIDStabilizerWrapper(gym.Wrapper):
    """
    Hybrid PID + EMA stabilizer.
    """

    def __init__(self, env: gym.Env, config: StabilizerConfig = None):
        super().__init__(env)
        self.cfg = config or StabilizerConfig()

        self._ema_v: float | None = None
        self._ema_omega: float | None = None

        self._integral_v: float = 0.0
        self._integral_omega: float = 0.0

        self._prev_e_v: float = 0.0
        self._prev_e_omega: float = 0.0
        self._d_initialized: bool = False

        self._step_count: int = 0
        self._raw_omegas: deque = deque(maxlen=100)
        self._smooth_omegas: deque = deque(maxlen=100)


    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_state()
        return obs, info

    def step(self, action: np.ndarray):
        v_rl     = float(np.clip(action[0], self.cfg.v_min,     self.cfg.v_max))
        omega_rl = float(np.clip(action[1], self.cfg.omega_min, self.cfg.omega_max))

        if self._ema_v is None:
            self._ema_v     = v_rl
            self._ema_omega = omega_rl

        e_v     = v_rl     - self._ema_v
        e_omega = omega_rl - self._ema_omega

        p_v     = self.cfg.kp_v     * e_v
        p_omega = self.cfg.kp_omega * e_omega

        self._integral_v     = float(np.clip(
            self._integral_v     + e_v     * self.cfg.dt,
            -self.cfg.integral_max, self.cfg.integral_max
        ))
        self._integral_omega = float(np.clip(
            self._integral_omega + e_omega * self.cfg.dt,
            -self.cfg.integral_max, self.cfg.integral_max
        ))
        i_v     = self.cfg.ki_v     * self._integral_v
        i_omega = self.cfg.ki_omega * self._integral_omega

        if not self._d_initialized:
            d_v = d_omega = 0.0
            self._prev_e_v     = e_v
            self._prev_e_omega = e_omega
            self._d_initialized = True
        else:
            d_v     = self.cfg.kd_v     * (e_v     - self._prev_e_v)     / self.cfg.dt
            d_omega = self.cfg.kd_omega * (e_omega - self._prev_e_omega) / self.cfg.dt
            self._prev_e_v     = e_v
            self._prev_e_omega = e_omega

        v_target     = self._ema_v     + p_v     + i_v     + d_v
        omega_target = self._ema_omega + p_omega + i_omega + d_omega

        self._ema_v     = self.cfg.ema_alpha_v     * v_target     + (1 - self.cfg.ema_alpha_v)     * self._ema_v
        self._ema_omega = self.cfg.ema_alpha_omega * omega_target + (1 - self.cfg.ema_alpha_omega) * self._ema_omega

        v_out     = float(np.clip(self._ema_v,     self.cfg.v_min,     self.cfg.v_max))
        omega_out = float(np.clip(self._ema_omega, self.cfg.omega_min, self.cfg.omega_max))

        obs, reward, terminated, truncated, info = self.env.step(
            np.array([v_out, omega_out], dtype=np.float32)
        )

        self._step_count += 1
        self._raw_omegas.append(omega_rl)
        self._smooth_omegas.append(omega_out)

        raw_jerk    = float(np.mean(np.abs(np.diff(list(self._raw_omegas)))))    if len(self._raw_omegas)    > 1 else 0.0
        smooth_jerk = float(np.mean(np.abs(np.diff(list(self._smooth_omegas))))) if len(self._smooth_omegas) > 1 else 0.0

        info["pid_stabilizer"] = {
            "v_rl":     v_rl,     "v_out":     v_out,
            "omega_rl": omega_rl, "omega_out": omega_out,
            "e_v":      e_v,      "e_omega":   e_omega,
            "omega_jerk_raw":      raw_jerk,
            "omega_jerk_smooth":   smooth_jerk,
            "jerk_reduction_pct":  100.0 * (1 - smooth_jerk / raw_jerk) if raw_jerk > 1e-6 else 0.0,
            "step": self._step_count,
        }

        return obs, reward, terminated, truncated, info

    #  Runtime tuning

    def set_ema_alphas(self, alpha_v: float, alpha_omega: float) -> None:
        self.cfg.ema_alpha_v     = float(np.clip(alpha_v,     0.0, 1.0))
        self.cfg.ema_alpha_omega = float(np.clip(alpha_omega, 0.0, 1.0))

    def set_gains(self, kp_v=None, kp_omega=None,
                  ki_v=None, ki_omega=None,
                  kd_v=None, kd_omega=None) -> None:
        """Hot-swap any gain without resetting integrators."""
        if kp_v     is not None: self.cfg.kp_v     = kp_v
        if kp_omega is not None: self.cfg.kp_omega = kp_omega
        if ki_v     is not None: self.cfg.ki_v     = ki_v
        if ki_omega is not None: self.cfg.ki_omega = ki_omega
        if kd_v     is not None: self.cfg.kd_v     = kd_v
        if kd_omega is not None: self.cfg.kd_omega = kd_omega


    def _reset_state(self) -> None:
        self._ema_v            = None
        self._ema_omega        = None
        self._integral_v       = 0.0
        self._integral_omega   = 0.0
        self._prev_e_v         = 0.0
        self._prev_e_omega     = 0.0
        self._d_initialized    = False
        self._step_count       = 0
        self._raw_omegas.clear()
        self._smooth_omegas.clear()
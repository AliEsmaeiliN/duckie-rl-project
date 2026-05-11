from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
import numpy as np

@dataclass
class PIDConfig:
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    output_min: float = -1.0
    output_max: float = 1.0
    integral_min: float = -1.0
    integral_max: float = 1.0
    deriv_alpha: float = 0.1 # Low-pass filter coefficient for derivative (0 = no filter, 1 = ignore new)


class PIDController:
    """
    Single-axis PID with:
      - Clamped integrator (anti-windup)
      - Exponential low-pass filter on derivative (dirty derivative)
      - Configurable output saturation
    """

    def __init__(self, config: PIDConfig):
        self.cfg = config
        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._filtered_deriv: float = 0.0
        self._initialized: bool = False

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._filtered_deriv = 0.0
        self._initialized = False

    def step(self, error: float, dt: float) -> Tuple[float, Dict[str, float]]:
        """
        Compute PID output for given error and timestep.

        Returns:
            output  : clipped control signal
            debug   : dict with P, I, D components
        """
        dt = max(dt, 1e-6)

        # On first call, skip derivative (avoids derivative kick)
        if not self._initialized:
            self._prev_error = error
            self._initialized = True

        P = self.cfg.kp * error

        self._integral += error * dt
        self._integral = float(
            np.clip(self._integral, self.cfg.integral_min, self.cfg.integral_max)
        )
        I = self.cfg.ki * self._integral

        raw_deriv = (error - self._prev_error) / dt
        self._filtered_deriv = (
            self.cfg.deriv_alpha * raw_deriv
            + (1.0 - self.cfg.deriv_alpha) * self._filtered_deriv
        )
        D = self.cfg.kd * self._filtered_deriv

        self._prev_error = error

        output = float(np.clip(P + I + D, self.cfg.output_min, self.cfg.output_max))
        return output, {"P": P, "I": I, "D": D, "integral": self._integral}


@dataclass
class StabilizerConfig:
    speed_pid: PIDConfig = field(default_factory=lambda: PIDConfig(
        kp=2.0, ki=0.05, kd=0.1,
        output_min=-0.5, output_max=0.5,     # delta applied on top of v_rl
        integral_min=-0.3, integral_max=0.3,
        deriv_alpha=0.15,
    ))

    # ── Steering PID (tracks ω_rl) ────────────────────────────────────────
    steering_pid: PIDConfig = field(default_factory=lambda: PIDConfig(
        kp=2.5, ki=0.08, kd=0.3,
        output_min=-2.0, output_max=2.0,     # rad/s
        integral_min=-1.0, integral_max=1.0,
        deriv_alpha=0.12,
    ))

    
    ema_alpha_v: float = 0.6       # speed smoothing
    ema_alpha_omega: float = 0.4   # steering smoothing (more aggressive)

    v_min: float = 0.0
    v_max: float = 1
    omega_min: float = -1.0
    omega_max: float = 1.0

    dt_nominal: float = 0.033      # seconds (≈30 Hz)

    log_window: int = 100          # rolling window for statistics


class PIDStabilizerWrapper(gym.Wrapper):
    """
    Hybrid PID Stabilizer Wrapper.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env: gym.Env,
        config: Optional[StabilizerConfig] = None,
    ):
        super().__init__(env)
        self.cfg = config or StabilizerConfig()

        self._speed_pid = PIDController(self.cfg.speed_pid)
        self._steering_pid = PIDController(self.cfg.steering_pid)

        # EMA state
        self._ema_v: Optional[float] = None
        self._ema_omega: Optional[float] = None

        # Timing
        self._last_step_time: Optional[float] = None

        # Rolling statistics for W&B logging
        self._omega_errors: deque = deque(maxlen=self.cfg.log_window)
        self._v_errors: deque = deque(maxlen=self.cfg.log_window)
        self._raw_omegas: deque = deque(maxlen=self.cfg.log_window)
        self._smooth_omegas: deque = deque(maxlen=self.cfg.log_window)
        self._step_count: int = 0

    

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        obs, info = self.env.reset(**kwargs)
        self._reset_state()
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict]:
        """
        Args:
            action: np.array([v, omega]) from RL policy

        Returns:
            Standard (obs, reward, terminated, truncated, info) tuple.
            info["pid_stabilizer"] contains full diagnostics.
        """
        
        v_rl = float(np.clip(action[0], self.cfg.v_min, self.cfg.v_max))
        omega_rl = float(np.clip(action[1], self.cfg.omega_min, self.cfg.omega_max))

        now = time.monotonic()
        dt = (
            now - self._last_step_time
            if self._last_step_time is not None
            else self.cfg.dt_nominal
        )
        dt = float(np.clip(dt, 0.005, 0.2))   # guard against outliers
        self._last_step_time = now

        if self._ema_v is None:
            self._ema_v = v_rl
            self._ema_omega = omega_rl

        e_v = v_rl - self._ema_v
        e_omega = omega_rl - self._ema_omega

        delta_v, speed_dbg = self._speed_pid.step(e_v, dt)
        delta_omega, steer_dbg = self._steering_pid.step(e_omega, dt)

        v_pid = self._ema_v + delta_v
        omega_pid = self._ema_omega + delta_omega

        alpha_v = self.cfg.ema_alpha_v
        alpha_w = self.cfg.ema_alpha_omega

        self._ema_v = alpha_v * v_pid + (1.0 - alpha_v) * self._ema_v
        self._ema_omega = alpha_w * omega_pid + (1.0 - alpha_w) * self._ema_omega

        v_out = float(np.clip(self._ema_v, self.cfg.v_min, self.cfg.v_max))
        omega_out = float(np.clip(self._ema_omega, self.cfg.omega_min, self.cfg.omega_max))

        stabilized_action = np.array([v_out, omega_out], dtype=np.float32)

        obs, reward, terminated, truncated, info = self.env.step(stabilized_action)

        self._step_count += 1
        self._omega_errors.append(abs(e_omega))
        self._v_errors.append(abs(e_v))
        self._raw_omegas.append(omega_rl)
        self._smooth_omegas.append(omega_out)

        info["pid_stabilizer"] = self._build_log(
            v_rl=v_rl, omega_rl=omega_rl,
            v_out=v_out, omega_out=omega_out,
            e_v=e_v, e_omega=e_omega,
            delta_v=delta_v, delta_omega=delta_omega,
            speed_dbg=speed_dbg, steer_dbg=steer_dbg,
            dt=dt,
        )

        return obs, reward, terminated, truncated, info


    def _reset_state(self) -> None:
        self._speed_pid.reset()
        self._steering_pid.reset()
        self._ema_v = None
        self._ema_omega = None
        self._last_step_time = None
        self._step_count = 0
        self._omega_errors.clear()
        self._v_errors.clear()
        self._raw_omegas.clear()
        self._smooth_omegas.clear()

    def _build_log(self, **kwargs) -> Dict[str, Any]:
        raw_omegas = list(self._raw_omegas)
        smooth_omegas = list(self._smooth_omegas)

        # Jerk = mean absolute difference between consecutive omega values
        raw_jerk = (
            float(np.mean(np.abs(np.diff(raw_omegas)))) if len(raw_omegas) > 1 else 0.0
        )
        smooth_jerk = (
            float(np.mean(np.abs(np.diff(smooth_omegas)))) if len(smooth_omegas) > 1 else 0.0
        )

        return {
            # Raw vs stabilized actions
            "v_rl": kwargs["v_rl"],
            "omega_rl": kwargs["omega_rl"],
            "v_out": kwargs["v_out"],
            "omega_out": kwargs["omega_out"],
            # Tracking errors
            "e_v": kwargs["e_v"],
            "e_omega": kwargs["e_omega"],
            # PID corrections
            "delta_v": kwargs["delta_v"],
            "delta_omega": kwargs["delta_omega"],
            # PID internals (speed)
            "speed_P": kwargs["speed_dbg"]["P"],
            "speed_I": kwargs["speed_dbg"]["I"],
            "speed_D": kwargs["speed_dbg"]["D"],
            "speed_integral": kwargs["speed_dbg"]["integral"],
            # PID internals (steering)
            "steer_P": kwargs["steer_dbg"]["P"],
            "steer_I": kwargs["steer_dbg"]["I"],
            "steer_D": kwargs["steer_dbg"]["D"],
            "steer_integral": kwargs["steer_dbg"]["integral"],
            # Rolling statistics
            "mean_omega_error": float(np.mean(self._omega_errors)) if self._omega_errors else 0.0,
            "mean_v_error": float(np.mean(self._v_errors)) if self._v_errors else 0.0,
            "omega_jerk_raw": raw_jerk,
            "omega_jerk_smooth": smooth_jerk,
            "jerk_reduction_pct": (
                100.0 * (1.0 - smooth_jerk / raw_jerk) if raw_jerk > 1e-6 else 0.0
            ),
            # Timing
            "dt": kwargs["dt"],
            "step": self._step_count,
        }


    def set_steering_gains(self, kp: float, ki: float, kd: float) -> None:
        """Hot-swap steering PID gains without resetting integrators."""
        self._steering_pid.cfg.kp = kp
        self._steering_pid.cfg.ki = ki
        self._steering_pid.cfg.kd = kd

    def set_speed_gains(self, kp: float, ki: float, kd: float) -> None:
        self._speed_pid.cfg.kp = kp
        self._speed_pid.cfg.ki = ki
        self._speed_pid.cfg.kd = kd

    def set_ema_alphas(self, alpha_v: float, alpha_omega: float) -> None:
        """Adjust EMA smoothing at runtime (e.g. loosen during eval)."""
        self.cfg.ema_alpha_v = float(np.clip(alpha_v, 0.0, 1.0))
        self.cfg.ema_alpha_omega = float(np.clip(alpha_omega, 0.0, 1.0))
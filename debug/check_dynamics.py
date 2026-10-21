#!/usr/bin/env python3
"""
Checks that gym_duckietown.dynamics.DB18Dynamics matches duckietown_world's DB18 model
(the PyGeometry implementation the simulator used before) and reports the speedup.

Usage: python debug/check_dynamics.py
"""
import time

import geometry
import numpy as np
from duckietown_world import get_DB18_nominal, get_DB18_uncalibrated
from duckietown_world.world_duckietown.pwm_dynamics import PWMCommands

from gym_duckietown.dynamics import DB18Dynamics

DT = 1 / 30


def run_reference(trim, actions, x0, y0, th0):
    p = get_DB18_uncalibrated(delay=0, trim=trim) if trim else get_DB18_nominal(delay=0)
    q = geometry.SE2_from_translation_angle(np.array([x0, y0]), th0)
    state = p.initialize(c0=(q, geometry.se2_from_linear_angular(np.array([0, 0]), 0)), t0=0)
    poses = []
    for left, right in actions:
        state = state.integrate(DT, PWMCommands(motor_left=left, motor_right=right))
        t, a = geometry.translation_angle_from_SE2(state.TSE2_from_state()[0])
        poses.append((t[0], t[1], a))
    return np.array(poses)


def run_numpy(trim, actions, x0, y0, th0):
    state = DB18Dynamics(x0, y0, th0, trim=trim)
    poses = []
    for left, right in actions:
        state.integrate(DT, motor_left=left, motor_right=right)
        poses.append(state.pose)
    return np.array(poses)


def main():
    rng = np.random.default_rng(0)
    worst = 0.0
    for episode in range(20):
        trim = 0.0 if episode % 2 == 0 else rng.uniform(-0.1, 0.1)
        # Mix of random, saturated (|u| > 1, clipped by the model) and straight commands
        actions = rng.uniform(-1.2, 1.2, size=(600, 2))
        actions[::7] = actions[::7, :1]
        x0, y0, th0 = rng.uniform(0, 3), rng.uniform(0, 3), rng.uniform(-np.pi, np.pi)
        ref = run_reference(trim, actions, x0, y0, th0)
        new = run_numpy(trim, actions, x0, y0, th0)
        diff = np.abs(ref - new)
        diff[:, 2] = np.abs(np.angle(np.exp(1j * (ref[:, 2] - new[:, 2]))))  # heading modulo 2*pi
        worst = max(worst, diff.max())
    print(f"max abs difference over 20 x 600 steps: {worst:.2e}")
    assert worst < 1e-9, "numpy dynamics diverge from duckietown_world"

    n = 3000
    actions = rng.uniform(-1, 1, size=(n, 2))
    for name, fn in [("duckietown_world", run_reference), ("numpy", run_numpy)]:
        t = time.perf_counter()
        fn(0.0, actions, 1.0, 1.0, 0.0)
        print(f"{name:17s} {1e3 * (time.perf_counter() - t) / n:.4f} ms per integrate")
    print("OK")


if __name__ == "__main__":
    main()

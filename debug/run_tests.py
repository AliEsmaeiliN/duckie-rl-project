#!/usr/bin/env python3
"""
Smoke test for a duckie-rl setup: GPU, OpenGL rendering and the oval training env.

Usage (from the repo root):  python debug/run_tests.py
"""
import sys

import numpy as np
import torch

from gym_duckietown.check_hw import get_graphics_information
from utils.rl_env import DuckieOvalEnv


def test_hardware():
    print("\n=== 1. Hardware & training dependencies ===")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        torch.rand(5, 3).cuda()  # a kernel actually runs on the GPU
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        print("MPS (Apple GPU): available")
    else:
        print("WARNING: no GPU, training will run on the CPU")

    import tensorboard, tyro, wandb  # noqa: F401  imported by the training scripts
    print("wandb, tensorboard, tyro: installed")


def test_rendering(env):
    print("\n=== 2. Oval env & rendering ===")
    print(f"OpenGL: {get_graphics_information()}")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4, 84, 84) and obs.dtype == np.uint8, f"unexpected observation {obs.shape} {obs.dtype}"
    first = obs[-1].astype(float)
    for _ in range(10):
        obs, _, terminated, truncated, _ = env.step(np.array([0.5, 0.0], dtype=np.float32))
        if terminated or truncated:
            break
    last = obs[-1].astype(float)

    # A correct frame shows lane paint (5-8% of the pixels above 150) and changes while driving.
    # Broken GL libraries can render flat grey polygons that never change (both ~0) without
    # raising any error, e.g. the conda env's libglvnd mixed with the system's Mesa EGL.
    paint = (first > 150).mean()
    change = np.abs(last - first).mean()
    hint = "rendering looks broken, check which OpenGL/EGL libraries the process loads"
    assert paint > 0.01, f"no lane markings in the camera frame ({paint:.1%} bright pixels): {hint}"
    assert change > 2.0, f"camera frame did not change while driving (mean change {change:.2f}): {hint}"
    print(f"Rendering: SUCCESS ({paint:.1%} lane-paint pixels, mean frame change {change:.1f})")


def test_spawns(env):
    print("\n=== 3. Spawns on the oval ===")
    sim = env.unwrapped
    for i in range(20):
        env.reset()
        assert sim._valid_pose(sim.cur_pos, sim.cur_angle), f"spawn {i} is off the drivable tiles"
        lane = sim.get_lane_pos2(sim.cur_pos, sim.cur_angle)  # raises NotInLane outside a lane
        # spawn difficulty 0 places the robot on the lane centre line, aligned with it
        assert abs(lane.dist) < 0.02 and abs(lane.angle_deg) < 5, f"spawn {i} off centre: {lane}"
    print("20 spawns on the lane centre: SUCCESS")


if __name__ == "__main__":
    print("Starting duckie-rl validation...")
    try:
        test_hardware()
        env = DuckieOvalEnv.create_wrapped(run_name="run_tests", domain_rand=False)
        test_rendering(env)
        test_spawns(env)
        env.close()
        print("\nALL TESTS PASSED! Your environment is ready for training.")
    except Exception as e:
        print(f"\nTEST FAILED: {type(e).__name__}: {e}")
        sys.exit(1)

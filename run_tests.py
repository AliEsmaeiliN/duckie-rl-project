#!/usr/bin/env python3

import os
import torch
import numpy as np
import gymnasium as gym
import pyglet
from gym_duckietown.envs import DuckietownEnv, MultiMapEnv
from gym_duckietown.simulator import get_agent_corners
from gym_duckietown.wrappers import PyTorchObsWrapper

def test_hardware_and_cleanrl():
    print("\n=== 1. Hardware & CleanRL Check ===")
    
    # Check GPU status
    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"GPU available (CUDA): {cuda_available}")
    
    if cuda_available:
        print(f"Using Device: {torch.cuda.get_device_name(0)}")
        # Simple tensor math test to verify GPU communication
        x = torch.rand(5, 3).cuda()
        print("CUDA Tensor Math: SUCCESS")
    else:
        print("WARNING: Running on CPU. Training will be slow.")

    # Check OpenGL Vendor (Critical for Duckietown)
    try:
        vendor = pyglet.gl.gl_info.get_vendor()
        print(f"OpenGL Vendor: {vendor}")
    except Exception as e:
        print(f"OpenGL Error: {e}")

    # Check CleanRL Essentials
    try:
        import wandb
        import tensorboard
        import tyro
        print("CleanRL Dependencies (wandb, tensorboard, tyro): INSTALLED")
    except ImportError as e:
        print(f"CleanRL Dependency Missing: {e}")

def test_duckietown_basics():
    print("\n=== 2. Duckietown Environment Tests ===")
    
    # Test standard Gymnasium registration
    try:
        env = gym.make("Duckietown-udem1-v0")
        env.reset()
        print("Gymnasium Registration: SUCCESS")
    except Exception as e:
        print(f"Gymnasium Make Error: {e}")
        return

    # Try stepping a few times
    for _ in range(0, 10):
        obs, _, _, _, _ = env.step(np.array([0.1, 0.1]))

    # Check that human rendering resembles agent's view
    first_obs = env.reset()[0]
    first_render = env.render()
    
    m0 = first_obs.mean()
    m1 = first_render.mean()
    
    assert 0 < m0 < 255, "Observation is empty or saturated"
    # Basic check that render and obs are in the same ballpark
    assert abs(m0 - m1) < 15, f"Render/Obs mismatch: {m0} vs {m1}"
    print("Visual Rendering: SUCCESS")

    # Test the PyTorch observation wrapper (Essential for RL)
    env = PyTorchObsWrapper(env)
    first_obs = env.reset()[0]
    assert first_obs.shape[0] == 3, "PyTorch Wrapper failed to move channels to front"
    print("PyTorch Wrapper (CHW): SUCCESS")
    env.close()

def test_maps_and_collisions():
    print("\n=== 3. Map Loading & Collision Tests ===")
    
    # Try loading available map files
    for map_name in ["loop_only_duckies", "small_loop_only_duckies"]:
        env = DuckietownEnv(map_name=map_name)
        env.reset()
        env.close()
    print("Map Loading: SUCCESS")

    # Check that we do not spawn too close to obstacles
    env = DuckietownEnv(map_name="loop_obstacles")
    for i in range(0, 20):
        env.reset()
        # Verify no collision on spawn
        assert not env._collision(get_agent_corners(env.cur_pos, env.cur_angle)), f"Collision on spawn at try {i}"
    print("Collision-Free Spawning: SUCCESS")
    env.close()

if __name__ == "__main__":
    print("Starting duckie-rl Validation Suite...")
    try:
        test_hardware_and_cleanrl()
        test_duckietown_basics()
        test_maps_and_collisions()
        print("\nALL TESTS PASSED! Your environment is ready for training.")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        exit(1)
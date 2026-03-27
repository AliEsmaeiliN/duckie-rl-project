#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""
from PIL import Image
import argparse
import sys
import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import pyglet
from pyglet.window import key

from src.gym_duckietown.envs import DuckietownEnv
from utils.wrappers import CropResizeWrapper, DebugRewardWrapper
from utils.env_lunch import EnvLunch

# from experiments.utils import save_img

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default= "Custom")
parser.add_argument("--map-name", default="oval_loop")
parser.add_argument("--distortion", default=False, action="store_true")
parser.add_argument("--camera_rand", default=False, action="store_true")
parser.add_argument("--draw-curve", action="store_true", help="draw the lane following curve")
parser.add_argument("--draw-bbox", action="store_true", help="draw collision detection bounding boxes")
parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
parser.add_argument("--dynamics_rand", action="store_true", help="enable dynamics randomization")
parser.add_argument("--frame-skip", default=1, type=int, help="number of frames to skip")
parser.add_argument("--seed", default=1, type=int, help="seed")
parser.add_argument("--no-grayscale", dest="grayscale", action="store_false", help="Disable the grayscale wrapper (default is True)")
args = parser.parse_args()


if args.env_name and args.env_name.find("Duckietown") != -1:
    env = DuckietownEnv(
        seed=args.seed,
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=4,
        distortion=args.distortion,
        camera_rand=args.camera_rand,
        dynamics_rand=args.dynamics_rand,
        accept_start_angle_deg = 4,
    )
    env = CropResizeWrapper(env, shape=(84, 84))
    env = DebugRewardWrapper(env)
else:
    env_lunch = EnvLunch(
        run_name="manaul_control",
        max_steps=5000,
        grayscale=args.grayscale,
        domain_rand = args.domain_rand,
        distortion = args.distortion,
        camera_rand=args.camera_rand,
        dynamics_rand=args.dynamics_rand,
        )
    env_func = env_lunch.make_env_fn(seed=1, idx=0)
    env = env_func()
    

render_modes = ["human", "top_down", "free_cam", "rgb_array"]
view = render_modes[1]

env.reset(seed=args.seed)

pure_internal_obs = env.unwrapped.render_obs()
print(f"Internal Renderer Shape: {pure_internal_obs.shape}")

env.unwrapped.render(mode=view)

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        env.reset()
        env.unwrapped.render(mode=view)
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)



# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)


def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    wheel_distance = 0.102
    min_rad = 0.08

    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action += np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action -= np.array([0.44, 0])
    if key_handler[key.LEFT]:
        action += np.array([0, 1])
    if key_handler[key.RIGHT]:
        action -= np.array([0, 1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    v1 = action[0]
    v2 = action[1]
    # Limit radius of curvature
    if v1 == 0 or abs(v2 / v1) > (min_rad + wheel_distance / 2.0) / (min_rad - wheel_distance / 2.0):
        # adjust velocities evenly such that condition is fulfilled
        delta_v = (v2 - v1) / 2 - wheel_distance / (4 * min_rad) * (v1 + v2)
        v1 += delta_v
        v2 -= delta_v

    action[0] = v1
    action[1] = v2

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, truncated, info = env.step(action)
    #print("step_count = %s, reward=%.3f" % (env.unwrapped.step_count, reward))
    #print(f"Observation shape: {obs.shape}")

    if key_handler[key.RETURN]:
        os.makedirs("screenshots", exist_ok=True)

        if obs.shape[0] in [1, 3, 4, 9, 12]:
            cnn_view = obs.transpose(1, 2, 0)
        else:
            cnn_view = obs
        
        suffix = "GrayScale"
        if args.grayscale:
            cnn_view_final = cnn_view[:, :, -1] 
            mode = 'L'
        else:
            cnn_view_final = cnn_view[:, :, -3:]
            mode = 'RGB'
            suffix = "RGB"
        
        raw_view = env.unwrapped.render_obs()

        cnn_img = Image.fromarray(cnn_view_final, mode=mode).convert('RGB')
        #cnn_img = cnn_img.resize((160, 120), Image.NEAREST)
        raw_img = Image.fromarray(raw_view).resize((160, 120), Image.NEAREST)

        # Combine into one image
        combined = Image.new('RGB', (320, 120))
        combined.paste(cnn_img, (0, 0))
        combined.paste(raw_img, (160, 0))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"screenshots/debug_{suffix}_{timestamp}.png"
        combined.save(file_path)
        print(f"Comparison saved! Mode: {'Grayscale' if args.grayscale else 'RGB'}")
        print(f"CNN Input Shape: {obs.shape} | Visualized Shape: {cnn_view_final.shape}")

    if done:
        print("done!")
        env.reset()
        env.unwrapped.render(mode=view)

    env.unwrapped.render(mode=view)


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()

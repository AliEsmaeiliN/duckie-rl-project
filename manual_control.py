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
from utils.wrappers import CropResizeWrapper
from utils.rl_env import DuckieOvalEnv
from rl.models import SACActor, TD3Actor
from rl.agent import DuckiebotAgent


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
parser.add_argument("--motion-blur", default=False, action="store_true")
parser.add_argument("--debug", default=False, action="store_true", help="Activating the Agent's action output")
parser.add_argument("--model", default=None, type=str, help="The agent's .cleanrl_model file")
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
else:
    env = DuckieOvalEnv.create_wrapped(
        run_name="manual_control",
        motion_blur=args.motion_blur, 
        grayscale=True,
        frame_stack=4,
        
        domain_rand=args.domain_rand,
        dynamics_rand=args.dynamics_rand,
        distortion=args.distortion
    )

render_modes = ["human", "top_down", "free_cam", "rgb_array"]
view = render_modes[0]

obs, info = env.reset(seed=args.seed)

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

rl_agent = None

if args.debug:
    if args.model:
        model_path = args.model
        algo = "sac" if "sac" in model_path.lower() else "td3"
        rl_agent = DuckiebotAgent(
            model_path=model_path, 
            algo_type= algo
            )
        print(f"{algo.upper()} Agent loaded from {args.model}")    

rl_label = pyglet.text.Label(
    'Agent: Waiting...',
    font_name='Arial',
    font_size=12,
    x=5, y=40,
    anchor_x='left', anchor_y='top',
    color=(255, 255, 0, 255), # Yellow color to make it stand out
    multiline=True, 
    width=800
    )


def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    global obs

    if rl_agent is not None:
        
        # Simulator raw observation with sim2real preprocessing
        last_obs = env.unwrapped.render_obs()
        #print(last_obs.shape)
        processed_frame = rl_agent.preprocess(last_obs)
        rl_agent.update_buffer(processed_frame)
        rl_action = rl_agent.get_action(last_obs)
        rl_action_motor = rl_agent.postprocess_kinematics(rl_action)
        raw_obs_action = f"RL Raw Action: v={rl_action[0]:.2f}, omega={rl_action[1]:.2f} , Motor Action: l={rl_action_motor[0]:.2f}, r={rl_action_motor[1]:.2f}"

        # Env wrapped observation
        stack_obs = obs
        #print(stack_obs.shape)
        rl_action = rl_agent.get_action(is_stacked=True, processed_stack_input=stack_obs)
        rl_action_motor = rl_agent.postprocess_kinematics(rl_action)
        wrapped_obs_action = f"RL Env Action: v={rl_action[0]:.2f}, omega={rl_action[1]:.2f} , Motor Action: l={rl_action_motor[0]:.2f}, r={rl_action_motor[1]:.2f}"
        rl_label.text = f"{raw_obs_action}, \n{wrapped_obs_action}"

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
        obs, _ = env.reset()
        env.unwrapped.render(mode=view)

    env.unwrapped.render(mode=view)
    rl_label.draw()


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()

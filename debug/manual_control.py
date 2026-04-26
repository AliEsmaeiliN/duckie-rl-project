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
import torch

from gym_duckietown.simulator import WINDOW_WIDTH, WINDOW_HEIGHT
from gym_duckietown.envs import DuckietownEnv
from utils.wrappers import CropResizeWrapper
from rl_env_debug import DuckieOvalEnv
from agent import DuckiebotAgent
from wrappers_debug import DebugRewardWrapper, ActionWrapper, KinematicActionWrapper
from rl.sac_continuous_action import Actor as sac_sim_actor
from rl.td3_continuous_action import Actor as td3_sim_actor


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
auto_mode = False

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
    global auto_mode

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        env.reset()
        env.unwrapped.render(mode=view)
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)
    elif symbol == key.A:
        auto_mode = not auto_mode
        print(f"Autonomous Mode: {auto_mode}")



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
        env.single_observation_space = env.observation_space
        env.single_action_space = env.action_space
        actor_class = sac_sim_actor if algo == "sac" else td3_sim_actor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rl_sim_agent = actor_class(env).to(device)
        checkpoint = torch.load(model_path, map_location=rl_sim_agent.fc_mean.weight.device)
        rl_sim_agent.load_state_dict(checkpoint['actor_state_dict'])
        rl_sim_agent.eval()
        print(f"{algo.upper()} Agent loaded from {args.model}")

rl_label = pyglet.text.Label(
    'Agent: Waiting...',
    font_name='Arial',
    font_size=10,
    x=5, y=40,
    anchor_x='left', anchor_y='top',
    color=(255, 255, 0, 255), # Yellow color to make it stand out
    multiline=True, 
    width=800
    )

debug_label = pyglet.text.Label(
    'Step: 0 | Reward: 0.00',
    font_name='Arial',
    font_size=10,
    x=5, y=WINDOW_HEIGHT - 60, 
    anchor_x='left', anchor_y='top',
    color=(0, 255, 0, 255),
    multiline=True,
    width=WINDOW_WIDTH - 10
)

ep_return = 0.0
prev_ep_return = "0.00"

def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    global obs, ep_return, prev_ep_return, device, auto_mode

    if rl_agent is not None:
        
        # Simulator raw observation with sim2real preprocessing and agent
        last_obs = env.unwrapped.render_obs()
        processed_frame = rl_agent.preprocess(last_obs)
        rl_agent.update_buffer(processed_frame)
        rl_action = rl_agent.get_action(last_obs)
        rl_action_motor = rl_agent.postprocess_kinematics(rl_action)
        raw_obs_action = f"RL Raw Action: v={rl_action[0]:.2f}, omega={rl_action[1]:.2f} , Motor Action: l={rl_action_motor[0]:.2f}, r={rl_action_motor[1]:.2f}"

        # Env wrapped observation through the training agents
        stack_obs = torch.Tensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():   
            if hasattr(rl_sim_agent, "get_action"):
                _, _, sim_action = rl_sim_agent.get_action(stack_obs) # Use mean_action for eval
            else:
                sim_action = rl_sim_agent(stack_obs) #TD3 actor returns action directly
            sim_action = sim_action.cpu().numpy().reshape(-1)
        sim_action_motor = rl_agent.postprocess_kinematics(sim_action)
        wrapped_obs_action = f"RL Env Action: v={sim_action[0]:.2f}, omega={sim_action[1]:.2f} , Motor Action: l={sim_action_motor[0]:.2f}, r={sim_action_motor[1]:.2f}"
        rl_label.text = f"{raw_obs_action}, \n{wrapped_obs_action}"


    wheel_distance = 0.102
    min_rad = 0.08

    manual_action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        manual_action += np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        manual_action -= np.array([0.44, 0])
    if key_handler[key.LEFT]:
        manual_action += np.array([0, 1])
    if key_handler[key.RIGHT]:
        manual_action -= np.array([0, 1])
    if key_handler[key.SPACE]:
        manual_action = np.array([0, 0])

    
    v1 = manual_action[0]
    v2 = manual_action[1]
    # Limit radius of curvature
    if v1 == 0 or abs(v2 / v1) > (min_rad + wheel_distance / 2.0) / (min_rad - wheel_distance / 2.0):
        # adjust velocities evenly such that condition is fulfilled
        delta_v = (v2 - v1) / 2 - wheel_distance / (4 * min_rad) * (v1 + v2)
        v1 += delta_v
        v2 -= delta_v

    manual_action[0] = v1
    manual_action[1] = v2

    # Speed boost
    if key_handler[key.LSHIFT]:
        manual_action *= 1.5

    if auto_mode:
        action = sim_action 
    else:
        action = manual_action

    obs, reward, done, truncated, info = env.step(action)

    ep_return += reward

    if done: 
        total_r = info["episode"]["r"]
        if isinstance(total_r, (np.ndarray, list)):
            total_r = total_r[0]
        
        prev_ep_return = f"{total_r:.2f}"
        print(f"DONE! Final Return: {prev_ep_return}")
        ep_return = 0.0

    sim = env.unwrapped
    
    sim_info = info.get("Simulator", {})
    curr = env
    reward_layer = None
    while hasattr(curr, 'env'):
        if isinstance(curr, DebugRewardWrapper):
            reward_layer = curr
            break
        curr = curr.env

    if reward_layer:
        comps = reward_layer.latest_reward_components
        reward_text = (
            f"--- REWARD BREAKDOWN (Step {sim.step_count}) ---\n"
            f"Reward: {reward:.2f} | Total: {ep_return:.2f}\n"
            f"Status: {sim_info.get('msg', 'Normal')}\n"
            f"Action L/R: {sim.last_action[0]:.2f}, {sim.last_action[1]:.2f}\n"
            f"Speed: {sim.speed:.2f} m/s | Lane Dist: {sim_info.get('lane_position', {}).get('dist', 0):.4f}\n"
            f"Distance: {comps['dist']:.4f} ,Heading: {comps['heading']:.4f}"
            f"SPEED REW: {comps['speed']:.4f}\n"
            f"DIST REW:  {comps['distance']:.4f}\n"
            f"ALIGN REW: {comps['alignment']:.4f}\n"
            f"JERK REW:  {comps['jerk']:.4f}"
        )
    
    debug_label.text = reward_text

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
    debug_label.draw()
    rl_label.draw()


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()

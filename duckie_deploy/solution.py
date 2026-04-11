#!/usr/bin/env python3
import os
from agent import DuckiebotAgent

ALGO = "sac" 
MODEL_PATH = "models/sac_Final.cleanrl_model"

agent = DuckiebotAgent(
    model_path=MODEL_PATH, 
    algo_type=ALGO, 
    grayscale=True, 
    frame_stack=4
)

def solve(context, observation):
    """
    'observation' is a BGR image from the physical camera.
    """
    action = agent.get_action(observation)
    
    u_l, u_r = agent.postprocess_kinematics(action)
    
    context.write('wheels', {'motor_left': u_l, 'motor_right': u_r})
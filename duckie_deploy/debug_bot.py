import socket
import pickle
import struct
import cv2
import numpy as np

LAPTOP_IP = "192.168.1.110" 
PORT = 8089

# Global socket to keep connection alive across steps
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    client_socket.connect((LAPTOP_IP, PORT))
except Exception as e:
    print(f"Could not connect to laptop: {e}")

def run_remote_debug(agent, context, observation):
    """
    Called by solution.py solve() function.
    """
    processed_obs = agent.preprocess(observation)
    
    view_img = processed_obs.squeeze() 
    
    view_img = view_img.astype(np.uint8) 

    action = agent.get_action(observation)
    wheel_cmds = agent.postprocess_kinematics(action)

    data = {
        "image": view_img, 
        "action": action.tolist(),
        "motors": wheel_cmds
    }
    
    # Send via socket...
    try:
        msg = pickle.dumps(data)
        client_socket.sendall(struct.pack("Q", len(msg)) + msg)
    except Exception as e:
        print(f"Send failed: {e}")
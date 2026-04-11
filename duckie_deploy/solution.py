#!/usr/bin/env python3
import os
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped
from agent import DuckiebotAgent
from debug_bot import run_remote_debug

class RLNode:
    def __init__(self):
        rospy.init_node('rl_agent_node')
        self.veh = os.environ.get('VEHICLE_NAME', 'duckie1nav')
        
        # Initialize Agent
        self.agent = DuckiebotAgent(
            model_path="models/sac_Final.cleanrl_model", 
            algo_type="sac"
        )
        
        # Publisher for the wheels
        self.wheel_pub = rospy.Publisher(f"/{self.veh}/wheels_driver_node/wheels_cmd", WheelsCmdStamped, queue_size=1)
        
        # Subscriber for the camera
        self.sub = rospy.Subscriber(f"/{self.veh}/camera_node/image/compressed", CompressedImage, self.callback, queue_size=1, buff_size=2**24)
        
        rospy.loginfo("Node started. Listening for camera frames...")
        rospy.on_shutdown(self.emergency_stop)

    def callback(self, msg):
        # Decode image
        np_arr = np.frombuffer(msg.data, np.uint8)
        obs = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Use your debug script to send data to laptop
        # We pass 'self' as the context
        run_remote_debug(self.agent, self, obs)

    def write(self, topic, data):
        """Matches the 'context.write' expected by your debug_bot.py"""
        if topic == 'wheels':
            pass
            
    def emergency_stop(self):
        rospy.loginfo("Shutting down... sending stop command to wheels.")
        stop_msg = WheelsCmdStamped()
        stop_msg.vel_left = 0.0
        stop_msg.vel_right = 0.0
        self.wheel_pub.publish(stop_msg)

if __name__ == '__main__':
    node = RLNode()
    rospy.spin()
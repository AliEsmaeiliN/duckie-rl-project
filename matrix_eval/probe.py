#!/usr/bin/env python3
"""
Connectivity check: verifies the engine is reachable and prints what each channel
actually sends (camera shape, pose message, delta_t, map layers), then nudges the
wheels so you can see the robot move. Run this once before evaluate.py.

    python matrix_eval/probe.py --robot map_0/vehicle_0
"""
import json
import math
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import tyro
from duckietown.sdk.robots.duckiebot import DB21J

from matrix_io import parse_pose


@dataclass
class Args:
    robot: str = "map_0/vehicle_0"
    host: str = "127.0.0.1"
    port: Optional[int] = None
    timeout: float = 5.0
    drive: bool = True
    """send a short forward PWM pulse"""
    save_frame: str = "matrix_eval/probe_frame.png"
    delta_t: bool = False
    """also read the delta_t channel (only exists when the engine runs with --gym)"""


def show(label, msg, limit=600):
    text = json.dumps(msg, default=str) if not hasattr(msg, "shape") else f"array{msg.shape} {msg.dtype}"
    print(f"\n=== {label} ===\n{text[:limit]}{' ...' if len(text) > limit else ''}")


def main(a: Args):
    bot = DB21J(a.robot, host=a.host, port=a.port, simulated=True)
    print(bot)

    cam = bot.camera
    cam.start()
    img = cam.capture(block=True, timeout=a.timeout)
    if img is None:
        print("\n!! No camera frame. Is the engine running and the robot name right?")
    else:
        show("camera (BGR)", img)
        cv2.imwrite(a.save_frame, img)
        print(f"saved -> {a.save_frame}")

    channels = [("pose", bot.pose), ("map/frames", bot.map_frames),
                ("map/tiles", bot.map_tiles), ("map/tile_maps", bot.map_tile_info)]
    if a.delta_t:
        channels.insert(1, ("delta_t", bot.delta_t))
    for label, drv in channels:
        drv.start()
        msg = drv.capture(block=True, timeout=a.timeout)
        show(label, msg if msg is not None else "<nothing received>")
        if label == "pose":
            print(f"parsed (x, y, yaw) -> {parse_pose(msg)}")
        drv.stop()

    if a.drive:
        bot.pose.start()
        motors = bot.motors
        motors.start()
        before = parse_pose(bot.pose.capture(block=True, timeout=a.timeout))
        for _ in range(20):
            motors.set_pwm(0.3, 0.3)
            time.sleep(0.05)
        motors.set_pwm(0.0, 0.0)
        time.sleep(0.3)
        after = parse_pose(bot.pose.latest)
        print(f"\npose before drive: {before}\npose after drive:  {after}")
        if before and after:
            dx, dy = after[0] - before[0], after[1] - before[1]
            d = math.hypot(dx, dy)
            print(f"moved {d:.3f} m, motion heading {math.degrees(math.atan2(dy, dx)):+.1f} deg, "
                  f"reported yaw {math.degrees(after[2]):+.1f} deg")
            print("yaw should match the motion heading (both CCW from +x)")
        motors.stop()
        bot.pose.stop()
    cam.stop()


if __name__ == "__main__":
    main(tyro.cli(Args))

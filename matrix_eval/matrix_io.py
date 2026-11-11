"""
Thin layer over duckietown-sdk for talking to a Duckiematrix engine:
camera, motors, reset, ground-truth pose and lane-relative position.
"""
import logging
import math
import time
from typing import Optional, Tuple

from duckietown.sdk.robots.duckiebot import DB21J # pyright: ignore[reportMissingImports]

from lane_geometry import LaneMap

logging.getLogger("dtps").setLevel(logging.WARNING)
logging.getLogger("dtps_http").setLevel(logging.WARNING)


def _num(d, *keys):
    for k in keys:
        if isinstance(d, dict) and k in d and isinstance(d[k], (int, float)):
            return float(d[k])
    return None


def _quat_yaw(q: dict) -> Optional[float]:
    w, x, y, z = (_num(q, k) for k in ("w", "x", "y", "z"))
    if None in (w, x, y, z):
        return None
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def parse_pose(msg) -> Optional[Tuple[float, float, float]]:
    """
    Best-effort (x, y, yaw) from a pose message. Handles flat {x, y, yaw},
    nested {position, orientation(quaternion)} and a 'data'/'pose' envelope.
    Run probe.py to see what your engine actually sends.
    """
    if not isinstance(msg, dict):
        return None
    for key in ("data", "pose"):
        if key in msg and isinstance(msg[key], dict):
            inner = parse_pose(msg[key])
            if inner is not None:
                return inner
    x, y, yaw = _num(msg, "x"), _num(msg, "y"), _num(msg, "yaw", "theta")
    if x is not None and y is not None and yaw is not None:
        return x, y, yaw
    # Duckiematrix sends {"position": {x, y, z}, "rotation": {w, x, y, z}} (z-up, meters)
    pos, ori = msg.get("position"), msg.get("rotation") or msg.get("orientation")
    if isinstance(pos, dict) and isinstance(ori, dict):
        x, y = _num(pos, "x"), _num(pos, "y")
        yaw = _num(ori, "yaw") if _num(ori, "yaw") is not None else _quat_yaw(ori)
        if None not in (x, y, yaw):
            return x, y, yaw
    return None


class MatrixRobot:
    def __init__(self, name: str = "map_0/vehicle_0", host: str = "127.0.0.1",
                 port: Optional[int] = None, use_pose: bool = True, pose_units: str = "meters",
                 lane_offset: float = 0.2):
        self.robot = DB21J(name, host=host, port=port, simulated=True)
        self.use_pose = use_pose
        self.pose_units = pose_units  # "meters" | "tiles"
        self.lane_offset = lane_offset
        self.lane_map: Optional[LaneMap] = None
        self._started = []

    # ---- lifecycle ----
    def start(self, map_timeout: float = 10.0):
        for comp in (self.robot.camera, self.robot.motors, self.robot.reset_flag):
            comp.start()
            self._started.append(comp)
        if self.use_pose:
            self.robot.pose.start()
            self._started.append(self.robot.pose)
            self._load_map(map_timeout)

    def stop(self):
        for comp in reversed(self._started):
            try:
                comp.stop()
            except Exception:
                pass
        self._started.clear()

    def _load_map(self, timeout: float):
        layers = {}
        for key, drv in (("frames", self.robot.map_frames), ("tiles", self.robot.map_tiles),
                         ("tile_maps", self.robot.map_tile_info)):
            drv.start()
            msg = drv.capture(block=True, timeout=timeout)
            drv.stop()
            if msg is None:
                print(f"[matrix] map layer '{key}' not received within {timeout}s: lane metrics disabled")
                return
            layers[key] = msg
        try:
            self.lane_map = LaneMap(**layers, lane_offset=self.lane_offset)
            print(f"[matrix] map loaded: {self.lane_map.n_drivable} drivable tiles, "
                  f"tile size {self.lane_map.ts} m")
        except Exception as e:
            print(f"[matrix] could not interpret map layers ({e!r}): lane metrics disabled")

    # ---- I/O ----
    def frame(self, timeout: float = 2.0):
        """Block until a frame newer than the last one returned arrives (BGR)."""
        return self.robot.camera.capture(block=True, timeout=timeout)

    def drive(self, left: float, right: float):
        self.robot.motors.set_pwm(left, right)

    def halt(self, repeats: int = 5):
        for _ in range(repeats):
            self.drive(0.0, 0.0)
            time.sleep(1 / 30)

    def _tick(self):
        """Let the simulation advance a little (realtime: wait; gym mode overrides this to step)."""
        time.sleep(1 / 30)

    def reset(self, settle: float = 1.0):
        """Ask the engine to respawn the robot, then wait for it to settle."""
        self.halt()
        self.robot.reset_flag.set_reset(True)
        time.sleep(0.2)
        self.robot.reset_flag.set_reset(False)
        time.sleep(settle)

    def u_turn(self, speed: float = 0.35, wheel_dist: float = 0.102, tol_deg: float = 3.0,
               timeout: float = 10.0, settle: float = 0.5) -> Optional[Tuple[float, float, float]]:
        """
        Forward U-turn into the opposite lane, closed-loop on the ground-truth heading.
        The arc radius is half the distance to the other lane's centre, so the robot ends up
        there facing the other way. Returns the final pose, or None without pose/map.
        """
        pose = self.world_pose()
        if pose is None or self.lane_map is None:
            return None
        lane = self.lane_position(pose)
        offset = lane[0] if lane else 0.0  # + = left of the lane centre, i.e. closer to the other lane
        radius = max(self.lane_offset * self.lane_map.ts - offset, 0.6 * wheel_dist)
        ratio = (radius - wheel_dist / 2) / (radius + wheel_dist / 2)  # left/right wheel speed for a left arc
        target = pose[2] + math.pi
        t0, ticks = time.perf_counter(), 0
        while time.perf_counter() - t0 < timeout and ticks < 30 * timeout:
            ticks += 1
            p = self.world_pose()
            err = (target - p[2] + math.pi) % (2 * math.pi) - math.pi
            if abs(err) < math.radians(tol_deg):
                break
            u = speed if abs(err) > math.radians(25) else 0.4 * speed  # slow down to limit overshoot
            self.drive(u * ratio, u)
            self._tick()
        self.halt()
        time.sleep(settle)
        return self.world_pose()

    # ---- ground truth ----
    def raw_pose(self):
        return self.robot.pose.latest if self.use_pose else None

    def world_pose(self) -> Optional[Tuple[float, float, float]]:
        """(x, y, yaw) in meters / rad (z-up, yaw CCW from +x), or None if unavailable."""
        p = parse_pose(self.raw_pose())
        if p is None:
            return None
        x, y, yaw = p
        if self.pose_units == "tiles" and self.lane_map:
            x, y = x * self.lane_map.ts, y * self.lane_map.ts
        return x, y, yaw

    def travel_direction(self, pose) -> Optional[str]:
        if self.lane_map is None or pose is None:
            return None
        return self.lane_map.travel_direction(*pose)

    def lane_position(self, pose):
        """Signed lateral offset (m, + = left) and heading error (rad) w.r.t. the right lane,
        or None when off the drivable tiles."""
        if self.lane_map is None or pose is None:
            return None
        return self.lane_map.lane_position(*pose)

"""
Lockstep ("gym mode") client for the Duckiematrix engine.

Start the engine with:  dts matrix run --standalone --no-pull --map <dir> --gym --delta-t 0.1

Protocol (read from the engine source, duckiematrix_engine/engine.py and world_side.py):
  1. the engine publishes one WorldInput on  robot/gym/in:
         {header, session_id, entities: {<robot>: {pose, twist, compressed_image, camera, ...}}}
  2. it then waits, with no timeout, for one WorldOutput on  robot/gym/out:
         {header, session_id, entities: {<robot>: {differential_pwm: {left, right},
                                                   state_reset_flag: {data}, car_lights}}}
  3. it applies the commands, advances physics by delta_t, renders, and goes back to 1.

Simulation time therefore only advances when we send an action, as fast as the renderer
and the round trip allow. In this mode the per-sensor topics used by MatrixRobot
(sensor/camera, state/pose, actuator/wheels, ...) do not exist.

The engine's duckietown-messages (0.1.x) has WorldInput/WorldOutput classes, but
duckietown-sdk pins duckietown-messages < 0.1, so the messages are built as plain CBOR dicts
here, in the same {header, fields...} layout BaseMessage.to_rawdata() produces.
"""
import asyncio
import threading
import time
import urllib.request
from typing import Dict, Optional, Tuple

import cbor2
import numpy as np
from dtps_http import RawData
from duckietown.sdk.middleware.dtps.base import DTPS
from duckietown.sdk.utils.jpeg import JPEG

from matrix_io import MatrixRobot, parse_pose


def _header() -> dict:
    return {"version": "1.0", "frame": None, "txt": None, "timestamp": time.time()}


class GymSession:
    """One connection to a gym-mode engine, shared by every robot on it."""

    _sessions: Dict[Tuple[str, int], "GymSession"] = {}

    @classmethod
    def get(cls, host: str, port: int) -> "GymSession":
        if (host, port) not in cls._sessions:
            cls._sessions[(host, port)] = GymSession(host, port)
        return cls._sessions[(host, port)]

    def __init__(self, host: str, port: int):
        self.host, self.port = host, port
        self._conn = None
        self._pub = None
        self._latest: Optional[dict] = None
        self._event = threading.Event()
        self.commands: Dict[str, Tuple[float, float]] = {}  # last command per robot
        self.started = False
        self.steps = 0

    # ---- connection ----
    async def _on_input(self, rd: RawData):
        self._latest = rd.get_as_native_object()
        self._event.set()

    async def _setup(self):
        await self._conn.context.navigate("robot", "gym", "in").subscribe(self._on_input)
        return await self._conn.context.navigate("robot", "gym", "out").publisher()

    def start(self, timeout: float = 10.0):
        if self.started:
            return
        self._conn = DTPS.get_connector(self.host, self.port)
        self._pub = asyncio.run_coroutine_threadsafe(self._setup(), DTPS._loop).result(timeout)
        # the engine published the current session before we subscribed: read it directly
        if self._latest is None and not self._event.wait(1.0):
            raw = urllib.request.urlopen(f"http://{self.host}:{self.port}/robot/gym/in/", timeout=timeout).read()
            self._latest = cbor2.loads(raw)
        self._event.clear()
        self.started = True
        dt = self.delta_time()
        print(f"[gym] connected: session {self._latest.get('session_id')}, delta_t = {dt} s per step")

    def delta_time(self) -> Optional[float]:
        try:
            raw = urllib.request.urlopen(f"http://{self.host}:{self.port}/map/delta_time/", timeout=3).read()
            return float(cbor2.loads(raw)["data"])
        except Exception:
            return None

    # ---- stepping ----
    @property
    def latest(self) -> dict:
        return self._latest

    def step(self, resets: Optional[set] = None, timeout: float = 10.0) -> dict:
        """Send the current commands (plus optional resets) and wait for the next WorldInput."""
        resets = resets or set()
        sid = self._latest.get("session_id")
        entities = {}
        for robot in set(self.commands) | resets:
            left, right = self.commands.get(robot, (0.0, 0.0))
            ent = {"header": _header(),
                   "differential_pwm": {"header": _header(), "left": float(left), "right": float(right)}}
            if robot in resets:
                ent["state_reset_flag"] = {"header": _header(), "data": True}
            entities[robot] = ent
        out = {"header": _header(), "session_id": sid, "entities": entities}
        self._event.clear()
        asyncio.run_coroutine_threadsafe(
            self._pub.publish(RawData.cbor_from_native_object(out)), DTPS._loop).result(timeout)
        deadline = time.perf_counter() + timeout
        while True:  # wait for a WorldInput from a newer session
            if not self._event.wait(max(0.0, deadline - time.perf_counter())):
                raise TimeoutError(f"no WorldInput after session {sid} within {timeout}s "
                                   "(is the engine running with --gym?)")
            self._event.clear()
            if self._latest.get("session_id") != sid:
                break
        self.steps += 1
        return self._latest


class GymMatrixRobot(MatrixRobot):
    """MatrixRobot for a gym-mode engine: each frame() call is one delta_t simulation step."""

    def __init__(self, name: str = "map_0/vehicle_0", host: str = "127.0.0.1",
                 port: Optional[int] = None, use_pose: bool = True, pose_units: str = "meters",
                 lane_offset: float = 0.2):
        super().__init__(name, host=host, port=port, use_pose=use_pose, pose_units=pose_units,
                         lane_offset=lane_offset)
        self.name = name
        self.session = GymSession.get(host, port or 7501)
        self._pending = False  # a command was set since the last step
        self._entity: Optional[dict] = None

    # ---- lifecycle ----
    def start(self, map_timeout: float = 10.0):
        self.session.start()
        self.session.commands.setdefault(self.name, (0.0, 0.0))
        self._entity = self._entity_from(self.session.latest)
        if self.use_pose:  # the map layers are published in gym mode too
            self._load_map(map_timeout)

    def stop(self):
        pass  # the engine keeps waiting for the next action; nothing to unsubscribe per robot

    def _entity_from(self, msg: dict) -> dict:
        ents = (msg or {}).get("entities") or {}
        if self.name not in ents:
            raise KeyError(f"robot '{self.name}' not in the engine's WorldInput (have: {list(ents)})")
        return ents[self.name]

    def _advance(self, resets=None):
        self._entity = self._entity_from(self.session.step(resets))
        self._pending = False

    # ---- I/O ----
    def frame(self, timeout: float = 2.0):
        """Apply the pending command for one delta_t and return the new camera frame (BGR)."""
        if self._pending:
            self._advance()
        img = (self._entity or {}).get("compressed_image") or {}
        data = img.get("data")
        return JPEG.decode(bytes(data)) if data else None

    def drive(self, left: float, right: float):
        self.session.commands[self.name] = (left, right)
        self._pending = True

    def halt(self, repeats: int = 1):
        self.drive(0.0, 0.0)
        for _ in range(repeats):
            self._advance()

    def reset(self, settle: float = 1.0):
        """Respawn via the reset flag, then step with zero PWM for `settle` seconds of sim time."""
        self.session.commands[self.name] = (0.0, 0.0)
        self._advance(resets={self.name})
        dt = self.session.delta_time() or 0.1
        self.halt(repeats=max(1, int(round(settle / dt))))

    def _tick(self):
        self._advance()

    # ---- ground truth ----
    def raw_pose(self):
        return (self._entity or {}).get("pose") if self.use_pose else None

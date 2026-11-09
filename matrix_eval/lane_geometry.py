"""
Right-lane geometry for Duckiematrix maps, built from the engine's map layers.

Why not duckietown.sdk.utils.lane_position: that helper places tile curves at the tile
corner instead of its centre and uses gym-duckietown's heading convention, so on a live
engine it reports ~64 deg heading error for a robot driving straight down a road.

Conventions (checked against the sandbox map and logged trajectories):
  - world frame is z-up: (x, y) in meters, yaw CCW from +x (same as the robot pose)
  - tile (i, j) covers [i*ts, (i+1)*ts) x [j*ts, (j+1)*ts); its frame pose is (x=i, y=j, yaw)
  - tile types at yaw 0:  straight = N-S road,  curve = joins N and W,  3way = open N, E, W
    (a tile's yaw rotates these CCW)
  - right-hand traffic; lane centre is `lane_offset * ts` right of the road centre
"""
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

# outward normals of the tile sides in the tile frame
SIDES = {"N": np.array([0.0, 1.0]), "E": np.array([1.0, 0.0]),
         "S": np.array([0.0, -1.0]), "W": np.array([-1.0, 0.0])}
OPEN_SIDES = {"straight": ("N", "S"), "curve": ("N", "W"), "3way": ("N", "E", "W"),
              "4way": ("N", "E", "S", "W")}


def _right(d: np.ndarray) -> np.ndarray:
    return np.array([d[1], -d[0]])


def _rot(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _lane_path(enter: str, leave: str, half: float, offset: float, n: int = 24) -> np.ndarray:
    """Points along the right lane from side `enter` to side `leave`, in the tile frame."""
    na, nb = SIDES[enter], SIDES[leave]
    p0 = half * na + offset * _right(-na)
    p2 = half * nb + offset * _right(nb)
    t = np.linspace(0.0, 1.0, n)[:, None]
    if np.allclose(na, -nb):  # straight through
        return (1 - t) * p0 + t * p2
    c = np.dot(p0, nb) * nb + np.dot(p2, na) * na  # corner of the two lane lines
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p2  # quadratic Bezier ~ quarter arc


class LaneMap:
    def __init__(self, frames: dict, tiles: dict, tile_maps: dict, lane_offset: float = 0.2):
        frames, tiles = frames.get("data", frames), tiles.get("data", tiles)
        info = tile_maps.get("data", tile_maps)
        self.ts = float(next(iter(info.values()))["tile_size"]["x"])
        half, off = self.ts / 2, lane_offset * self.ts

        # per drivable tile: list of (points (N,2), tangents (N,2)) in world coordinates
        self.paths: Dict[Tuple[int, int], List[Tuple[np.ndarray, np.ndarray]]] = {}
        for name, t in tiles.items():
            kind = t.get("type")
            if kind not in OPEN_SIDES:
                continue
            i, j = (int(v) for v in name.split("tile_")[-1].split("_"))
            yaw = float(frames[name]["pose"]["yaw"])
            R, centre = _rot(yaw), (np.array([i, j]) + 0.5) * self.ts
            sides = OPEN_SIDES[kind]
            lanes = []
            for a in sides:
                for b in sides:
                    if a == b:
                        continue
                    pts = _lane_path(a, b, half, off) @ R.T + centre
                    tan = np.gradient(pts, axis=0)
                    tan /= np.linalg.norm(tan, axis=1, keepdims=True)
                    lanes.append((pts, tan))
            self.paths[(i, j)] = lanes

    @property
    def n_drivable(self) -> int:
        return len(self.paths)

    @property
    def centre(self) -> np.ndarray:
        """Mean of the drivable tile centres (the middle of a loop)."""
        return (np.array(list(self.paths.keys()), dtype=float) + 0.5).mean(axis=0) * self.ts

    def travel_direction(self, x: float, y: float, yaw: float) -> str:
        """'ccw' or 'cw' around the map centre, from a pose."""
        r = np.array([x, y]) - self.centre
        return "ccw" if r[0] * math.sin(yaw) - r[1] * math.cos(yaw) > 0 else "cw"

    def tile_of(self, x: float, y: float) -> Tuple[int, int]:
        return math.floor(x / self.ts), math.floor(y / self.ts)

    def lane_position(self, x: float, y: float, yaw: float) -> Optional[Tuple[float, float]]:
        """
        (signed offset m, heading error rad) w.r.t. the closest right lane going the robot's way.
        Offset > 0 means left of the lane centre; heading error > 0 means pointing left of the lane.
        None when the robot is not on a drivable tile.
        """
        lanes = self.paths.get(self.tile_of(x, y))
        if not lanes:
            return None
        p, h = np.array([x, y]), np.array([math.cos(yaw), math.sin(yaw)])
        best = None
        for pts, tan in lanes:
            k = int(np.argmin(np.sum((pts - p) ** 2, axis=1)))
            if np.dot(tan[k], h) < 0:  # lane runs the other way
                continue
            d = p - pts[k]
            signed = float(tan[k][0] * d[1] - tan[k][1] * d[0])  # cross(tangent, d): + = left
            if best is None or abs(signed) < abs(best[0]):
                best = (signed, _wrap(yaw - math.atan2(tan[k][1], tan[k][0])))
        if best is None:  # facing backwards on this tile: report against the nearest lane anyway
            pts, tan = min(lanes, key=lambda l: np.min(np.sum((l[0] - p) ** 2, axis=1)))
            k = int(np.argmin(np.sum((pts - p) ** 2, axis=1)))
            d = p - pts[k]
            best = (float(tan[k][0] * d[1] - tan[k][1] * d[0]),
                    _wrap(yaw - math.atan2(tan[k][1], tan[k][0])))
        return best

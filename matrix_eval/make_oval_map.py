#!/usr/bin/env python3
"""
Build Duckiematrix versions of the gym-duckietown `oval_loop` training map
(src/gym_duckietown/maps/oval_loop.yaml: a 4x3-tile loop, 0.585 m tiles) from the
template in matrix_eval/loop_map (a 3x3 Duckietown loop).

Writes one map directory per driving direction, since the engine respawns the robot
at the pose stored in frames.yaml:

    matrix_eval/maps/oval_loop_ccw/   robot starts heading +x on the bottom straight
    matrix_eval/maps/oval_loop_cw/    robot starts heading -x on the bottom straight

    python matrix_eval/make_oval_map.py                    # lens distortion off (matches gym training)
    python matrix_eval/make_oval_map.py --lens-distortion  # keep the fisheye render (closer to the real bot)

Tile conventions (Duckiematrix, yaw 0): straight = N-S road, curve = joins N and W.
Tile (i, j) has frame pose x=i, y=j in tile units; x grows east, y grows north.
"""
import math
import os
import shutil
from dataclasses import dataclass

import tyro
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

W, H = 4, 3  # drivable loop size in tiles (oval_loop without its asphalt border)
PI = math.pi

# (type, yaw) per tile, following the template's yaw values
CORNERS = {
    (0, 0): ("curve", 3 * PI / 2),   # joins E and N
    (W - 1, 0): ("curve", 0.0),      # joins N and W
    (W - 1, H - 1): ("curve", PI / 2),  # joins W and S
    (0, H - 1): ("curve", PI),       # joins S and E
}


def oval_tiles():
    tiles = {}
    for i in range(W):
        for j in range(H):
            if (i, j) in CORNERS:
                tiles[(i, j)] = CORNERS[(i, j)]
            elif j in (0, H - 1):
                tiles[(i, j)] = ("straight", 3 * PI / 2)  # E-W road
            elif i in (0, W - 1):
                tiles[(i, j)] = ("straight", 0.0)         # N-S road
            else:
                tiles[(i, j)] = ("floor", 0.0)
    return tiles


@dataclass
class Args:
    template: str = os.path.join(HERE, "loop_map")
    out_dir: str = os.path.join(HERE, "maps")
    lens_distortion: bool = False
    """keep the renderer's fisheye lens model (gym training used an undistorted pinhole camera)"""
    spawn_x_tiles: float = 2.0
    """spawn position along the bottom straight, in tiles from the west edge"""
    lane_y: float = 0.185
    """spawn distance (m) from the bottom map edge for the CCW lane, as in the template map"""


def load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def dump(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def build(a: Args, direction: str):
    dst = os.path.join(a.out_dir, f"oval_loop_{direction}")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(a.template, dst)

    ts = load(os.path.join(dst, "tile_maps.yaml"))["tile_maps"]["map_0"]["tile_size"]["x"]
    tiles = oval_tiles()
    template_frames = load(os.path.join(dst, "frames.yaml"))["frames"]

    frames = {"map_0": template_frames["map_0"]}
    tile_types = {}
    for (i, j), (kind, yaw) in sorted(tiles.items()):
        name = f"map_0/tile_{i}_{j}"
        tile_types[name] = {"type": kind}
        frames[name] = {"relative_to": None, "unit": "tiles",
                        "pose": {"x": float(i), "y": float(j), "z": 0.0,
                                 "roll": 0.0, "pitch": 0.0, "yaw": float(yaw)}}

    x = a.spawn_x_tiles * ts
    y, yaw = (a.lane_y, 0.0) if direction == "ccw" else (ts - a.lane_y, PI)
    for name, fr in template_frames.items():  # vehicles and anything attached to them
        if name.startswith("map_0/vehicle_0") and "/" not in name[len("map_0/vehicle_0"):]:
            frames[name] = {**fr, "pose": {**fr["pose"], "x": x, "y": y, "yaw": yaw}}
        elif not name.startswith("map_0/tile_") and name != "map_0":
            frames[name] = fr

    dump(os.path.join(dst, "tiles.yaml"), {"version": 1.0, "tiles": tile_types})
    dump(os.path.join(dst, "frames.yaml"), {"version": 1.0, "frames": frames})

    rc_path = os.path.join(dst, "rendering_configuration.yaml")
    if os.path.exists(rc_path):
        rc = load(rc_path)
        for cam in rc.get("rendering_configuration", {}).values():
            if "lens_distortion" in cam:
                cam["lens_distortion"] = a.lens_distortion
        dump(rc_path, rc)

    print(f"wrote {dst}  (spawn x={x:.3f} y={y:.3f} yaw={yaw:.3f}, lens_distortion={a.lens_distortion})")
    return dst


def preview(tiles):
    sym = {("curve",): "+", ("straight", 0.0): "|", ("straight", 3 * PI / 2): "-", ("floor",): "."}
    for j in reversed(range(H)):
        row = ""
        for i in range(W):
            kind, yaw = tiles[(i, j)]
            row += sym.get((kind, yaw), sym.get((kind,), "?"))
        print("   " + row)


if __name__ == "__main__":
    args = tyro.cli(Args)
    preview(oval_tiles())
    for d in ("ccw", "cw"):
        build(args, d)

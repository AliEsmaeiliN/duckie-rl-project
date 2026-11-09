#!/usr/bin/env python3
"""
Build Duckiematrix versions of the gym-duckietown `oval_loop` training map
(src/gym_duckietown/maps/oval_loop.yaml: a 4x3-tile loop, 0.585 m tiles) from the
template in matrix_eval/loop_map (a 3x3 Duckietown loop).

The engine respawns each robot at the pose stored in frames.yaml, so the driving
direction is fixed per robot:

    matrix_eval/maps/oval_loop_ccw/   vehicle_0 heading +x on the bottom straight
    matrix_eval/maps/oval_loop_cw/    vehicle_0 heading -x on the bottom straight
    matrix_eval/maps/oval_loop_both/  vehicle_0 CCW (bottom) + vehicle_1 CW (top straight), with --two-robots
                                      (the idle robot stays parked in the other lane,
                                      and the active one passes it once per lap)

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
    two_robots: bool = False
    """also build oval_loop_both (vehicle_0 CCW + vehicle_1 CW). Usually not needed:
    evaluate.py --alternate-direction U-turns one robot instead"""
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


def spawn_pose(a: Args, ts: float, direction: str, top: bool = False):
    """Right-lane spawn on the bottom (or top) straight, heading `direction` around the loop."""
    x = a.spawn_x_tiles * ts
    if not top:
        return (x, a.lane_y, 0.0) if direction == "ccw" else (x, ts - a.lane_y, PI)
    y0 = (H - 1) * ts  # top straight row
    return (x, y0 + ts - a.lane_y, PI) if direction == "ccw" else (x, y0 + a.lane_y, 0.0)


def is_vehicle_key(key: str, vehicle: str) -> bool:
    return key == vehicle or key.startswith(vehicle + "/")


def clone_vehicle(dst: str, src: str, new: str, tag_id_offset: int):
    """Copy every per-vehicle entry (camera, wheels, lights, tag, ...) of `src` to `new`."""
    for fname in sorted(os.listdir(dst)):
        if not fname.endswith(".yaml") or fname in ("frames.yaml", "tiles.yaml", "main.yaml"):
            continue
        path = os.path.join(dst, fname)
        doc = load(path)
        changed = False
        for section, body in doc.items():
            if not isinstance(body, dict):
                continue
            for key in list(body):
                if is_vehicle_key(key, src):
                    entry = yaml.safe_load(yaml.safe_dump(body[key]))  # deep copy
                    if section == "vehicle_tags" and isinstance(entry.get("id"), int):
                        entry["id"] += tag_id_offset
                    body[new + key[len(src):]] = entry
                    changed = True
                elif isinstance(body[key], dict) and isinstance(body[key].get("entities"), list):
                    ents = body[key]["entities"]  # renderer_assignments
                    for e in list(ents):
                        if isinstance(e, str) and is_vehicle_key(e, src):
                            ents.append(new + e[len(src):])
                            changed = True
        if changed:
            dump(path, doc)
    assets = os.path.join(dst, "assets")
    src_assets, new_assets = os.path.join(assets, src), os.path.join(assets, new)
    if os.path.isdir(src_assets):
        shutil.copytree(src_assets, new_assets, dirs_exist_ok=True)


def build(a: Args, name: str, spawns):
    """spawns: [(vehicle_name, 'ccw' | 'cw', on_top_straight), ...]; the first is the template's vehicle_0."""
    dst = os.path.join(a.out_dir, name)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(a.template, dst)

    ts = load(os.path.join(dst, "tile_maps.yaml"))["tile_maps"]["map_0"]["tile_size"]["x"]
    tiles = oval_tiles()
    template_frames = load(os.path.join(dst, "frames.yaml"))["frames"]
    base = "map_0/vehicle_0"

    frames = {"map_0": template_frames["map_0"]}
    tile_types = {}
    for (i, j), (kind, yaw) in sorted(tiles.items()):
        tname = f"map_0/tile_{i}_{j}"
        tile_types[tname] = {"type": kind}
        frames[tname] = {"relative_to": None, "unit": "tiles",
                         "pose": {"x": float(i), "y": float(j), "z": 0.0,
                                  "roll": 0.0, "pitch": 0.0, "yaw": float(yaw)}}
    for fname, fr in template_frames.items():  # anything that is neither a tile nor a vehicle
        if fname != "map_0" and not fname.startswith("map_0/tile_") and not is_vehicle_key(fname, base):
            frames[fname] = fr

    for k, (vehicle, direction, top) in enumerate(spawns):
        if vehicle != base:
            clone_vehicle(dst, base, vehicle, tag_id_offset=k)
        x, y, yaw = spawn_pose(a, ts, direction, top)
        for fname, fr in template_frames.items():
            if is_vehicle_key(fname, base):
                new_name = vehicle + fname[len(base):]
                if fname == base:
                    fr = {**fr, "pose": {**fr["pose"], "x": x, "y": y, "yaw": yaw}}
                frames[new_name] = fr
        print(f"  {vehicle}: {direction} spawn x={x:.3f} y={y:.3f} yaw={yaw:.3f}")

    dump(os.path.join(dst, "tiles.yaml"), {"version": 1.0, "tiles": tile_types})
    dump(os.path.join(dst, "frames.yaml"), {"version": 1.0, "frames": frames})

    rc_path = os.path.join(dst, "rendering_configuration.yaml")
    if os.path.exists(rc_path):
        rc = load(rc_path)
        for cam in rc.get("rendering_configuration", {}).values():
            if "lens_distortion" in cam:
                cam["lens_distortion"] = a.lens_distortion
        dump(rc_path, rc)

    print(f"wrote {dst}  (lens_distortion={a.lens_distortion})")
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
    build(args, "oval_loop_ccw", [("map_0/vehicle_0", "ccw", False)])
    build(args, "oval_loop_cw", [("map_0/vehicle_0", "cw", False)])
    if args.two_robots:  # second robot on the opposite straight, so neither starts next to the other
        build(args, "oval_loop_both", [("map_0/vehicle_0", "ccw", False), ("map_0/vehicle_1", "cw", True)])

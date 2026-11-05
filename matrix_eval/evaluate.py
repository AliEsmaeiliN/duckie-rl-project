#!/usr/bin/env python3
"""
Evaluate trained SAC/TD3 lane-following policies in the Duckiematrix.

Start the engine + renderer first (see matrix_eval/README.md), then:

    python matrix_eval/evaluate.py --models ~/workspace/rl_models/sac_vr3.cleanrl_model \
        --episodes 5 --max-steps 1000

Outputs go to matrix_eval/results/<timestamp>/:
    episodes.csv            one row per (model, episode)
    summary.json            per-model mean/std of each metric
    <model>/ep<k>.csv       per-step log (action, PWM, pose, lane offset, latency)
    <model>/trajectories.png  (when ground-truth pose is available)
    <model>/ep<k>.mp4       (with --video)
"""
import csv
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import cv2
import numpy as np
import tyro

from live_view import LiveView
from matrix_io import MatrixRobot
from policy import MatrixPolicy

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Args:
    models: List[str] = field(default_factory=list)
    """one or more .cleanrl_model checkpoints"""
    episodes: int = 5
    max_steps: int = 1000
    """policy steps per episode (at 10 Hz, 1000 steps = 100 s)"""
    control_hz: float = 10.0
    """policy rate. Training used 30 Hz sim with frame_skip=3 -> 10 Hz. Set 0 to step as fast as
    frames arrive (use this when the engine runs in --gym lockstep mode)"""

    robot: str = "map_0/vehicle_0"
    host: str = "127.0.0.1"
    port: Optional[int] = None
    """engine DTPS port (SDK default for simulated robots is 7501)"""
    frame_timeout: float = 2.0
    reset: bool = True
    """respawn the robot through the engine reset flag before each episode"""
    settle_time: float = 1.0

    use_pose: bool = True
    """subscribe to ground-truth pose and map layers for lane metrics"""
    pose_units: str = "meters"
    """meters | tiles (the Duckiematrix robot pose is in meters)"""
    lane_offset: float = 0.2
    """right-lane centre offset from the road centre, in tile sizes (gym-duckietown uses 0.2)"""
    off_lane_patience: int = 10
    """end the episode after this many consecutive steps out of a drivable lane"""

    frame_stack: int = 4
    preprocessing: bool = False
    """CLAHE + blur + contrast stretch (training flag --preprocessing)"""
    ema: bool = False
    ema_alpha: float = 0.8
    v_scale: float = 0.8
    omega_scale: float = 3.0
    gain: float = 1.0
    trim: float = 0.0

    out_dir: str = os.path.join(HERE, "results")
    tag: str = ""
    video: bool = False
    show: bool = False
    """live window: camera next to the stacked policy input, actions and lane state (q quits, p pauses)"""
    rectify: Optional[str] = None
    """undistort frames with this map dir's cameras.yaml before the crop, like the real-robot
    pipeline. Use with maps rendered with lens distortion on"""
    seed: int = 0


STEP_FIELDS = ["step", "t", "v", "omega", "pwm_l", "pwm_r", "x", "y", "yaw",
               "lane_dist", "lane_angle", "on_road", "infer_ms", "loop_ms"]


def run_episode(policy: MatrixPolicy, robot: MatrixRobot, a: Args, ep_dir: str, ep: int,
                view: Optional[LiveView] = None):
    if a.reset:
        robot.reset(settle=a.settle_time)
    policy.reset()

    writer = None
    rows = []
    prev_xy, distance, off_lane_run = None, 0.0, 0
    termination = "max_steps"
    period = 1.0 / a.control_hz if a.control_hz > 0 else 0.0
    t_start = last_print = time.perf_counter()

    try:
        for step in range(a.max_steps):
            t0 = time.perf_counter()
            img = robot.frame(timeout=a.frame_timeout)
            if img is None:
                termination = "no_frames"
                break

            obs = policy.observe(img)
            ti = time.perf_counter()
            action = policy.act(obs)
            infer_ms = (time.perf_counter() - ti) * 1e3
            pwm_l, pwm_r = policy.to_pwm(action)
            robot.drive(pwm_l, pwm_r)

            pose = robot.world_pose()
            lane = robot.lane_position(pose)
            if pose is not None:
                if prev_xy is not None:
                    distance += math.hypot(pose[0] - prev_xy[0], pose[1] - prev_xy[1])
                prev_xy = pose[:2]
                off_lane_run = 0 if lane is not None else off_lane_run + 1

            if view is not None:
                cmd = view.update(policy.last_view, obs, policy.spec.grayscale, action, (pwm_l, pwm_r),
                                  lane, (lane is not None) if pose is not None else None,
                                  f"{policy.name}  ep{ep}  step {step}/{a.max_steps}", on_pause=robot.halt)
                if cmd == "quit":
                    raise KeyboardInterrupt

            if a.video:
                if writer is None:
                    h, w = img.shape[:2]
                    writer = cv2.VideoWriter(os.path.join(ep_dir, f"ep{ep}.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), a.control_hz or 10, (w, h))
                vis = img.copy()
                txt = f"{policy.name} ep{ep} t{step} v={action[0]:.2f} w={action[1]:+.2f}"
                if lane is not None:
                    txt += f" d={lane[0]:+.3f}m"
                cv2.putText(vis, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                writer.write(vis)

            rows.append({
                "step": step, "t": time.perf_counter() - t_start,
                "v": float(action[0]), "omega": float(action[1]), "pwm_l": pwm_l, "pwm_r": pwm_r,
                "x": pose[0] if pose else None, "y": pose[1] if pose else None, "yaw": pose[2] if pose else None,
                "lane_dist": lane[0] if lane else None, "lane_angle": lane[1] if lane else None,
                "on_road": int(lane is not None) if pose is not None else None,
                "infer_ms": infer_ms, "loop_ms": (time.perf_counter() - t0) * 1e3,
            })

            if time.perf_counter() - last_print > 1.0:
                last_print = time.perf_counter()
                lane_txt = f"d={lane[0]:+.3f}m hdg={math.degrees(lane[1]):+5.1f}deg" if lane else (
                    "OFF-ROAD" if pose is not None else "no pose")
                line = (f"  ep{ep} step {step:>5}/{a.max_steps}  v={action[0]:.2f} w={action[1]:+.2f}  "
                        f"{lane_txt:<26}  dist={distance:6.2f}m  "
                        f"{(step + 1) / (time.perf_counter() - t_start):4.1f}Hz")
                print("\r" + line.ljust(100), end="", flush=True)

            if pose is not None and off_lane_run >= a.off_lane_patience:
                termination = "off_road"
                break

            if period:
                sleep = period - (time.perf_counter() - t0)
                if sleep > 0:
                    time.sleep(sleep)
    except KeyboardInterrupt:
        termination = "interrupted"
    print()

    robot.halt()
    if writer is not None:
        writer.release()

    with open(os.path.join(ep_dir, f"ep{ep}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STEP_FIELDS)
        w.writeheader()
        w.writerows(rows)

    return summarize(rows, distance, termination, a), rows


def summarize(rows, distance, termination, a: Args) -> dict:
    n = len(rows)
    acts = np.array([[r["v"], r["omega"]] for r in rows]) if rows else np.zeros((0, 2))
    dists = np.array([r["lane_dist"] for r in rows if r["lane_dist"] is not None])
    angles = np.array([r["lane_angle"] for r in rows if r["lane_angle"] is not None])
    in_lane = [r["on_road"] for r in rows if r["on_road"] is not None]
    duration = rows[-1]["t"] if rows else 0.0
    nan = float("nan")
    return {
        "steps": n,
        "termination": termination,
        "success": int(termination == "max_steps"),
        "duration_s": duration,
        "distance_m": distance if in_lane else nan,
        "mean_speed_mps": distance / duration if in_lane and duration > 0 else nan,
        "on_road_ratio": float(np.mean(in_lane)) if in_lane else nan,
        "mean_abs_lane_dist_m": float(np.mean(np.abs(dists))) if dists.size else nan,
        "rmse_lane_dist_m": float(np.sqrt(np.mean(dists ** 2))) if dists.size else nan,
        "mean_abs_heading_deg": float(np.degrees(np.mean(np.abs(angles)))) if angles.size else nan,
        "mean_v": float(acts[:, 0].mean()) if n else nan,
        "mean_abs_omega": float(np.abs(acts[:, 1]).mean()) if n else nan,
        "action_jerk": float(np.abs(np.diff(acts, axis=0)).sum(1).mean()) if n > 1 else nan,
        "mean_infer_ms": float(np.mean([r["infer_ms"] for r in rows])) if n else nan,
        "mean_loop_hz": n / duration if duration > 0 else nan,
    }


def plot_trajectories(model_dir: str, name: str, episodes_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    drew = False
    for ep, rows in enumerate(episodes_rows):
        xy = np.array([[r["x"], r["y"]] for r in rows if r["x"] is not None])
        if len(xy):
            ax.plot(xy[:, 0], xy[:, 1], lw=1.2, label=f"ep{ep}")
            ax.plot(*xy[0], "o", ms=4, color="black")
            drew = True
    if drew:
        ax.set_aspect("equal")
        ax.set_title(f"{name} — Duckiematrix trajectories")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend(fontsize=7)
        fig.savefig(os.path.join(model_dir, "trajectories.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(a: Args):
    if not a.models:
        raise SystemExit("pass at least one checkpoint with --models")
    np.random.seed(a.seed)
    stamp = datetime.now().strftime("%m%d_%H%M%S")
    run_dir = os.path.join(a.out_dir, f"{stamp}{'_' + a.tag if a.tag else ''}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(a), f, indent=2)

    robot = MatrixRobot(a.robot, host=a.host, port=a.port, use_pose=a.use_pose, pose_units=a.pose_units,
                        lane_offset=a.lane_offset)
    robot.start()
    view = LiveView() if a.show else None
    all_rows, summary, stop = [], {}, False
    try:
        for path in a.models:
            policy = MatrixPolicy(os.path.expanduser(path), frame_stack=a.frame_stack,
                                  preprocessing=a.preprocessing, ema=a.ema, ema_alpha=a.ema_alpha,
                                  v_scale=a.v_scale, omega_scale=a.omega_scale, gain=a.gain, trim=a.trim,
                                  rectify_calib=os.path.expanduser(a.rectify) if a.rectify else None)
            print("\n" + policy.describe())
            model_dir = os.path.join(run_dir, policy.name)
            os.makedirs(model_dir, exist_ok=True)

            ep_metrics, ep_rows = [], []
            for ep in range(a.episodes):
                m, rows = run_episode(policy, robot, a, model_dir, ep, view)
                ep_metrics.append(m)
                ep_rows.append(rows)
                all_rows.append({"model": policy.name, "episode": ep, **m})
                print(f"  ep{ep}: {m['termination']:<9} steps={m['steps']:<5} "
                      f"dist={m['distance_m']:.2f}m on_road={m['on_road_ratio']:.2f} "
                      f"|d|={m['mean_abs_lane_dist_m']:.3f}m |hdg|={m['mean_abs_heading_deg']:.1f}deg "
                      f"jerk={m['action_jerk']:.3f} {m['mean_loop_hz']:.1f}Hz")
                if m["termination"] == "interrupted":
                    stop = True
                    break

            plot_trajectories(model_dir, policy.name, ep_rows)
            numeric = [k for k, v in ep_metrics[0].items() if isinstance(v, (int, float))]
            summary[policy.name] = {
                "checkpoint": path,
                "meta": {k: v for k, v in policy.spec.meta.items() if isinstance(v, (str, int, float, dict))},
                **{k: {"mean": float(np.nanmean([e[k] for e in ep_metrics])),
                       "std": float(np.nanstd([e[k] for e in ep_metrics]))} for k in numeric},
            }
            if stop:
                print("\ninterrupted, writing partial results")
                break
    except KeyboardInterrupt:
        print("\ninterrupted, writing partial results")
    finally:
        robot.halt()
        robot.stop()
        if view is not None:
            view.close()

    if all_rows:
        with open(os.path.join(run_dir, "episodes.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'model':<24}{'success':>9}{'dist[m]':>10}{'on_road':>9}{'|d|[m]':>9}{'jerk':>8}")
    for name, s in summary.items():
        print(f"{name:<24}{s['success']['mean']:>9.2f}{s['distance_m']['mean']:>10.2f}"
              f"{s['on_road_ratio']['mean']:>9.2f}{s['mean_abs_lane_dist_m']['mean']:>9.3f}"
              f"{s['action_jerk']['mean']:>8.3f}")
    print(f"\nresults -> {run_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))

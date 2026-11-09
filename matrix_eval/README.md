# matrix_eval — evaluating trained policies in the Duckiematrix

Runs the SAC/TD3 actors trained in this repo (`.cleanrl_model` checkpoints) on a
simulated Duckiebot in the Duckiematrix, and logs lane-following metrics from
ground-truth pose.

The scripts don't import `gym_duckietown` or the training code. They have their own
copy of the observation and action pipeline, so they run in a separate conda env and
leave the `duckie-rl` env untouched.

| file | purpose |
|---|---|
| `policy.py` | rebuilds the actor from a checkpoint and reproduces the training pipeline |
| `matrix_io.py` | camera, motors, reset and pose via `duckietown-sdk` |
| `lane_geometry.py` | right-lane centre lines built from the engine's map layers |
| `probe.py` | checks the connection and prints the raw message formats |
| `make_oval_map.py` | builds `maps/oval_loop_{ccw,cw,both}` (the training track) from `loop_map/` |
| `live_view.py` | the `--show` window |
| `evaluate.py` | runs the episodes and writes CSV, JSON and plot results |
| `environment.yaml` | the `duckie-matrix` conda env |

## 1. Environment

```bash
conda env create -f matrix_eval/environment.yaml   # already created on this machine
conda activate duckie-matrix
```

## 2. Build the training-track maps

```bash
python matrix_eval/make_oval_map.py                    # lens distortion off: matches gym training
python matrix_eval/make_oval_map.py --lens-distortion  # fisheye render, closer to the real bot
```

This turns the 3x3 template loop in `loop_map/` into the 4x3 `oval_loop` training
track, as `maps/oval_loop_ccw` and `maps/oval_loop_cw`. They differ only in the spawn
pose; the engine respawns the robot there on every reset.

Training used `--direction mixed`, so evaluate both directions. The easiest way is one
engine run on `oval_loop_ccw` with `evaluate.py --alternate-direction`. On odd episodes
the robot then makes a scripted forward U-turn into the opposite lane after the reset,
before the policy takes over. The turn is closed-loop on the ground-truth heading, and
its radius is half the distance between the two lanes. `oval_loop_cw` is there if you
want clockwise episodes that start straight from a spawn.

(`--two-robots` also builds `oval_loop_both`, with a second, clockwise robot as
`vehicle_1`. Select it with `--robots map_0/vehicle_0 map_0/vehicle_1`. The idle robot
stays parked in its lane, in view of the active one.)

## 3. Start the Duckiematrix

You need the dts **ente** profile. The Duckiematrix commands aren't in `daffy`.

```bash
dts profile switch ente
dts matrix run --standalone --sandbox            # engine + renderer, default sandbox map
# the training oval (both directions with evaluate.py --alternate-direction):
dts matrix run --standalone --map matrix_eval/maps/oval_loop_ccw
# lockstep (simulation-time) mode:
dts matrix run --standalone --sandbox --gym --delta-t 0.1
```

## 4. Probe once

```bash
python matrix_eval/probe.py --robot map_0/vehicle_0
```

The probe:

- saves a camera frame to `matrix_eval/probe_frame.png`
- prints the pose, `delta_t` and map-layer messages
- sends a short forward pulse and shows the pose before and after

If `parsed (x, y, yaw)` prints `None`, update `parse_pose()` in `matrix_io.py` to match
the message it printed.

## 5. Evaluate

```bash
python matrix_eval/evaluate.py \
    --models ~/workspace/rl_models/sac_vr3.cleanrl_model ~/workspace/rl_models/td3_vr2.cleanrl_model \
    --episodes 5 --max-steps 1000 --tag oval_ccw --show

# both directions on oval_loop_ccw: even episodes CCW, odd episodes U-turn then CW
python matrix_eval/evaluate.py --models runs/sac_unf_s1_0310.cleanrl_model \
    --alternate-direction --episodes 6 --tag oval_mixed --show
```

Each episode records its `direction` (`ccw`/`cw`), worked out from the spawn pose.
`summary.json` gives each model an `overall` block and a `groups` block per robot and direction. Step logs
go under `<model>/<vehicle>/`.

`--show` opens a live window. It shows the camera frame, with the crop line at 1/3,
next to the four stacked frames the policy actually receives, plus v/omega, wheel PWM,
lane offset and heading. Press `q`/`Esc` to stop (results are still written) and
`p`/space to pause (the wheels stop while paused). The renderer window shows the
robot from outside at the same time.

`--rectify <map_dir>` undistorts each frame with that map's `cameras.yaml`
before the crop, like the real-robot pipeline. Use it for maps built with
`--lens-distortion`.

Results go to `matrix_eval/results/<timestamp>_<tag>/` (this folder is git-ignored):

- `episodes.csv`: one row per model, robot and episode, including `direction`
- `summary.json`: mean and standard deviation per model, overall and per robot
- `<model>/<vehicle>/ep<k>.csv`: per-step log
- `<model>/<vehicle>/trajectories.png`: driven trajectories
- `<model>/<vehicle>/ep<k>.mp4`: video of each episode, with `--video`

### Matching training settings

The network shape is read from the checkpoint:

- algorithm: SAC or TD3
- input: grayscale or RGB
- resolution: 84 or 42 px

Wrapper settings are **not** stored in the checkpoint. Pass the flags the run was
trained with:

| training flag | eval flag | default |
|---|---|---|
| `--preprocessing` | `--preprocessing` | off |
| `--ema` | `--ema` (alpha 0.8) | off |
| `KinematicActionWrapper` scales | `--v-scale`, `--omega-scale` | 0.8, 3.0 |
| motor trim/gain | `--trim`, `--gain` | 0, 1 |

### Metrics

| metric | meaning |
|---|---|
| `success` | ran to `--max-steps` without leaving the lane |
| `termination` | `max_steps`, `off_road` (off the drivable tiles for `--off-lane-patience` consecutive steps), `no_frames`, or `interrupted` |
| `distance_m`, `mean_speed_mps` | from the ground-truth pose |
| `on_road_ratio` | fraction of steps on a drivable tile |
| `mean_abs_lane_dist_m`, `rmse_lane_dist_m` | lateral offset from the right-lane centre line (per step: + = left) |
| `mean_abs_heading_deg` | heading error against the lane tangent |
| `action_jerk` | mean of the summed absolute change in `[v, omega]` between steps |
| `mean_infer_ms`, `mean_loop_hz` | policy latency and the control rate actually reached |

Run with `--no-use-pose` to drive from the camera only. Lane metrics are then NaN.

## Assumptions to check on first run

- **Control rate.** Training stepped at 10 Hz (30 Hz sim, `frame_skip=3`). In realtime
  mode, `--control-hz 10` keeps that rate. In `--gym` mode, use `--control-hz 0`: the
  loop then steps once per new camera frame.
- **Camera.** Frames are BGR. They are converted to RGB and resized to 640x480, the
  training camera size, before the crop and resize.
- **Pose.** The robot pose arrives as `position` {x, y, z} in meters (z up) plus a
  `rotation` quaternion. Yaw is CCW from +x and matches the direction of motion.
- **Lane geometry.** `lane_geometry.py` replaces the SDK's `lane_position` helper, which
  puts tile curves at the tile corner and reported about 64 deg heading error for a robot
  driving straight. Tile types at yaw 0 are: straight = N-S, curve = joins N and W,
  3way = open N/E/W. These were checked against the sandbox map. Lane centre is
  `--lane-offset` x tile size (0.2, as in gym-duckietown) right of the road centre.
- **`delta_t`.** This channel exists only when the engine runs with `--gym`
  (`probe.py --delta-t`).
- **Reset.** Each episode starts by sending the engine reset flag (`True` then
  `False`). Use `--no-reset` if your engine handles respawns differently.
- **Visual gap.** The Duckiematrix renders very differently from gym-duckietown, so
  treat these results as a sim-to-sim transfer test.

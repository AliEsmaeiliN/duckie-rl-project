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
| `evaluate.py` | runs the episodes and writes CSV, JSON and plot results |
| `environment.yaml` | the `duckie-matrix` conda env |

## 1. Environment

```bash
conda env create -f matrix_eval/environment.yaml   # already created on this machine
conda activate duckie-matrix
```

## 2. Start the Duckiematrix

You need the dts **ente** profile. The Duckiematrix commands aren't in `daffy`.

```bash
dts profile switch ente
dts matrix run --standalone --sandbox            # engine + renderer, default sandbox map
# or with a custom map directory:
dts matrix run --standalone --map <path/to/map_dir>
# lockstep (simulation-time) mode:
dts matrix run --standalone --sandbox --gym --delta-t 0.1
```

## 3. Probe once

```bash
python matrix_eval/probe.py --robot map_0/vehicle_0
```

The probe:

- saves a camera frame to `matrix_eval/probe_frame.png`
- prints the pose, `delta_t` and map-layer messages
- sends a short forward pulse and shows the pose before and after

If `parsed (x, y, yaw)` prints `None`, update `parse_pose()` in `matrix_io.py` to match
the message it printed.

## 4. Evaluate

```bash
python matrix_eval/evaluate.py \
    --models ~/workspace/rl_models/sac_vr3.cleanrl_model ~/workspace/rl_models/td3_vr2.cleanrl_model \
    --episodes 5 --max-steps 1000 --tag sandbox
```

Results go to `matrix_eval/results/<timestamp>_<tag>/` (this folder is git-ignored):

- `episodes.csv`: one row per model and episode
- `summary.json`: mean and standard deviation per model
- `<model>/ep<k>.csv`: per-step log
- `<model>/trajectories.png`: driven trajectories
- `<model>/ep<k>.mp4`: video of each episode, with `--video`

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

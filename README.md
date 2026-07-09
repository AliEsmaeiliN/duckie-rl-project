
# Gym-Duckietown (Duckie-RL Fork)


Duckietown self-driving car simulator environments optimized for **Deep Reinforcement Learning** using modern **Gymnasium** APIs and the **CleanRL** ecosystem.

This repository is specifically refactored to support visual lane-navigation and advanced continuous-control algorithms (SAC, TD3) on customizable maps (such as `oval_loop`), tracking precision control metrics directly via localized wrappers.



## Introduction

**Duckie-RL** is a specialized, high-performance reinforcement learning fork of the Duckietown Universe simulator. Built entirely on Python/OpenGL (`Pyglet`), this version swaps out classical robotics components for streamlined CleanRL execution scripts, custom high-stride CNN encoders, modular perception wrappers, and advanced multi-track evaluation logging.

### Key Enhancements

* **Gymnasium API Alignment:** Fully converted wrapper architectures and environment definitions to standard `gymnasium` vector and wrapper classes.
* **Kinematic Actuation Core:** Implements physical motor dynamics (Gain/Trim/PWM constraints) directly via `KinematicActionWrapper` to match physical robot behavior.
* **ImpalaCNN Features:** Integrates high-efficiency, stride-based visual feature extraction networks optimized for fast spatial convergence.
* **Risk-Adjusted Evaluation Stack:** Automatically monitors evaluation metrics across parallel tracks (`eval_perfect` and `eval_imperfect`) tracking $\text{Mean} - \beta \cdot \text{Std}$ scores and trajectory telemetry on weights and biases.

---

## Workspace Architecture

When creating modifications or expanding core environment behavior, adhere to the following project file structure:

* **Modifying/Adding Wrappers:** Place your custom behavioral modifications (observation, action, or reward modifiers) inside their dedicated files within the `utils/wrappers/` folder (e.g., `observation_wrappers.py`, `action_wrappers.py`, `reward_wrappers.py`).
* **Environment Assembly Stack:** Connect and instantiate new wrappers within the environment compilation pipeline inside `utils/rl_env.py` (specifically under the `create_wrapped` method stack).
* **Reward Tuning Lab:** Before running full-scale training workloads, use the sandbox testing file `debug/simlab.py` to test, verify, and print your step-by-step reward function dynamics under controlled tracking conditions.

---

## Installation

### Requirements

* Python 3.10+
* Gymnasium
* NumPy
* Pyglet
* PyYAML
* PyTorch (with CUDA support)
* OpenCV (cv2)

### Installation Using Conda (Recommended)

Set up your isolated workspace containing the proper graphic linking configurations and deep learning dependencies:

```bash
git clone https://github.com/AliEsmaeiliN/duckie-rl-project.git
cd duckie-rl-project
conda env create -f environment.yaml
conda activate duckie-rl
pip install -e .

```

Ensure your python environment knows how to map execution pathways:

```bash
export PYTHONPATH="${PYTHONPATH}:`pwd`"

```

---

## Usage

### Physical Telemetry Verification

Verify your installation, CUDA visibility, and host OpenGL vendor configurations by executing the system testing script:

```bash
./run_tests.py

```

To drive the agent manually using keyboard control mapping inside the default `oval_loop` tracking environment:

```bash
./manual_control.py --env-name Duckietown

```

---

### Reinforcement Learning Training

This environment contains dedicated algorithm scripts written using CleanRL paradigms. To train an autonomous driving agent on a vectorized continuous-action setup, execute either from your terminal:

```bash
# Train a Twin Delayed DDPG (TD3) Agent
python rl/td3_continuous_action.py --seed 42 --env-id oval_loop

# Train a Soft Actor-Critic (SAC) Agent
python rl/sac_continuous_action.py --seed 42 --env-id oval_loop

```

### Policy Evaluation & Visual Tracking

To evaluate a trained checkpoint saved down locally or streamed down from WandB:

```bash
python rl/eval_sac.py --model-path runs/models/sac_latest_step.cleanrl_model --local True

```

---

## Pipeline Architecture & Design

### 1. Observations Pipeline

The environment produces raw camera tensors shaped to $(120, 160, 3)$ matching the downscaled Raspberry Pi field-of-view matrices. The wrapping pipeline modifies these parameters sequentially:

```
[Raw Frame: 120x160x3 RGB] 
         │
         ▼
[CropResizeWrapper] ───► Crops horizon / Resizes down to (84, 84, 3)
         │
         ▼
[GrayscaleWrapper]  ───► Converts to (1, 84, 84) single-channel
         │
         ▼
[FrameStackObservation]► Combines historical sequences into (4, 84, 84) Tensors

```

### 2. Actions & Kinematics Space

The model calculates a continuous control vector bounded to $[-1, 1]^2$, translating explicitly to forward velocity ($v$) and steering angular momentum ($\omega$).

```
[Actor Network Output] ──► [v, ω]
                              │
                              ▼
                  [KinematicActionWrapper]
                              │
      ┌───────────────────────┴───────────────────────┐
      ▼                                               ▼
[Left Wheel PWM (u_l)]                        [Right Wheel PWM (u_r)]

```

The `KinematicActionWrapper` transforms these choices into specific left/right wheel duty cycles ($u_l, u_r$) mapping the real-world motor coefficients ($k$), wheel distance base ($d$), and tire radius ($r$):

$$omega_r = \frac{v + 0.5 \cdot \omega \cdot d}{r}, \quad omega_l = \frac{v - 0.5 \cdot \omega \cdot d}{r}$$

### 3. Objective Matrix (Reward Function)

The standard reward uses asymmetric and lookahead tracking to penalize cross-track deviations ($e_{\text{cte}}$) and orientation mismatch relative to a local Bezier centerline curve layout:

$$\text{Reward} = w_{\text{speed}} \cdot r_{\text{speed}} + w_{\text{lane}} \cdot r_{\text{lane}} + w_{\text{heading}} \cdot r_{\text{heading}} + w_{\text{jerk}} \cdot r_{\text{jerk}}$$

* **Jerk Penalization:** Smooths physical movement by enforcing structural consistency between steps ($\| a_t - a_{t-1} \|_2$) using the `AdditiveJerkPenalty` wrapper.
* **FSM Recovery Protocols:** Dynamically inserts a `RecoveryTrainingWrapper` to allow exploratory policy reconstruction during out-of-bounds events without triggering hard termination loops.

---

## Headless Execution & Cluster Deployment

### Running Headless via Xvfb (MILA Cluster Syntax)

For cluster environments requiring a virtual X11 server instance to complete internal Pyglet drawing requests:

```bash
# Allocate cluster hardware node
sinter --mem=12000 -c2 --gres=gpu:1

# Setup virtual render targets
Xvfb :$SLURM_JOB_ID -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log &
export DISPLAY=:$SLURM_JOB_ID
export PYGLET_DEBUG_GL=False

# Launch background execution pipeline
python rl/sac_continuous_action.py --track True

```

---

## Citation

```
@misc{gym_duckietown,
  author = {Chevalier-Boisvert, Maxime and Golemo, Florian and Cao, Yanjun and Mehta, Bhairav and Paull, Liam},
  title = {Duckietown Environments for OpenAI Gym},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/duckietown/gym-duckietown}},
}

```

```

```
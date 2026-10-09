#!/usr/bin/env python3
"""
Short SAC training run to measure throughput, tracked on Weights & Biases.

Evaluation and checkpointing are disabled so only env stepping and learning are timed.
Metrics to look at in wandb (group "speedtest-<host>"):
  perf/SPS_recent              steps/s over the last 100 steps
  perf/env_step_ms             time per step spent in the simulator
  perf/policy_and_update_ms    time per step spent in the policy and gradient updates
Before --learning-starts only the simulator runs, after it learning kicks in.

Usage (from the repo root):
  python scripts/speed_test.py                      # one run, 20k steps
  python scripts/speed_test.py --runs 4             # 4 seeds in parallel, to measure total throughput
  python scripts/speed_test.py -- --grayscale False # anything after -- goes to the training script
"""
import argparse
import os
import platform
import subprocess
import sys
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algo", choices=["sac", "td3"], default="sac")
    parser.add_argument("--steps", type=int, default=20000, help="total timesteps per run")
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--runs", type=int, default=1, help="number of seeds to run in parallel")
    parser.add_argument("--seed", type=int, default=1, help="seed of the first run")
    parser.add_argument("--project", default="Duckie-RL-V2", help="wandb project")
    parser.add_argument("--entity", default=None, help="wandb entity (team or user)")
    parser.add_argument("--no-track", action="store_true", help="skip wandb, TensorBoard only")
    args, extra = parser.parse_known_args()
    extra = [a for a in extra if a != "--"]

    host = platform.node().split(".")[0]
    stamp = datetime.now().strftime("%m%d_%H%M")
    group = f"speedtest-{host}"
    log_dir = os.path.join(REPO, "logs")
    os.makedirs(log_dir, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # Keep macOS from sleeping or App-Napping the run
    prefix = ["caffeinate", "-i"] if sys.platform == "darwin" else []

    procs = []
    for i in range(args.runs):
        seed = args.seed + i
        cmd = prefix + [
            sys.executable, os.path.join(REPO, "rl", f"{args.algo}_continuous_action.py"),
            "--seed", str(seed),
            "--total-timesteps", str(args.steps),
            "--learning-starts", str(args.learning_starts),
            "--start-evaluation", str(args.steps + 1),
            "--no-save-model",
            "--wandb-group", group,
            "--run-notes", f"speed test on {host}, {args.runs} parallel run(s)",
        ]
        if not args.no_track:
            cmd += ["--track", "--wandb-project-name", args.project]
            if args.entity:
                cmd += ["--wandb-entity", args.entity]
        cmd += extra

        log_path = os.path.join(log_dir, f"speedtest_{host}_{stamp}_{args.algo}_s{seed}.log")
        print(f"[run {i}] seed={seed} log={log_path}")
        if args.runs == 1:
            # Single run: stream to the terminal and the log
            with open(log_path, "w") as log:
                p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in p.stdout:
                    sys.stdout.write(line)
                    log.write(line)
                procs.append(p)
        else:
            log = open(log_path, "w")
            procs.append(subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT))

    if args.runs > 1:
        print(f"Waiting for {args.runs} runs, follow one with: tail -f {log_dir}/speedtest_{host}_{stamp}_*.log")
    codes = [p.wait() for p in procs]

    print(f"\nDone. Exit codes: {codes}")
    print(f"Compare in wandb: project '{args.project}', group '{group}', metrics perf/*")
    sys.exit(max(codes))


if __name__ == "__main__":
    main()

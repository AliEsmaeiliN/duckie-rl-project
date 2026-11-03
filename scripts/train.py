#!/usr/bin/env python3
"""
Launch SAC and/or TD3 training runs.

The launcher only picks the algorithm(s) and seeds; every other argument is forwarded
unchanged to rl/<algo>_continuous_action.py (see utils/shared_args.py and each script's Args).
One run streams to the terminal, several runs go to logs/ (in parallel, or one after another
with --sequential).

Usage (from the repo root):
  python scripts/train.py --algo sac --track --total-timesteps 500000
  python scripts/train.py --algo sac td3 --seeds 1 2 3 --track          # 6 runs in parallel
  python scripts/train.py --algo td3 --sequential --seeds 1 2 -- --num-envs 4 --exploration-noise 0.2
  python scripts/train.py --algo sac --dry-run --reward-type pid        # print the commands only
  python scripts/train.py --algo sac --script-help                      # show the training script's arguments
"""
import argparse
import os
import platform
import subprocess
import sys
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    # allow_abbrev=False so forwarded flags (e.g. --seed) are not swallowed as abbreviations of ours
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument("--algo", nargs="+", choices=["sac", "td3"], default=["sac"], help="algorithm(s) to train")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1], help="one run per seed and algorithm")
    parser.add_argument("--sequential", action="store_true", help="run one after another instead of in parallel")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    parser.add_argument("--script-help", action="store_true", help="print the training script's --help and exit")
    args, extra = parser.parse_known_args()
    extra = [a for a in extra if a != "--"]
    if any(a == "--seed" or a.startswith("--seed=") for a in extra):
        parser.error("use --seeds instead of --seed")

    env = os.environ.copy()
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    def script(algo):
        return os.path.join(REPO, "rl", f"{algo}_continuous_action.py")

    if args.script_help:
        for algo in args.algo:
            subprocess.run([sys.executable, script(algo), "--help"], cwd=REPO, env=env)
        return

    # Keep macOS from sleeping or App-Napping the run
    prefix = ["caffeinate", "-i"] if sys.platform == "darwin" else []

    host = platform.node().split(".")[0]
    stamp = datetime.now().strftime("%m%d_%H%M")
    log_dir = os.path.join(REPO, "logs")
    jobs = []
    for algo in args.algo:
        for seed in args.seeds:
            cmd = prefix + [sys.executable, script(algo), "--seed", str(seed)] + extra
            log_path = os.path.join(log_dir, f"train_{host}_{stamp}_{algo}_s{seed}.log")
            jobs.append((f"{algo} seed={seed}", cmd, log_path))

    if args.dry_run:
        for name, cmd, _ in jobs:
            print(f"[{name}] {subprocess.list2cmdline(cmd)}")
        return

    os.makedirs(log_dir, exist_ok=True)
    stream = len(jobs) == 1 or args.sequential

    def start(name, cmd, log_path):
        print(f"[{name}] log={log_path}")
        log = open(log_path, "w")
        if not stream:
            return subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        # Stream to the terminal and the log
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line)
            log.write(line)
        p.wait()
        log.close()
        return p

    codes = []
    try:
        if stream:
            codes = [start(*job).returncode for job in jobs]
        else:
            procs = [start(*job) for job in jobs]
            print(f"Waiting for {len(procs)} runs, follow one with: tail -f {log_dir}/train_{host}_{stamp}_*.log")
            codes = [p.wait() for p in procs]
    except KeyboardInterrupt:
        print("\nInterrupted, stopping runs")
        sys.exit(130)

    print(f"\nDone. Exit codes: {dict(zip((j[0] for j in jobs), codes))}")
    sys.exit(max(codes))


if __name__ == "__main__":
    main()

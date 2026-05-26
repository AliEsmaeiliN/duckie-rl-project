#!/bin/bash
#SBATCH --job-name=ali_td3
#SBATCH --output=output/duckie_%j.out
#SBATCH -e output/duckie_%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=pgpu_most
#SBATCH --account=dei_most
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4


cd $HOME/duckie-rl-project
mkdir -p output

export DISPLAY=:$((SLURM_JOB_ID % 100 + 100))

SIF_IMAGE="$HOME/duckie_rl.sif"

export WANDB_DIR="/tmp"
export WANDB_CACHE_DIR="/tmp"

export PYGLET_HEADLESS=True
export PYGLET_DEBUG_GL=False



srun --cpu-bind=none singularity exec --nv \
    -B .:/app \
    --pwd /app \
    $SIF_IMAGE \
    xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render" \
    bash -c "export PYTHONPATH=/app:/app/src:\$PYTHONPATH && \
    python3 rl/td3_continuous_action.py \
    --seed 1 \
    --env-id unified1 \
    --total-timesteps 1500000 \
    --track \
    --version 0 \
    --buffer-size 300000 \
    --learning-starts 40000 \
    --domain-rand \
    --camera-rand \
    --dynamics-rand \
    --action-latency \
    --curriculum-randomization \
    --jerk-penalty \
    --recovery \
    --run-notes 'unified v1 with complete curriculum'"

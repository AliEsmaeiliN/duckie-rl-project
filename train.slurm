#!/bin/bash
#SBATCH --job-name=ali_sac
#SBATCH --output=output/duckie_%j.out
#SBATCH -e output/duckie_%j.err
#SBATCH --time=12:00:00
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
    python3 rl/sac_continuous_action.py \
    --seed 1 \
    --env-id singularity \
    --total-timesteps 2000 \
    --track \
    --version 0 \
    --buffer-size 300000 \
    --learning-starts 1000 \
    --domain-rand \
    --camera-rand \
    --dynamics-rand \
    --action-latency \
    --curriculum_randomization \
    --jerk_penalty \
    --recovery \
    --eval-interval 200 \
    --start-evaluation 1000 \
    --run-notes 'Checking singularity'"

#!/bin/bash
#SBATCH -J SAC
#SBATCH -o logs/output_%j.txt
#SBATCH -e logs/errors_%j.txt
#SBATCH -t 20:00:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -p allgroups
#SBATCH --mem=12G
#SBATCH --gres=gpu:l40s

cd $HOME/duckie-rl-project
mkdir -p logs

export DISPLAY=:$((SLURM_JOB_ID % 100 + 100))
Xvfb $DISPLAY -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log &

SIF_IMAGE="$HOME/duckie_rl.sif"

export LIBGL_ALWAYS_SOFTWARE=1
export PYGLET_HEADLESS=True
export GALLIUM_DRIVER=llvmpipe
export WANDB_MODE=offline

srun --cpu-bind=none singularity exec --nv \
    -B .:/app \
    --pwd /app \
    $SIF_IMAGE \
    bash -c "export PYTHONPATH=\$PYTHONPATH:/app:/app/src && \
    python3 rl/sac_continuous_action.py \
    --seed 1 \
    --env-id 'Downsized' \
    --total-timesteps 1000000 \
    --buffer-size 100000 \
    --batch-size 256 \
    --learning-starts 25000 \
    --track \
    --version 7 \
    --domain-rand \
    --camera-rand \
    --dynamics-rand \
    --run-notes 'Downsized observation + ALD r v1'"

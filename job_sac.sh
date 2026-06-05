#!/bin/bash
#SBATCH --job-name=ali_sac
#SBATCH --output=output/duckie_%j.out
#SBATCH -e output/duckie_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=pgpu_most
#SBATCH --account=dei_most
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4


cd $HOME/duckie-rl-project
mkdir -p output

SIF_IMAGE="$HOME/duckie_rl.sif"

export WANDB_DIR="${PROJECT_DIR}"
export WANDB_CACHE_DIR="/tmp/${USER}_wandb_cache"


export PYGLET_HEADLESS=True
export PYGLET_DEBUG_GL=False



srun --cpu-bind=none singularity exec --nv \
    -B .:/app \
    --pwd /app \
    $SIF_IMAGE \
    bash -c "export PYTHONPATH=/app:/app/src:\$PYTHONPATH && \
    export WANDB_API_KEY='$USER_WANDB_KEY' && \
    python3 rl/sac_continuous_action.py \
    --seed 1 \
    --env-id Final_r2 \
    --reward_type 'unified' \
    --total-timesteps 1500000 \
    --track \
    --version 0 \
    --buffer-size 300000 \
    --learning-starts 10000 \
    --domain-rand \
    --camera-rand \
    --dynamics-rand \
    --action-latency \
    --ema \
    --curriculum-randomization \
    --jerk-penalty \
    --recovery \
    --wandb_project_name: 'Duckie-RL-Final'
    --run-notes 'unified v1 with complete configuration for final eval'"

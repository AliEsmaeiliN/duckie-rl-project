#!/bin/bash
#SBATCH --job-name=ali_td3
#SBATCH --output=output/duckie_%j.out
#SBATCH -e output/duckie_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=pgpu_most
#SBATCH --account=dei_most
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

source $(conda info --base)/etc/profile.d/conda.sh
conda activate duckie-rl

echo "Using Python from: $(which python)"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYGLET_DEBUG_GL=False
export PYGLET_HEADLESS=True

export WANDB_DIR="/tmp/wandb_scratch_$SLURM_JOB_ID"
export WANDB_CACHE_DIR="/tmp/wandb_cache_$SLURM_JOB_ID"
export WANDB_CONFIG_DIR="/tmp/wandb_config_$SLURM_JOB_ID"

mkdir -p "$WANDB_DIR"
mkdir -p "$WANDB_CACHE_DIR"
mkdir -p "$WANDB_CONFIG_DIR"

if [ ! -f $CONDA_PREFIX/lib/libtiff.so.5 ]; then
    ln -s $CONDA_PREFIX/lib/libtiff.so.6 $CONDA_PREFIX/lib/libtiff.so.5
fi

python rl/td3_continuous_action.py \
    --seed 1 \
    --env-id Final_test \
    --total-timesteps 1500000 \
    --track \
    --version 0 \
    --buffer-size 300000 \
    --learning-starts 40000 \
    --domain-rand \
    --camera-rand \
    --dynamics-rand \
    --action-latency \
    --curriculum_randomization \
    --jerk_penalty \
    --recovery \
    --run-notes "Checking the data logs"
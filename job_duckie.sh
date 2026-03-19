#!/bin/bash
#SBATCH --job-name=duckie_rl
#SBATCH --output=output/duckie_%j.out
#SBATCH -e output/duckie_%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=pgpu_most
#SBATCH --account=dei_most
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# --- 1. Environment Setup ---
source $(conda info --base)/etc/profile.d/conda.sh
conda activate duckie-rl

# --- 2. Path Configuration ---
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

if [ ! -f $CONDA_PREFIX/lib/libtiff.so.5 ]; then
    ln -s $CONDA_PREFIX/lib/libtiff.so.6 $CONDA_PREFIX/lib/libtiff.so.5
fi

# --- 4. Launch Training ---
python rl/sac_continuous_action.py \
    --seed 2 \
    --env-id FastR_V2 \
    --total-timesteps 1000001 \
    --track \
    --grayscale \
    --exp-name "Speed Reward"

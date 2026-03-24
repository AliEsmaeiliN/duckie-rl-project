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
python rl/td3_continuous_action.py \
    --seed 1 \
    --env-id AsymetricR \
    --total-timesteps 1000001 \
    --track \
    --run-notes "Trying a new reward function which depends on the right/left turn"

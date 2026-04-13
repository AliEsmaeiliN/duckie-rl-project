#!/bin/bash
#SBATCH --job-name=duckie_rl_td3
#SBATCH --output=output/duckie_%j.out
#SBATCH -e output/duckie_%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=pgpu_most
#SBATCH --account=dei_most
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

source $(conda info --base)/etc/profile.d/conda.sh
conda activate duckie-rl

echo "Using Python from: $(which python)"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYGLET_DEBUG_GL=False
export PYGLET_HEADLESS=True

if [ ! -f $CONDA_PREFIX/lib/libtiff.so.5 ]; then
    ln -s $CONDA_PREFIX/lib/libtiff.so.6 $CONDA_PREFIX/lib/libtiff.so.5
fi

python rl/td3_continuous_action.py \
    --seed 1 \
    --env-id AdaptiveV1 \
    --total-timesteps 1000000 \
    --buffer-size 150000 \
    --track \
    --domain-rand \
    --learning-starts 30000 \
    --run-notes "New Adaptive Reward with Domain Randomization and MotionBlur"

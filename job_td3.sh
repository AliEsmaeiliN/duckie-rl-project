#!/bin/bash
#SBATCH --job-name=duckie_td3
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
    --env-id Sim2Real \
    --total-timesteps 1000000 \
    --buffer-size 150000 \
    --track \
    --motion-blur \
    --learning-starts 10000 \
    --run-notes "Starting the Sim2real Process with Hybrid Reward/MotionBlur Fixed-also changing the action clamp and other learning factors"

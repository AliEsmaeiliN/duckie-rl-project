#!/bin/bash
#SBATCH --job-name=duckie_sac
#SBATCH --output=output/duckie_%j.out
#SBATCH -e output/duckie_%j.err
#SBATCH --time=24:00:00
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

python rl/sac_continuous_action.py \
    --seed 1 \
    --env-id Sim2Real \
    --total-timesteps 1500000 \
    --track \
    --buffer-size 100000 \
    --motion-blur \
    --learning-starts 40000 \
    --domain-rand \
    --camera-rand \
    --dynamics-rand \
    --distortion \
    --run-notes "Adaptive reward with curriculum learning: 500k DR, 800k Dyn, 1M Distort" 

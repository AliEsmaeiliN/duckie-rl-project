#!/bin/bash
#SBATCH --job-name=duckie_rl
#SBATCH --output=output/duckie_%j.out
#SBATCH -e output/duckie_%j.err
#SBATCH --time=20:00:00
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

if [ ! -f $CONDA_PREFIX/lib/libtiff.so.5 ]; then
    ln -s $CONDA_PREFIX/lib/libtiff.so.6 $CONDA_PREFIX/lib/libtiff.so.5
fi

python rl/sac_continuous_action.py \
    --seed 3 \
    --num-envs 1 \
    --env-id AsymmetricR_v2 \
    --total-timesteps 1000001 \
    --track \
    --buffer-size 100000 \
    --learning-starts 20000 \
    --run-notes "Trying sac with the new reward and bigger buffer size. reduced buffer" 

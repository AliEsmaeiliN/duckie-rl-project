#!/bin/bash
#SBATCH -J DuckieRL
#SBATCH -o logs/output_%j.txt
#SBATCH -e logs/errors_%j.txt
#SBATCH -t 00:30:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -p allgroups
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx

cd $HOME/duckie-rl-project
mkdir -p logs

export DISPLAY=:$((SLURM_JOB_ID % 100 + 100))
Xvfb $DISPLAY -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log &

SIF_IMAGE="$HOME/duckie_rl.sif"

export LIBGL_ALWAYS_SOFTWARE=1
export PYGLET_HEADLESS=True
export GALLIUM_DRIVER=llvmpipe


srun singularity exec --nv \
    -B .:/app \
    --pwd /app \
    $SIF_IMAGE \
    bash -c "export PYTHONPATH=\$PYTHONPATH:/app:/app/src && \
    python3 rl/sac_continuous_action.py \
    --env-id 'testforCL' \
    --total-timesteps 2000 \
    --buffer-size 5000 \
    --batch-size 64 \
    --learning-starts 500 \
    --track"

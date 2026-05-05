#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate duckie-rl

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

export PYGLET_DEBUG_GL=False
export PYGLET_HEADLESS=True
export CUDA_VISIBLE_DEVICES=1
export PYGLET_HEADLESS_DEVICE=2

COMMON_ARGS=(
    --seed 1
    --env-id 'Downsized_aq2'
    --total-timesteps 1000000
    --buffer-size 100000
    --batch-size 256
    --track
    --domain-rand
    --camera-rand
    --dynamics-rand
    --run-notes "Downsized observation + ALD r + CR v1"
)


echo "Starting SAC Training..."
python rl/sac_continuous_action.py "${COMMON_ARGS[@]}" --version 1 --learning-starts 25000 && \
echo "SAC Complete. Starting TD3 Training..." && \
python rl/td3_continuous_action.py "${COMMON_ARGS[@]}" --version 1 --learning-starts 10000
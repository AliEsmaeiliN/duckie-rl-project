#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate duckie-rl

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Starting Duckie-RL smoke test..."
python rl/sac_continuous_action.py \
    --env-id "testforCL" \
    --buffer-size 10000 \
    --batch-size 64 \
    --learning-starts 1000 \
    --total-timesteps 2000 \
    --domain-rand \
    --distortion \
    --camera-rand \
    --dynamics-rand \
    --action-latency \
    --no-track 
#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate duckie-rl

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Starting Duckie-RL smoke test..."
python rl/sac_continuous_action.py \
    --env-id "test_intervalEval_newObsPipline" \
    --buffer-size 10000 \
    --batch-size 64 \
    --learning-starts 300 \
    --total-timesteps 2000 \
    --domain-rand \
    --camera-rand \
    --dynamics-rand \
    --action-latency \
    --direction "cw" \
    --motion-blur \
    --ema \
    --track \
    --eval-interval 100
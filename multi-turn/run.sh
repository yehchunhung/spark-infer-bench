#!/bin/bash
MODEL="openai/gpt-oss-20b"
HOST="localhost"
PORT=8000

python3 benchmark_serving_multi_turn.py \
    --model $MODEL \
    --served-model-name $MODEL \
    --url "http://$HOST:$PORT" \
    --input-file generate_multi_turn.json \
    --num-clients 4 \
    --max-active-conversations 12 \
    --no-early-stop \
    --verbose
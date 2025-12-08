#!/bin/bash
MODEL="openai/gpt-oss-20b"

python3 benchmark_serving_multi_turn.py \
    --model $MODEL \
    --served-model-name $MODEL \
    --input-file generate_multi_turn.json \
    --num-clients 2 \
    --max-active-conversations 6
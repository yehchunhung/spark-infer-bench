#!/bin/bash
MODEL="openai/gpt-oss-20b"
MAX_INPUT_LEN=2048
MAX_OUTPUT_LEN=128
OUTPUT_DIR="./results/gpt-oss-20b"

# NOTE: trtllm is unsuppported...
# https://github.com/ai-dynamo/aiperf/blob/42c68299c2fa8aa124e20922bc13d89df81dd4d1/docs/genai-perf-feature-comparison.md#endpoint-types-support-matrix 
aiperf profile \
    --model $MODEL \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --endpoint /v1/chat/completions \
    --synthetic-input-tokens-mean $MAX_INPUT_LEN \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean $MAX_OUTPUT_LEN \
    --output-tokens-stddev 0 \
    --request-timeout-seconds 30.0 \
    --random-seed 42
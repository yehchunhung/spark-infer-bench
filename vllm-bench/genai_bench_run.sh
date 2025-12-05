#!/bin/bash
MODEL="openai/gpt-oss-20b"
MAX_INPUT_LEN=2048
MAX_OUTPUT_LEN=256
ACT_PRECISION=mxfp8
MAX_ITER=1

# arg def
# https://github.com/sgl-project/genai-bench/blob/dacc5be91a89144308914bcb17184086364e97bb/docs/getting-started/cli-guidelines.md
# https://github.com/sgl-project/genai-bench/pull/124
for ((i=1; i<=MAX_ITER; i++)); do
    OUTPUT_DIR="./results/gpt-oss-20b/genai-bench/$ACT_PRECISION/run_$i"
    
    TRANSFORMERS_VERBOSITY=error \
    genai-bench benchmark \
        --api-backend vllm \
        --api-base "http://localhost:8000" \
        --api-key "xxx" \
        --api-model-name $MODEL \
        --model-tokenizer $MODEL \
        --task text-to-text \
        --traffic-scenario "D($MAX_INPUT_LEN,$MAX_OUTPUT_LEN)" \
        --max-time-per-run 10 \
        --max-requests-per-run 300 \
        --experiment-folder-name $OUTPUT_DIR \
        --additional-request-params="{'reasoning_effort': 'medium', 'max_tokens': $((MAX_OUTPUT_LEN * 4))}"
done
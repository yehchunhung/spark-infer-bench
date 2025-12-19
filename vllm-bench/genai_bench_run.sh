#!/bin/bash
MODEL_NAME="gpt-oss-20b"
MODEL="openai/$MODEL_NAME"
MAX_INPUT_LEN=2048
MAX_OUTPUT_LEN=128

ATTN_BACKEND="triton_attn"
EXP_DIR="kv-bf16-act-bf16-dp2-ep/concurrency_256"

MAX_ITER=1

# arg def
# https://github.com/sgl-project/genai-bench/blob/dacc5be91a89144308914bcb17184086364e97bb/docs/getting-started/cli-guidelines.md
# https://github.com/sgl-project/genai-bench/pull/124
for ((i=1; i<=MAX_ITER; i++)); do
    # NOTE: append /run_$i for multi exp runs
    OUTPUT_DIR="./results/$MODEL_NAME/genai-bench/$ATTN_BACKEND/$EXP_DIR"

    # don't set api-beckend as vllm, otherwise usual harmony parsing error may occur
    # https://github.com/vllm-project/vllm/issues/22519
    TRANSFORMERS_VERBOSITY=error \
    GENAI_BENCH_LOGGING_LEVEL=INFO \
    genai-bench benchmark \
        --api-backend openai \
        --api-base "http://localhost:8000" \
        --api-key "xxx" \
        --api-model-name $MODEL \
        --model-tokenizer $MODEL \
        --task text-to-text \
        --traffic-scenario "D($MAX_INPUT_LEN,$MAX_OUTPUT_LEN)" \
        --max-time-per-run 10 \
        --max-requests-per-run 300 \
        --log-dir $OUTPUT_DIR \
        --experiment-folder-name $OUTPUT_DIR \
        --additional-request-params="{\"reasoning_effort\": \"low\", \"max_tokens\": $((MAX_OUTPUT_LEN * 8))}" \
        --num-concurrency 256
done
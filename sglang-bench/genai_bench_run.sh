#!/bin/bash
MODEL_NAME="gpt-oss-20b"
MODEL="openai/$MODEL_NAME"
MAX_INPUT_LEN=2048
MAX_OUTPUT_LEN=128
OUTPUT_DIR="./results/genai-bench/$MODEL_NAME"

TRANSFORMERS_VERBOSITY=error \
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
  --experiment-folder-name $OUTPUT_DIR \
  --additional-request-params="{\"reasoning_effort\": \"low\", \"max_tokens\": $((MAX_OUTPUT_LEN * 8))}"
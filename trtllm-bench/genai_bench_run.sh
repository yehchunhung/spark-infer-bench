#!/bin/bash
MODEL="openai/gpt-oss-120b"
MAX_INPUT_LEN=2048
MAX_OUTPUT_LEN=256
OUTPUT_DIR="./results/gpt-oss-120b/genai-bench"

# arg def
# https://github.com/sgl-project/genai-bench/blob/dacc5be91a89144308914bcb17184086364e97bb/docs/getting-started/cli-guidelines.md
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
  --additional-request-params="{'reasoning_effort': 'medium', 'max_tokens': $((MAX_OUTPUT_LEN * 4))}"
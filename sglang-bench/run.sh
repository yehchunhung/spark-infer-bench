#!/bin/bash
MODEL="openai/gpt-oss-20b"
MAX_INPUT_LEN=2048
MAX_OUTPUT_LEN=128
OUTPUT_DIR="./results/gpt-oss-20b"

# guidellm benchmark \
#   --model $MODEL \
#   --target "http://localhost:8000" \
#   --request-type text_completions \
#   --profile sweep \
#   --max-seconds 30 \
#   --data "prompt_tokens=$MAX_INPUT_LEN,output_tokens=$MAX_OUTPUT_LEN" \
#   --output-dir $OUTPUT_DIR

TRANSFORMERS_VERBOSITY=error \
genai-bench benchmark \
  --api-backend sglang \
  --api-base "http://localhost:8000" \
  --api-key "xxx" \
  --api-model-name $MODEL \
  --model-tokenizer $MODEL \
  --task text-to-text \
  --max-time-per-run 30 \
  --max-requests-per-run 10
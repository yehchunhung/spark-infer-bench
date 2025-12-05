#!/bin/bash
MODEL="openai/gpt-oss-20b"
MAX_INPUT_LEN=2048
MAX_OUTPUT_LEN=128
OUTPUT_DIR="./results/gpt-oss-20b/genai-bench/test"

# arg def
# https://github.com/sgl-project/genai-bench/blob/dacc5be91a89144308914bcb17184086364e97bb/docs/getting-started/cli-guidelines.md
# (IMPORTANT!) trtllm sets reasoning effort to LOW by default.
# Always failed if it's medium.
# https://github.com/NVIDIA/TensorRT-LLM/blob/68253d9d292fbf113a9d3068bf36e042d539387a/tensorrt_llm/serve/openai_protocol.py#L563
TRANSFORMERS_VERBOSITY=error \
genai-bench benchmark \
  --api-backend openai \
  --api-base "http://localhost:8001" \
  --api-key "xxx" \
  --api-model-name $MODEL \
  --model-tokenizer $MODEL \
  --task text-to-text \
  --traffic-scenario "D($MAX_INPUT_LEN,$MAX_OUTPUT_LEN)" \
  --max-time-per-run 10 \
  --max-requests-per-run 300 \
  --log-dir $OUTPUT_DIR \
  --experiment-folder-name $OUTPUT_DIR \
  --additional-request-params="{\"reasoning_effort\": \"low\", \"max_tokens\": $((MAX_OUTPUT_LEN * 8)), \"ignore_eos\": true}"
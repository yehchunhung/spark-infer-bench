#!/bin/bash
MODEL="openai/gpt-oss-120b"
MAX_INPUT_LEN=2048
MAX_OUTPUT_LEN=128
OUTPUT_DIR="./results/gpt-oss-120b-eagle3/guidellm"


guidellm benchmark \
  --model $MODEL \
  --target "http://localhost:8001" \
  --profile sweep \
  --max-seconds 30 \
  --data "prompt_tokens=$MAX_INPUT_LEN,output_tokens=$MAX_OUTPUT_LEN" \
  --output-dir $OUTPUT_DIR
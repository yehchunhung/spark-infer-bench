#!/bin/bash
MODEL_HANDLE="openai/gpt-oss-20b" # "nvidia/Qwen3-30B-A3B-NVFP4"
HOST="localhost"
PORT=8001 # 8001

# # -s; -X POST
curl -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL_HANDLE"'",
    "messages": [
      {"role": "user", "content": "Give me a two-sentence summary of Eagle3 speculative decoding."}
    ],
    "max_tokens": 128,
    "stream": false
  }'
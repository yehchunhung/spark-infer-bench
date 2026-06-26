#!/bin/bash
MODEL_HANDLE="openai/gpt-oss-20b" # "nvidia/Qwen3-30B-A3B-NVFP4"
HOST="localhost"
PORT=8001 # 8000

# # -s; -X POST
curl -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL_HANDLE"'",
    "messages": [
      {"role": "user", "content": "What is speculative decoding?"}
    ],
    "max_tokens": 512,
    "stream": false
  }'
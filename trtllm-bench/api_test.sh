#!/bin/bash
MODEL_HANDLE="openai/gpt-oss-20b"
PORT=8001

# -s; -X POST
curl -s http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL_HANDLE"'",
    "messages": [{"role": "user", "content": "Paris is great because"}],
    "max_tokens": 64
  }'
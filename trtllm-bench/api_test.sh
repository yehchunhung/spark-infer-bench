#!/bin/bash
MODEL_HANDLE="openai/gpt-oss-20b"
PORT=8000

# -s
curl -X POST http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL_HANDLE"'",
    "messages": [{"role": "user", "content": "Paris is great because"}],
    "max_tokens": 64
  }'
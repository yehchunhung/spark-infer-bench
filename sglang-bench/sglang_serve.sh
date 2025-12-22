#!/bin/bash
MODEL="openai/gpt-oss-20b"

# 1. launch_server: launch local server
# https://github.com/sgl-project/sglang/blob/236a7c237002250b148c79bd93780d870b8b50d2/python/sglang/README.md
# 2. arg def
# https://github.com/sgl-project/sglang/blob/2970f22917cbfb559e73e254ea884394e556d5e0/docs/advanced_features/server_arguments.md
python3 -m sglang.launch_server \
    --model-path $MODEL \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --tp 1 \
    --attention-backend triton \
    --mem-fraction-static 0.75 \
    --enable-deterministic-inference \
    --max-running-requests 256 \
    --disable-radix-cache
#!/bin/bash
MODEL="openai/gpt-oss-20b"

# launch_server: launch local server
# https://github.com/sgl-project/sglang/blob/236a7c237002250b148c79bd93780d870b8b50d2/python/sglang/README.md
python3 -m sglang.launch_server \
    --model-path $MODEL \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --tp 1 \
    --attention-backend triton \
    --mem-fraction-static 0.75
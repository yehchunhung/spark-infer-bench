#!/bin/bash
MODEL="openai/gpt-oss-20b"
DOCKER_WORK_DIR="/root/spark-dev-workspace/dev/spark-infer-bench"

# WARNING: 
# 1. don't use "-p 8000:8000". use "--network host" instead 
# 2. https://github.com/vllm-project/vllm/issues/22525#issuecomment-3172271363
docker run \
    -it --gpus all --rm \
    --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --network host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/spark-dev-workspace/dev/spark-infer-bench:$DOCKER_WORK_DIR \
    --env "TIKTOKEN_ENCODINGS_BASE=$DOCKER_WORK_DIR/tiktoken" \
    nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
    trtllm-serve $MODEL \
        --max_batch_size 64 \
        --trust_remote_code \
        --port 8001 \
        --extra_llm_api_options $DOCKER_WORK_DIR/trtllm-bench/trtllm_config.yml
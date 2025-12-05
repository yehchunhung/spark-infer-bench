#!/bin/bash
HF_TOKEN="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"
DOCKER_WORK_DIR="spark-dev-workspace/dev/sglang-bench"
TIKTOKEN_DIR="spark-dev-workspace/dev/trtllm-bench/tiktoken"

# attention-backend: "flashinfer" unspported in GptOssForCausalLM
docker run \
    -it --gpus all --rm \
    --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/$DOCKER_WORK_DIR:/root/$DOCKER_WORK_DIR \
    -v ~/$TIKTOKEN_DIR:/root/$TIKTOKEN_DIR \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "TIKTOKEN_ENCODINGS_BASE=/root/$TIKTOKEN_DIR" \
    lmsysorg/sglang:spark \
    bash
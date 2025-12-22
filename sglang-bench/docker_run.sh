#!/bin/bash
BASE_WORK_DIR="spark-dev-workspace/dev/spark-infer-bench"

# attention-backend: "flashinfer" unspported in GptOssForCausalLM
docker run \
    -it --gpus all --rm \
    --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/$BASE_WORK_DIR/sglang-bench:/root/$BASE_WORK_DIR/sglang-bench \
    -v ~/$BASE_WORK_DIR/tiktoken:/root/$BASE_WORK_DIR/tiktoken \
    --env "TIKTOKEN_ENCODINGS_BASE=/root/$BASE_WORK_DIR/tiktoken" \
    lmsysorg/sglang:spark \
    bash
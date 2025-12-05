#!/bin/bash
MODEL="openai/gpt-oss-20b"

DOCKER_WORK_DIR="/root/spark-dev-workspace/dev/spark-infer-bench"
YAML_CONFIG="$DOCKER_WORK_DIR/vllm-bench/vllm_blackwell.yaml"

# VVLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1 => mxfp8 activation for MoE.
docker run \
    -it --gpus all --rm \
    --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/spark-dev-workspace/dev/spark-infer-bench:$DOCKER_WORK_DIR \
    --env "TIKTOKEN_ENCODINGS_BASE=$DOCKER_WORK_DIR/tiktoken" \
    --env "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1" \
    nvcr.io/nvidia/vllm:25.11-py3 \
    vllm serve $MODEL \
        --gpu-memory-utilization 0.75 \
        --swap-space 16 \
        --config $YAML_CONFIG
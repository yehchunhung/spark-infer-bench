#!/bin/bash
MODEL="openai/gpt-oss-20b"

DOCKER_WORK_DIR="/root/spark-dev-workspace/dev/spark-infer-bench"
YAML_CONFIG="$DOCKER_WORK_DIR/vllm-bench/vllm_blackwell.yaml"

# VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1 => mxfp8 activation for MoE.
# VLLM_ATTENTION_BACKEND=TORCH_SDPA" => torch SDPA attention backend (default is triton)
# https://docs.vllm.ai/en/latest/configuration/env_vars
# https://docs.vllm.ai/en/latest/api/vllm/attention/backends/registry/#vllm.attention.backends.registry.AttentionBackendEnum.FLASHINFER
docker run \
    -it --gpus all --rm \
    --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/spark-dev-workspace/dev/spark-infer-bench:$DOCKER_WORK_DIR \
    --env "TIKTOKEN_ENCODINGS_BASE=$DOCKER_WORK_DIR/tiktoken" \
    --env "VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=1" \
    nvcr.io/nvidia/vllm:25.11-py3 \
    vllm serve $MODEL \
        --gpu-memory-utilization 0.75 \
        --swap-space 16 \
        --config $YAML_CONFIG
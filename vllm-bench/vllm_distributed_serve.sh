#!/bin/bash
MODEL="openai/gpt-oss-20b"

DOCKER_WORK_DIR="/root/spark-dev-workspace/dev/spark-infer-bench"
YAML_CONFIG="$DOCKER_WORK_DIR/vllm-bench/vllm_blackwell.yaml"

export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')

# Check if the YAML config file exists in the container
if docker exec $VLLM_CONTAINER test -f "$YAML_CONFIG"; then
    echo "Config file already exists in the container: $YAML_CONFIG"
else
    echo "Config file not found in the container. Copying it now..."
    # Create the target directory in the container if it doesn't exist
    docker exec $VLLM_CONTAINER mkdir -p "$DOCKER_WORK_DIR/vllm-bench"
    # Copy the YAML config file into the container
    docker cp dev/spark-infer-bench/vllm-bench/vllm_blackwell.yaml $VLLM_CONTAINER:$YAML_CONFIG
fi

# Check the number of available GPUs in the container
AVAILABLE_GPUS=$(docker exec $VLLM_CONTAINER nvidia-smi --query-gpu=count --format=csv,noheader)
echo "Available GPUs in each container: $AVAILABLE_GPUS"


# VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1 => mxfp8 activation for MoE.
# VLLM_ATTENTION_BACKEND=TORCH_SDPA" => torch SDPA attention backend (default is triton)
# https://docs.vllm.ai/en/latest/configuration/env_vars
# https://docs.vllm.ai/en/latest/api/vllm/attention/backends/registry/#vllm.attention.backends.registry.AttentionBackendEnum.FLASHINFER
# --config $YAML_CONFIG -> BUG: something makes tp can be 1 only
# --tensor-parallel-size 2 -> this takes a long time to process
docker exec -it $VLLM_CONTAINER /bin/bash -c "
    TIKTOKEN_ENCODINGS_BASE=$DOCKER_WORK_DIR/tiktoken \
    VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=1 \
    vllm serve $MODEL \
        --gpu-memory-utilization 0.7 \
        --swap-space 16 \
        --tensor-parallel-size 2 \
        --enable-expert-parallel \
        --config $YAML_CONFIG"
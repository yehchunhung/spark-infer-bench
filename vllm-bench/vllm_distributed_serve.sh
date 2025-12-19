#!/bin/bash
YELLOW=$'\e[93m'
GREEN=$'\e[92m'
RED=$'\e[91m'
RESET=$'\e[0m'

MODEL="openai/gpt-oss-20b"
DOCKER_WORK_DIR="/root/spark-dev-workspace/dev/spark-infer-bench"
YAML_CONFIG="$DOCKER_WORK_DIR/vllm-bench/config/vllm_distributed_blackwell.yaml"

VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')

# check if the YAML config file exists in the container
if docker exec $VLLM_CONTAINER test -f "$YAML_CONFIG"; then
    echo "[INFO] Config file already exists in the container: $YAML_CONFIG"
else
    echo "[WARNING] Config file not found in the container. Copying it now..."
    docker exec $VLLM_CONTAINER mkdir -p "$DOCKER_WORK_DIR/vllm-bench"
    docker cp dev/spark-infer-bench/vllm-bench/vllm_blackwell.yaml $VLLM_CONTAINER:$YAML_CONFIG
fi

# check the number of available GPUs in the container
AVAILABLE_GPUS=$(docker exec $VLLM_CONTAINER nvidia-smi --query-gpu=count --format=csv,noheader)
echo "[INFO] Available GPUs in each container: $AVAILABLE_GPUS"

# display a menu for the user to choose between TP+EP and DP+EP
PS3="${YELLOW}Choose a mode to run on two Sparks: ${RESET}"
options=(
    "${GREEN}TP + EP (Tensor Parallelism + Expert Parallelism)${RESET}" 
    "${GREEN}DP + EP (Data Parallelism + Expert Parallelism)${RESET}"
)

select opt in "${options[@]}"; do
    case $opt in
        "${GREEN}TP + EP (Tensor Parallelism + Expert Parallelism)${RESET}")
            MODE="TP+EP"
            break
            ;;
        "${GREEN}DP + EP (Data Parallelism + Expert Parallelism)${RESET}")
            MODE="DP+EP"
            break
            ;;
        *)
            echo "${RED}Invalid option. Please try again.${RESET}"
            ;;
    esac
done

# VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1 => mxfp8 activation for MoE.
# VLLM_ATTENTION_BACKEND=TORCH_SDPA" => torch SDPA attention backend (default is triton)
# https://docs.vllm.ai/en/latest/configuration/env_vars
# https://docs.vllm.ai/en/latest/api/vllm/attention/backends/registry/#vllm.attention.backends.registry.AttentionBackendEnum.FLASHINFER

if [ "$MODE" == "TP+EP" ]; then
    # --tensor-parallel-size 2 -> this takes long to process at the first time
    # TEST: --enforce-eager 
    echo "[INFO] Starting vllm serve with ${YELLOW}TP = 2 + EP...${RESET}"
    docker exec -it $VLLM_CONTAINER /bin/bash -c "
        TIKTOKEN_ENCODINGS_BASE=$DOCKER_WORK_DIR/tiktoken \
        VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=1 \
        vllm serve $MODEL \
            --gpu-memory-utilization 0.7 \
            --swap-space 16 \
            --tensor-parallel-size 2 \
            --enable-expert-parallel \
            --enforce-eager \
            --config $YAML_CONFIG"

elif [ "$MODE" == "DP+EP" ]; then
    # CRITICAL: we must add `--enforce-eager` to deactivate torch.compile and CUDA graph
    # otherwise we could get the kernel error
    # https://github.com/yehchunhung/spark-infer-bench/issues/8
    # https://github.com/vllm-project/vllm/blob/e83b7e37/docs/design/debug_vllm_compile.md
    echo "[INFO] Starting vllm serve with ${YELLOW}DP = 2 + EP...${RESET}"
    docker exec -it $VLLM_CONTAINER /bin/bash -c "
        TIKTOKEN_ENCODINGS_BASE=$DOCKER_WORK_DIR/tiktoken \
        VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=1 \
        VLLM_RAY_DP_PACK_STRATEGY=\"span\" \
        vllm serve $MODEL \
            --gpu-memory-utilization 0.7 \
            --swap-space 16 \
            --data-parallel-size 2 \
            --data-parallel-size-local 1 \
            --data-parallel-backend ray \
            --enable-expert-parallel \
            --enforce-eager \
            --config $YAML_CONFIG"

else
    echo "${RED}Invalid mode selected. Please choose either TP+EP or DP+EP.${RESET}"
fi
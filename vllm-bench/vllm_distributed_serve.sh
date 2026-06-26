#!/bin/bash
YELLOW=$'\e[93m'
GREEN=$'\e[92m'
RED=$'\e[91m'
RESET=$'\e[0m'

MODEL="openai/gpt-oss-20b"
DOCKER_WORK_DIR="/root/spark-dev-workspace/dev/spark-infer-bench"
YAML_CONFIG="$DOCKER_WORK_DIR/vllm-bench/config/vllm_distributed_blackwell.yaml"

VLLM_CONTAINER=$(docker ps --format '{{.Names}}') # $(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}"

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
    "${GREEN}1P + 1D (PD Disaggregation (1 Prefill + 1 Decode))${RESET}"
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
        "${GREEN}1P + 1D (PD Disaggregation (1 Prefill + 1 Decode))${RESET}")
            MODE="1P+1D"
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
    echo "[INFO] Starting vllm serve with ${YELLOW}TP = 2 + EP...${RESET}"
    docker exec -it $VLLM_CONTAINER /bin/bash -c "
        TIKTOKEN_ENCODINGS_BASE=$DOCKER_WORK_DIR/tiktoken \
        VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=1 \
        HF_TOKEN=$HF_TOKEN \
        vllm serve $MODEL \
            --gpu-memory-utilization 0.7 \
            --swap-space 16 \
            --tensor-parallel-size 2 \
            --enable-expert-parallel \
            --config $YAML_CONFIG
    "

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
        HF_TOKEN=$HF_TOKEN \
        vllm serve $MODEL \
            --gpu-memory-utilization 0.7 \
            --swap-space 16 \
            --data-parallel-size 2 \
            --data-parallel-size-local 1 \
            --data-parallel-backend ray \
            --enable-expert-parallel \
            --enforce-eager \
            --config $YAML_CONFIG
    "

elif [ "$MODE" == "1P+1D" ]; then
    # NOTE: we need to launch `vllm serve`` from head and worker nodes separately...
    # not all from the head node
    echo "[INFO] Starting vllm serve with ${YELLOW}1 Prefill + 1 Decode${RESET}"

    # Install aiohttp in the head node
    VLLM_CONTAINER=$(docker ps --format '{{.Names}}')
    echo "[INFO] Installing aiohttp in the container..."
    docker exec $VLLM_CONTAINER pip install aiohttp quart

    # Copy the proxy server to the container
    echo "[INFO] Copying proxy server to the container..."
    docker cp /home/mtkuser/spark-dev-workspace/dev/spark-infer-bench/vllm-bench/disagg_proxy_server.py \
        $VLLM_CONTAINER:/root/spark-dev-workspace/dev/spark-infer-bench/vllm-bench/

    # Run the proxy server inside the container in detached mode
    echo "[INFO] Starting proxy server in head node..."
    docker exec -it $VLLM_CONTAINER /bin/bash -c "
        cd /root/spark-dev-workspace/dev/spark-infer-bench/vllm-bench/
        python3 disagg_proxy_server.py \
            --port 8001 \
            --kv-host 10.248.112.22 \
            --prefill-url http://10.248.112.22:8000 \
            --decode-url http://10.248.112.121:8000
    "

    sleep 1

else
    echo "${RED}Invalid mode selected. Please choose either TP+EP, DP+EP, or 1P+1D.${RESET}"
fi



#!/bin/bash
YELLOW=$'\e[93m'
GREEN=$'\e[92m'
RED=$'\e[91m'
RESET=$'\e[0m'

MODEL="openai/gpt-oss-20b"
DOCKER_WORK_DIR="/root/spark-dev-workspace/dev/spark-infer-bench"
CMD_TO_EXECUTE=""    # Command to run inside container
MODE_ENV_VARS=""     # Mode-specific environment variables
MODE_DEVICES=""      # Mode-specific device mounts

HOST_IP=$(hostname -I | awk '{print $1}')
PORT=8000
PROXY_IP=10.248.112.22
PROXY_PORT=8001

# VLLM_NIXL_SIDE_CHANNEL_HOST=$HOST_IP #buggy
# VLLM_NIXL_SIDE_CHANNEL_PORT=5600 # buggy

# --- Mode Functions ---

debug_mode() {
    echo "[INFO] Running in debug mode (interactive bash shell)..."
    CMD_TO_EXECUTE="bash"
}

single_node_mode() {
    echo "[INFO] Running vLLM in single node mode..."
    YAML_CONFIG="$DOCKER_WORK_DIR/vllm-bench/config/vllm_blackwell.yaml"

    MODE_ENV_VARS="--env VLLM_LOGGING_LEVEL=DEBUG"

    # The command is built up in a single string for simplicity with docker run
    CMD_TO_EXECUTE="vllm serve $MODEL \
        --gpu-memory-utilization 0.75 \
        --swap-space 16 \
        --config $YAML_CONFIG"
}

one_p_mode() {
    echo "[INFO] Running vLLM for 1P+1D distributed inference... Now launching ${YELLOW}Prefill ${RESET}..."
    YAML_CONFIG="$DOCKER_WORK_DIR/vllm-bench/config/vllm_blackwell_1p1d.yaml"

    # For multi-node 1P1D on DGX Spark with RoCE
    # These device mounts are required for RDMA/IB communication between nodes
    MODE_DEVICES="--device=/dev/infiniband/"

    # UCX configuration for RoCE transport
    # 5600 is the default port for Nixl side channel communication
    MODE_ENV_VARS="--env UCX_TLS=all --env UCX_NET_DEVICES=all --env VLLM_LOGGING_LEVEL=DEBUG"

    # require pip install nixl if using NixlConnector
    # pip install nixl[cu13];
    CMD_TO_EXECUTE="
    vllm serve $MODEL \
        --host $HOST_IP \
        --port $PORT \
        --gpu-memory-utilization 0.6 \
        --swap-space 16 \
        --enable-request-id-headers \
        --config $YAML_CONFIG \
        --kv-transfer-config '{
            \"kv_connector\": \"P2pNcclConnector\",
            \"kv_role\": \"kv_producer\",
            \"kv_port\":\"14579\",
            \"kv_buffer_device\": \"cuda\",
            \"kv_connector_extra_config\":{
                \"proxy_ip\":\"$PROXY_IP\",
                \"proxy_port\":\"$PROXY_PORT\",
                \"http_ip\":\"$HOST_IP\",
                \"http_port\":\"$PORT\",
                \"send_type\":\"PUT_ASYNC\",
                \"nccl_num_channels\":\"16\"
            }
        }'"
}

one_d_mode() {
    echo "[INFO] Running vLLM for 1P+1D distributed inference... Now launching ${YELLOW}Decode ${RESET}..."
    YAML_CONFIG="$DOCKER_WORK_DIR/vllm-bench/config/vllm_blackwell_1p1d.yaml"

    # For multi-node 1P1D on DGX Spark with RoCE
    # These device mounts are required for RDMA/IB communication between nodes
    MODE_DEVICES="--device=/dev/infiniband/"

    # UCX configuration for RoCE transport
    # 5600 is the default port for Nixl side channel communication
    MODE_ENV_VARS="--env UCX_TLS=all --env UCX_NET_DEVICES=all --env VLLM_LOGGING_LEVEL=DEBUG"

    # require pip install nixl if using NixlConnector
    # pip install nixl[cu13];
    CMD_TO_EXECUTE="
    vllm serve $MODEL \
        --host $HOST_IP \
        --port $PORT \
        --gpu-memory-utilization 0.6 \
        --swap-space 16 \
        --enable-request-id-headers \
        --config $YAML_CONFIG \
        --kv-transfer-config '{
            \"kv_connector\": \"P2pNcclConnector\",
            \"kv_role\": \"kv_consumer\",
            \"kv_port\":\"14580\",
            \"kv_buffer_device\": \"cuda\",
            \"kv_connector_extra_config\":{
                \"proxy_ip\":\"$PROXY_IP\",
                \"proxy_port\":\"$PROXY_PORT\",
                \"http_ip\":\"$HOST_IP\",
                \"http_port\":\"$PORT\",
                \"send_type\":\"PUT_ASYNC\",
                \"nccl_num_channels\":\"16\"
            }
        }'"
}


# P2pNcclConnector
# NixlConnector \"kv_connector_extra_config\": {\"backends\":[\"TCP\"]}
# --port必须与--kv-transfer-config中的http_port保持一致。


# --- Interactive Menu Function ---
interactive_mode_select() {
    PS3="${YELLOW}Please select an execution mode: ${RESET}"
    options=(
        "${GREEN}debug (Interactive Bash shell)${RESET}"
        "${GREEN}single_node (vLLM with vllm_blackwell.yaml)${RESET}"
        "${GREEN}1p1d (vLLM with vllm_blackwell_1p1d.yaml - 1 Prefill)${RESET}"
        "${GREEN}1p1d (vLLM with vllm_blackwell_1p1d.yaml - 1 Decode)${RESET}"
        "${RED}exit${RESET}"
    )

    select opt in "${options[@]}"; do
        case $opt in
            "${GREEN}debug (Interactive Bash shell)${RESET}")
                debug_mode
                break
                ;;
            "${GREEN}single_node (vLLM with vllm_blackwell.yaml)${RESET}")
                single_node_mode
                break
                ;;
            "${GREEN}1p1d (vLLM with vllm_blackwell_1p1d.yaml - 1 Prefill)${RESET}")
                one_p_mode
                break
                ;;
            "${GREEN}1p1d (vLLM with vllm_blackwell_1p1d.yaml - 1 Decode)${RESET}")
                one_d_mode
                break
                ;;
            "${RED}exit${RESET}")
                echo "Exiting script."
                exit 0
                ;;
            *)
                echo "${RED}Invalid selection. Please enter the number corresponding to your choice.${RESET}"
                ;;
        esac
    done
}


# --- Main Script Logic ---
interactive_mode_select

# --- Docker Execution ---
echo "[INFO] Executing command in Docker: ${YELLOW}$CMD_TO_EXECUTE${RESET}"
echo "[INFO] Mode-specific env vars: ${YELLOW}$MODE_ENV_VARS${RESET}"
echo "[INFO] Mode-specific devices: ${YELLOW}$MODE_DEVICES${RESET}"

# VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1 => mxfp8 activation for MoE.
# VLLM_ATTENTION_BACKEND=TORCH_SDPA" => torch SDPA attention backend (default is triton)
# https://docs.vllm.ai/en/latest/configuration/env_vars
# https://docs.vllm.ai/en/latest/api/vllm/attention/backends/registry/#vllm.attention.backends.registry.AttentionBackendEnum.FLASHINFER
# -p 8000:8000 \
docker run \
    -it --gpus all --rm \
    --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --network host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/spark-dev-workspace/dev/spark-infer-bench:$DOCKER_WORK_DIR \
    --env "TIKTOKEN_ENCODINGS_BASE=$DOCKER_WORK_DIR/tiktoken" \
    --env "VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=1" \
    $MODE_ENV_VARS \
    $MODE_DEVICES \
    nvcr.io/nvidia/vllm:25.11-py3 \
    /bin/bash -c "$CMD_TO_EXECUTE"
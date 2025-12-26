#!/bin/bash
MODEL="openai/gpt-oss-20b" # "nvidia/Qwen3-30B-A3B-NVFP4"
HF_TOKEN=""
DOCKER_WORK_DIR="/root/spark-dev-workspace/dev/spark-infer-bench"
CONFIG=$DOCKER_WORK_DIR/trtllm-bench/config/trtllm_config_distributed.yml

export TRTLLM_MN_CONTAINER="trtllm-multinode"

# https://github.com/mark-ramsey-ri/trt-dgx-spark/blob/7d36681982880a4f8446b8162313a68b7b339a4a/start_cluster.sh#L589C3-L589C76
# create wrapper script that sets up environment for MPI remote processes
docker exec "$TRTLLM_MN_CONTAINER" bash -c 'cat > /tmp/mpi-wrapper.sh << '\''WRAPPER'\''
#!/bin/bash
# Environment setup for MPI remote processes (Triton needs CUDA toolchain)
export PATH=/usr/local/cuda/bin:/usr/local/cmake/bin:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/bin:/usr/local/nvidia/bin:/usr/local/mpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/amazon/efa/bin:/opt/tensorrt/bin
export CUDA_HOME=/usr/local/cuda
export CPATH=/usr/local/cuda/include
export C_INCLUDE_PATH=/usr/local/cuda/include
export CPLUS_INCLUDE_PATH=/usr/local/cuda/include
export LIBRARY_PATH=/usr/local/cuda/lib64

# Triton: Explicitly set CUDA tools paths
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump
export TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm

# Full LD_LIBRARY_PATH from container (TensorRT, CUDA, PyTorch, etc.)
export LD_LIBRARY_PATH=/opt/nvidia/nvda_nixl/lib/aarch64-linux-gnu:/opt/nvidia/nvda_nixl/lib64:/usr/local/ucx/lib:/usr/local/tensorrt/lib:/usr/local/cuda/lib64:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
exec "$@"
WRAPPER
chmod +x /tmp/mpi-wrapper.sh'

# copy wrapper to remote nodes
docker exec "${TRTLLM_MN_CONTAINER}" bash -c \
    'for host in $(cat /etc/openmpi-hostfile); do scp -o StrictHostKeyChecking=no /tmp/mpi-wrapper.sh $host:/tmp/mpi-wrapper.sh 2>/dev/null || true; done'

# -e HF_TOKEN=$HF_TOKEN \
# --tp_size 2 \
docker exec \
    -e MODEL=$MODEL \
    -e CONFIG=$CONFIG \
    -e TIKTOKEN_ENCODINGS_BASE=$DOCKER_WORK_DIR/tiktoken \
    -it $TRTLLM_MN_CONTAINER bash -c '
        mpirun -x TIKTOKEN_ENCODINGS_BASE /tmp/mpi-wrapper.sh \
            trtllm-llmapi-launch \
            trtllm-serve $MODEL \
                --backend pytorch \
                --max_batch_size 64 \
                --trust_remote_code \
                --extra_llm_api_options $CONFIG \
                --port 8001
    '
#!/bin/bash
UCX_NET_DEVICES="enp1s0f0np0,enp1s0f1np1"
NCCL_SOCKET_IFNAME="enp1s0f0np0,enp1s0f1np1"

DOCKER_WORK_DIR="/root/spark-dev-workspace/dev/spark-infer-bench"
TARGET_FILE=https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/trt-llm/assets/trtllm-mn-entrypoint.sh

# BUG: 1.0.0rc3 doesn't support GPT-OSS
# https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release
# https://github.com/mark-ramsey-ri/trt-dgx-spark/blob/7d36681982880a4f8446b8162313a68b7b339a4a/docker-compose.yml#L24
docker run -d --rm \
  --name trtllm-multinode \
  --gpus '"device=all"' \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --device /dev/infiniband:/dev/infiniband \
  -e UCX_NET_DEVICES=$UCX_NET_DEVICES \
  -e NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME \
  -e OMPI_MCA_btl_tcp_if_include=$UCX_NET_DEVICES \
  -e OMPI_MCA_orte_default_hostfile="/etc/openmpi-hostfile" \
  -e OMPI_MCA_rmaps_ppr_n_pernode="1" \
  -e OMPI_ALLOW_RUN_AS_ROOT="1" \
  -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1" \
  -v ~/.cache/huggingface/:/root/.cache/huggingface/ \
  -v ~/.ssh:/tmp/.ssh:ro \
  -v ~/spark-dev-workspace/dev/spark-infer-bench:$DOCKER_WORK_DIR \
  nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc4 \
  sh -c "curl $TARGET_FILE | sh"

# UP NEXT: run this after starting the container on the head node
# docker cp openmpi-hostfile trtllm-multinode:/etc/openmpi-hostfile
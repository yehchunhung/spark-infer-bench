#!/bin/bash
export UCX_NET_DEVICES=enp1s0f1np1
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export OMPI_MCA_btl_tcp_if_include=enp1s0f1np1

# Add the directory containing libnccl.so.2 to LD_LIBRARY_PATH
# This path was found to be: ~/spark-dev-workspace/dev/spark-infer-bench/nccl-bench/nccl/build/lib
export LD_LIBRARY_PATH=~/spark-dev-workspace/dev/spark-infer-bench/nccl-bench/nccl/build/lib:${LD_LIBRARY_PATH}

# Run the all_gather performance test across both nodes
echo "[info] ==== Setting 1 ===="
mpirun -np 2 \
  -H 10.248.112.22:1,10.248.112.121:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  ./nccl-tests/build/all_gather_perf
# mpirun -np 2 \
#   -H 10.248.112.22:1,10.248.112.121:1 \
#   --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
#   -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
#   nsys profile \
#     -o ./nsys-results/all_gather_perf_setting1 \
#     --trace=cuda,nvtx,mpi \
#     ./nccl-tests/build/all_gather_perf

echo "[info] ==== Setting 2 ===="
mpirun -np 2 \
  -H 10.248.112.22:1,10.248.112.121:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  ./nccl-tests/build/all_gather_perf -b 16G -e 16G -f 2
# mpirun -np 2 \
#   -H 10.248.112.22:1,10.248.112.121:1 \
#   --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
#   -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
#   nsys profile \
#     -o ./nsys-results/all_gather_perf_setting2 \
#     --trace=cuda,nvtx,mpi \
#     ./nccl-tests/build/all_gather_perf -b 16G -e 16G -f 2
# NCCL Benchmark

Clone and build the following before running the all gather perf scripts:

```bash
cd nccl-bench

git clone https://github.com/NVIDIA/nccl.git
cd nccl && make -j src.build && cd ..

git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests && make CUDA_HOME=/usr/local/cuda NCCL_HOME=../nccl/build && cd ..
```

Then run the scripts from the `nccl-bench` directory:

```bash
cd nccl-bench
bash all_gather_perf_script-1phyport.sh
```

# 🚀 DGX Spark Inference Benchmark Suite

A comprehensive benchmark suite designed specifically for evaluating LLM inference performance on NVIDIA Spark systems. Compare TensorRT-LLM, vLLM, and SGLang frameworks with realistic multi-user workloads.

**Note**: Examples use GPT-OSS models, but the benchmark works with **any model** supported by your framework (Llama, Mistral, Mixtral, Qwen, etc.).

## ⚡ Quick Start (3 Steps)

### 1. Start Your LLM Server

Choose your framework:

```bash
# TensorRT-LLM (optimized for Spark)
bash trtllm-bench/trtllm_serve.sh

# vLLM
bash vllm-bench/vllm_serve.sh

# SGLang
bash sglang-bench/sglang_serve.sh
```

### 2. Run Your First Benchmark

#### Single-turn

```bash
bash trtllm-bench/genai_bench_run.sh    # For TensorRT-LLM
bash vllm-bench/genai_bench_run.sh      # For vLLM
```

#### Multi-turn

```bash
cd multi-turn
python benchmark_serving_multi_turn.py \
  --model "openai/gpt-oss-120b" \
  --url "http://localhost:8001" \
  --input-file generate_multi_turn.json \
  --num-clients 4 \
  --max-active-conversations 12 \
  --excel-output
```

**💡 Tip**: Replace `openai/gpt-oss-120b` with **your model name** (e.g., `meta-llama/Meta-Llama-3-70B`, `mistralai/Mixtral-8x7B`, etc.).

**Output**: Results saved to `./results/` with Excel spreadsheets and JSON data.

## 📊 What You Can Do

| Feature | Description |
|---------|-------------|
| **Multi-Turn Conversations** | Simulate realistic chat sessions with context carryover |
| **Throughput Testing** | Measure requests/sec with configurable concurrency |
| **Latency Analysis** | TTFT, TPOT, end-to-end latency at various percentiles |
| **Framework Comparison** | Side-by-side comparison of TensorRT-LLM, vLLM, SGLang |
| **Cluster Benchmarking** | Scale across multiple Spark nodes |
| **Custom Workloads** | Generate synthetic conversations from your data |


## 🎯 Key Components Explained

### Multi-Turn Benchmark (`multi-turn/`)

The flagship tool for realistic LLM testing:

```bash
python benchmark_serving_multi_turn.py \
  --model "your-model" \
  --url "http://localhost:8000" \
  --input-file generate_multi_turn.json \
  --num-clients 4 \              # Concurrent clients
  --max-active-conversations 12 \ # Parallel sessions per client
  --request-rate 10 \            # Requests/sec per client (Poisson)
  --warmup-step \                # Warmup before measuring
  --excel-output                 # Save results to Excel
```

**Why multi-turn?** Real applications have context. Simple single-turn benchmarks miss conversation patterns, prefix caching effects, and memory pressure over time.

### Cluster Discovery (`discover-spark.sh`)

Set up passwordless SSH across multiple Spark nodes:

```bash
bash discover-spark.sh
```

This automatically:
- Discovers all Spark systems on your network via avahi
- Generates shared SSH keys
- Configures node-to-node access
- Creates MPI hosts file for distributed workloads

### Framework Configurations

**TensorRT-LLM** (`trtllm-bench/trtllm_config.yml`):
```yaml
# Spark-specific optimizations
kv_cache_config:
  free_gpu_memory_fraction: 0.9  # Use 90% of GPU memory
  enable_block_reuse: false      # Optimized for benchmarking

disable_overlap_scheduler: true
enable_chunked_prefill: true     # Long sequence support

moe_config:
  backend: CUTLASS               # Max throughput on Spark
```

**vLLM** (`vllm-bench/vllm_blackwell.yaml`):
- Optimized memory utilization (75%)
- FlashInfer MoE MXFP4/BF16 acceleration
- Torch SDPA attention backend

## 🔧 Configuration Reference

### Multi-Turn Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num-clients` | Concurrent client threads | 1 |
| `--max-active-conversations` | Max sessions per client | 8 |
| `--request-rate` | Requests/sec (Poisson) | Unlimited |
| `--max-num-requests` | Total requests to send | 100 |
| `--max-turns` | Max turns per conversation | 10 |
| `--warmup-step` | Enable warmup phase | Disabled |
| `--limit-min-tokens / --max` | Output token bounds | 50-256 |
| `--conversation-sampling` | `round_robin` or `random` | round_robin |
| `--excel-output` | Save to Excel file | Disabled |

### Common Benchmark Options

```bash
# Control test duration
--max-num-requests 1000          # Stop after 1000 requests
--max-time-sec 300               # Stop after 5 minutes

# Timeout handling
--request-timeout-sec 600        # For large models

# Debug & verbose
--verbose                        # Show all requests
--print-content                  # Print LLM responses

# Output format
--excel-output                   # Excel with stats + raw data
--json-output                    # JSON conversations
```

## 📈 Metrics You Get

### Per-Request Metrics
- **TTFT** (Time To First Token): Time from request to first response token
- **TPOT** (Time Per Output Token): Average time per generated token
- **Latency**: Total end-to-end request time
- **Input/Output Tokens**: Token counts for analysis

### Aggregate Statistics
```
Runtime: 123.5s | Throughput: 12.3 req/s
Success Rate: 99.8% (249/250 requests)

Latency Distribution:
  p50:  456ms
  p90:  789ms
  p99: 1234ms

TTFT: avg 123ms | min 45ms | max 890ms
TPOT: avg 23ms  | min 12ms | max 67ms
```

### Results Files
- **Excel**: `results/<model>/benchmark_<timestamp>.xlsx`
  - Summary sheet with statistics
  - Raw data sheet with all requests
- **JSON**: `results/<model>/conversations_<timestamp>.json`
  - Original prompts + LLM responses

## 🚀 Advanced Usage

### Custom Conversation Dataset

```python
from multi-turn.bench_dataset import generate_conversations, GenConvArgs

args = GenConvArgs(
    text_files=["your_data.txt", "docs/"],
    input_num_turns=4,
    input_num_tokens=512,
    output_num_tokens=128,
    num_conversations=100
)

conversations = generate_conversations(args, tokenizer)
```

### Distributed Multi-Node Benchmarking

1. **Setup cluster**:
   ```bash
   bash discover-spark.sh
   ```

2. **Distribute scripts**:
   ```bash
   rsync -av . spark-node-01:/path/to/bench/
   ```

3. **Run coordinated benchmarks**:
   ```bash
   # On head node
   mpirun -np 4 -hostfile mpi_hosts \
     python multi-turn/benchmark_serving_multi_turn.py ...
   ```

### Custom Metrics

Add your own metrics by extending the stats processor:

```python
# In multi-turn/benchmark_serving_multi_turn.py
def process_statistics(client_metrics, ...):
    df["custom_score"] = df["ttft"] / df["output_tokens"]
    # ... your analysis
```

## 🎓 Best Practices

1. **Always Warmup**: Use `--warmup-step` to stabilize before measuring
2. **Multiple Runs**: Run 3-5 times, take median results
3. **Match Your Workload**: Use realistic conversation patterns
4. **Monitor Resources**: Track GPU memory/usage during tests
5. **Tune Per Framework**: Each framework needs different configs
6. **Watch Token Limits**: Long outputs = higher variance

## 🐛 Troubleshooting

### Server Won't Start
```bash
# Check if port is already in use
lsof -i :8000

# Check GPU memory
nvidia-smi

# View container logs
docker logs <container_id> --tail 100
```

### Benchmark Timeouts
```bash
# Increase timeout for large models
--request-timeout-sec 600

# Reduce load to avoid OOM
--num-clients 2 --max-active-conversations 4
```

### Connection Refused
```bash
# Verify server is running
curl http://localhost:8000/v1/models

# Check firewall
sudo ufw status
```

### Memory Issues
```bash
# TRTLLM: Reduce memory fraction
# Edit trtllm-bench/trtllm_config.yml:
kv_cache_config:
  free_gpu_memory_fraction: 0.7  # Lower from 0.9

# vLLM: Reduce GPU utilization
--gpu-memory-utilization 0.6
```

### Tokenization Errors
```bash
# Ensure tokenizer cache exists
ls ~/.cache/huggingface/tokenizers/

# Force refresh
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('your-model')"
```

### No Results Generated
```bash
# Check output directory
ls -la results/

# Verify write permissions
mkdir -p results/test && touch results/test/test.txt
```

## 🔗 Resources

| Resource | Documentation | Notes |
|-----------|---------------|-------|
| **TensorRT-LLM** | [docs](https://github.com/NVIDIA/TensorRT-LLM) | Nvidia GPU optimized |
| **vLLM** | [docs](https://github.com/vllm-project/vllm) | High throughput, PagedAttention |
| **SGLang** | [docs](https://github.com/sgl-project/sglang) | Fast radix attention |
| **GenAI-Bench** | [docs](https://github.com/sgl-project/genai-bench) | Unified benchmarking |
| **NVIDIA/dgx-spark-playbooks** | [docs](https://github.com/NVIDIA/dgx-spark-playbooks) | Collection of step-by-step playbooks for NVIDIA DGX Spark. |


---

**Ready to benchmark?** Start with the [Quick Start](#-quick-start-3-steps) above! 🚀
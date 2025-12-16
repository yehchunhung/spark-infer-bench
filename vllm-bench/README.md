## Get started

### Prerequisites

Before proceeding, ensure the following prerequisites are met:

- **Docker**: Install and ensure Docker daemon is running.
- **Hugging Face Account**: Create an account at [Hugging Face](https://huggingface.co/) and obtain your API token for model downloads.
- **Virtual Environment**: The `infer-bench` virtual environment must be available. It contains dependencies like `guidellm` and `genai-bench` required for benchmarking.

### 1. Serve the Model

Run the `vllm_serve.sh` script to launch the vLLM server. This will download the model from Hugging Face and serve it via Docker.

```bash
./vllm_serve.sh
```

### 2. Activate Benchmarking Environment

Source the virtual environment containing benchmarking tools:

```bash
source infer-bench/bin/activate
```

This step loads dependencies such as `guidellm` required to execute the benchmark suite.

### 3. Run Benchmarks

Execute the benchmark runner:

```bash
./genai_bench_run.sh
```

or 

```bash
./guidellm_run.sh
```

This script will invoke the benchmarking tools against the served vLLM instance and output performance metrics.

## Configuration

Adjust benchmark behavior by modifying parameters in `run.sh` or passing additional arguments to the server. For example, to increase the maximum output tokens while maintaining benchmark validity:

```bash
--additional-request-params='{"reasoning_effort": "medium", "max_tokens": 512}'
```

See the `vllm_blackwell.yaml` configuration file for model-specific parameters and adjustments.

## Troubleshooting

### 1. Encounter `openai_harmony.HarmonyError: Unexpected token 200002 while expecting start token 200006`

This error arises because vLLM/openai-harmony expects control tokens (BOS/EOS) beyond the specified `MAX_OUTPUT_LEN`. When `max_tokens` equals this limit (e.g., 128), the output buffer fills before essential tokens can be generated, causing a mismatch.

#### Fix
Set `max_tokens` significantly larger than `MAX_OUTPUT_LEN` (e.g., 4x larger) to accommodate:
- Beginning-of-sequence (BOS) tokens
- End-of-sequence (EOS) tokens
- Internal padding/control characters

Example for `MAX_OUTPUT_LEN=128`:
`--additional-request-params='{"reasoning_effort": "medium", "max_tokens": 512}'`

This ensures sufficient buffer space for control tokens while maintaining benchmark accuracy.

### 2. Docker Permission Errors

If Docker commands fail due to permissions, add your user to the Docker group:

```bash
sudo usermod -aG docker $USER
```

Log out and back in for changes to apply.

### 3. Model Download Failures

Ensure your Hugging Face API token is properly configured:

```bash
export HUGGINGFACE_TOKEN=your_api_token
```

### 4. How do I benchmark a different model?

Modify the `MODEL_ID` variable in `vllm_serve.sh` to point to your desired model on Hugging Face.

### 5. How to set KV cache as `bfloat16`?

Set `kv-cache-dtype: auto` in `vllm_blackwell.yaml`.  
> [!WARNING]
> Don't use `kv-cache-dtype: bfloat16`.
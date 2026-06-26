
#!/bin/env bash

# Prerequisite
# - git clone latest modelopt repo
# - install modelopt
# - use examples/llm_ptq/hf_ptq.py 

Core Features
- unified Hugging Face export API 
- Quantization (Weight Precision Format/Activation Precision Format/ Calibration algorithm)
  - NVFP4, FP8, INT8, INT4, etc and supports advanced algorithms such as SmoothQuant, AWQ, SVDQuant, and Double Quantization
  - Post-training quantization (PTQ) and Quantization-aware training (QAT)
  - List of features
      - `nvfp4`    |   w4a4 w4a16?   | fp4: per block 
      - `nvfp4_awq`| 
      - `fp8`      |                 |fp8: per tensor
      - `fp8_pb_wo`|
      - `fp8_pc_pt`|

      - `int8_sq`  |   w8a8          | int8: per channel/per tensor (smooth quant)
      - `int8_wo`  |   w8a16         |?

      - `int4_awq` |   w4a16 (AWQ)   | block-wise int4  
      - `w4a8_awq` |   w4a8 (AWQ)    | block-wise int4 + per tensor fp8
      - `w4a8_nvfp4_fp8`
      - `w4a8_mxfp4_fp8`
      - `nvfp4_mlp_only`
      - `nvfp4_svdquant`
      - `mxfp8`
      - (the naming in llm-compressor is more understandable) see https://docs.vllm.ai/projects/llm-compressor/en/latest/getting-started/choosing-scheme/

  - Calibration algorithm
      - algos
          - max calibration 
          - Entropy Calibration
          - SmoothQuant
          - AWQ
          - SVDQuant
          - (llm-compressor has integrated autoround & RNT(data-free method) )
           ?? GPTQ requires a large amount of auxiliary memory. This hessian matrix is used to increase the accuracy recovery of the algorithm, and is approximately the same size as the original weights.
      - Batched Calibration Support
  - Attention quantization & KV Cache quantization
      - fp8, fp8_affine, nvfp4, nvfp4_affine, nvfp4_rotate

  - SpinQuant's R3 rotation: applying transforms to attention
  - Where to target? 
     - fp8: ignore lm_head, moe's gate, mla's kv_a_proj_with_mqa, q_a_proj, model.embed_tokens
     - moe: 
     - vision: vision_encoder, vision_language_adapter

  - (+) modelopt can support fsdp & TP? | LLM Compressor does not currently support tensor parallelism for compression
  

(Questions)
Which one is GPTQ?
Where is Deepseek's fine-grained quant & Qwen3-fp8 ckpts & Kimi's int8 quant & gpt-oss's MXFP4 Quant
Rule of thumb --> fp8 --> INT4-FP8 AWQ --> nvfp4_fp8 --> nvfp4 (layer specific)
Comparison to llm-compressor & Autoround
Sensitivity of mamba: 
 - (sb) generally the order of sensitivity is MLPs < non-recurrent parameters inside SSM block < A/Lambda/recurrent matrix in SSM block.
 

- QAT (STE Straight Through Estimator)
- KD
  - mtd.convert()
- Pruning
  - mtp.prune()
- Speculative Decoding
  - mtsp.convert()
- Sparsity
  - mts.sparsify()
  - NVIDIA 2:4 sparsity pattern and various sparsification methods, such as NVIDIA ASP and SparseGPT. It supports both post-training sparsity (PTS) and sparsity-aware training (SAT).
  - (llm-compressor integrated sparsegpt, spinquant and quip)
Precompile
python -c "import modelopt.torch.quantization.extensions as ext; ext.precompile()"

TODO: See how to do https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_non_uniform

TODO: model history

kimi
mistral large 3 --> NVFP4 format, built with llm-compressor. 
gpt-oss
Jamba 1.5 large -->  ExpertsInt8 (w8a16), a novel quantization technique tailored for MoE models. 
- Doesn't require calibration preprocess.
- No quality degradation.

sglang quantization
- (offline quant) GPTQ and AWQ, which collect and pre-compute various statistics from the original weights using the calibration dataset.
- (online quant) Take note, if your model is per-channel quantized (INT8 or FP8) with per-token dynamic quantization activation, you can opt to include --quantization w8a8_int8 or --quantization w8a8_fp8 to invoke the corresponding CUTLASS int8_kernel or fp8_kernel in sgl-kernel. 
   - torchao online quant --torchao-config int4wo-128
   - 
autoround
GPTQModel


echo "Quantizing original model..."


python hf_ptq.py \
    --pyt_ckpt_path=$MODEL_PATH \
    --export_path=$SAVE_PATH \
    --sparsity_fmt=$SPARSITY_FMT \
    --qformat="${QFORMAT// /,}" \
    --calib_size=$CALIB_SIZE \
    
    --inference_tensor_parallel=$TP \
    --inference_pipeline_parallel=$PP \
    $PTQ_ARGS \
    $AWQ_ARGS


python3 hf_ptq.py \
    --pyt_ckpt_path /home/omniml_data_3/models/Qwen3.5-397B-A17B \
    --qformat nvfp4_mlp_only \
    --kv_cache_qformat \
    --sparsity_fmt= \
    --export_path /home/omniml_data_3/zhiyuc/checkpoints/Qwen3.5-397B-A17B-NVFP4 \
    --trust_remote_code \
    --gpu_max_mem_percentage 0.7 \
    --batch_size=$CALIB_BATCH_SIZE \

#--qformat: Quantization formats fp8, nvfp4, nvfp4_mlp_only
#--kv_cache_qformat: KV cache quantization format (default: fp8)
#Note: The default kv_cache_qformat may not be optimal for all use cases. Consider setting this explicitly.

#--dataset
#--low_memory_mode
#--kv_cache_qformat
#--use_seq_device_map
#--calib_seq

# --batch_size --> default to 0, and max batch size will be calculated on the fly
# --calib_size --> default to 512
# --calib_seq --> default to 512, Maximum sequence length for calibration.



# auto quantize
# --auto_quantize_bits
# --auto_quantize_method
# --auto_quantize_score_size

# AQW
# --awq_block_size=


# FP8 2D blockwise fake quantization config for deepseek models
# FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG = {
#     "quant_cfg": {
#         "*weight_quantizer": {
#             "num_bits": (4, 3),
#             "block_sizes": {-1: 128, -2: 128},
#             "enable": True,
#         },
#         "*input_quantizer": {"enable": False},
#         **_default_disabled_quantizer_cfg,
#     },
#     "algorithm": "max",
# }

# Deepseek v3 Quantization strategy:
    # Weight Scaling: Applied at a 128 × 128 block-wise level.
    # Activation Scaling: Applied at a 1 × 128 tile-wise level.



# TODO
# from grok

python hf_ptq.py \
  --pyt_ckpt_path Qwen/Qwen2.5-32B-Instruct \  # Hugging Face model path or local dir
  --qformat nvfp4 \  # Quantization format (nvfp4 for NVFP4)
  --kv_cache_qformat nvfp4 \  # Optional: Quantize KV cache (omit to disable)
  --batch_size 32 \  # Calibration batch size (adjust based on GPU memory)
  --export_path ./modelopt_quantized_model \  # Output dir for exported checkpoint
  --calib_size 512 \  # Number of calibration samples
  --low_memory_mode \  # Optional: Compress weights before calibration to save memory
  --use_seq_device_map  # Optional: Distribute model sequentially across GPUs if OOM


# For Qwen3.5 MoE:
# default nvfp4 recipe keeps all GDN layers, visual modules and lm_head, mtp.layers intact
# the process takes over 90GB
# --kv_cache_qformat fp8 \  # default kv_cache quant to fp8
# ,open_code_reasoning

TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas python3 hf_ptq.py \
  --pyt_ckpt_path Qwen/Qwen3.5-35B-A3B \
  --qformat nvfp4_omlp_only \
  --kv_cache_qformat none \
  --calib_seq 4096 \
  --export_path /home/mtkuser/spark-dev-workspace/dev/spark-infer-bench/jeremy-sglang-baremetal/modelopt_quantized_models \
  --inference_tensor_parallel 1 \
  --inference_pipeline_parallel 1 \
  --gpu_max_mem_percentage 0.7 \
  --verbose \
  --dataset "nemotron-post-training-dataset-v2" \
  --trust_remote_code \
  --low_memory_mode

# bugs all the way

# 1. cuz qwen3.5 is an VLM ... --> AttributeError: 'Qwen3_5MoeConfig' object has no attribute 'vocab_size' 
     --> patch hf_ptq.py 
     model = AutoModelForCausalLM.from_pretrained --> 

# 2. --low_memory_mode --> 
  File "/home/mtkuser/spark-dev-workspace/torch291-sglang/lib/python3.12/site-packages/causal_conv1d/cpp_functions.py", line 22, in _causal_conv1d_fwd_cpp
    causal_conv1d_cuda.causal_conv1d_fwd(
  RuntimeError: Expected x.is_cuda() to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.
  
  Interestingly, without --low_memory_mode, there is no following warning

  Qwen3_5MoeModel LOAD REPORT from: Qwen/Qwen3.5-35B-A3B
  Key            | Status     | Details
  ---------------+------------+--------
  lm_head.weight | UNEXPECTED |      

# 3. decoding error 
  File "/home/mtkuser/spark-dev-workspace/torch291-sglang/lib/python3.12/site-packages/transformers/generation/utils.py", line 2668, in generate
    result = decoding_method(
             ^^^^^^^^^^^^^^^^
  File "/home/mtkuser/spark-dev-workspace/torch291-sglang/lib/python3.12/site-packages/transformers/generation/utils.py", line 2914, in _sample
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  torch.AcceleratorError: CUDA error: device-side assert triggered
  Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.


  --attn_implementation \

  # "quant_algo": "NVFP4",
  # "kv_cache_quant_algo": "FP8",
  # "group_size": 16,
  # "exclude_modules": [
  #   "lm_head",
  #   "linear_attn*",
  #   "mlp.shared_expert_gate",
  #   "model.language_model.layers.3.mlp.shared_expert_gate",
  #   "model.language_model.layers.3.self_attn*",
  #   "model.visual*",
  #   "mtp.layers.0*"




# compare the eval benchmark on `Qwen/Qwen3.5-35B-A3B-FP8` and 



model = mtq.quantize(model, quant_cfg, forward_loop)
# 2. Compute MSE for all quantizers
mse = mtq.compute_quantization_mse(model, forward_loop)
# 3. Print the top-5 noisiest quantizers
for name, err in sorted(mse.items(), key=lambda x: -x[1])[:5]:
   print(f"{name}: {err:.4e}")

#!/bin/bash
TARGET_DIR="./results/gpt-oss-20b/genai-bench/triton_attn/benchmark-plot/1node_vs_2node/tp2_eager_vs_dp2_eager"

# use group-key to determine the legend name for each class
# available key are in experiment_metadata.json
genai-bench plot \
    --experiments-folder $TARGET_DIR \
    --group-key experiment_folder_name \
    --preset 2x4_default \
    --verbose
#!/bin/bash
TARGET_DIR="../trtllm-bench/results/gpt-oss-20b/benchmark-plot"

# use group-key to determine the legend name for each class
# available key are in experiment_metadata.json
genai-bench plot \
    --experiments-folder $TARGET_DIR \
    --group-key experiment_folder_name \
    --preset 2x4_default
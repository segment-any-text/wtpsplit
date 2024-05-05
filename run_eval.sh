#!/bin/bash

# Check if sufficient arguments are provided
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 MODEL_PATH 'threshold_list'"
    echo "Example: $0 /path/to/model '0.1 0.2 0.3'"
    exit 1
fi

# Assign arguments to variables
MODEL_PATH="$1"
threshold_list=($2)

# Loop over threshold_list
for threshold in "${threshold_list[@]}"; do
    # Execute the Python script
    python3 wtpsplit/evaluation/intrinsic.py --model_path "$MODEL_PATH" --threshold "$threshold" --keep_logits
done
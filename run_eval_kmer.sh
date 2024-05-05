#!/bin/bash

# Check if sufficient arguments are provided
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 MODEL_PATH 'k_list' 'threshold_list'"
    echo "Example: $0 /path/to/model '1 2 3' '0.1 0.2 0.3'"
    exit 1
fi

# Assign arguments to variables
MODEL_PATH="$1"
k_list=($2)
threshold_list=($3)

# Loop over k_list
for k in "${k_list[@]}"; do
    # Loop over threshold_list
    for threshold in "${threshold_list[@]}"; do
        # Execute the Python script
        python3 wtpsplit/evaluation/intrinsic_pairwise.py --model_path "$MODEL_PATH" --k "$k" --threshold "$threshold" --keep_logits
    done
done
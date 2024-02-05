#!/bin/bash

# This script takes one argument: --model_path
MODEL_PATH="$1"

# array to hold the following additional commands:
# "--do_lowercase", "--do_remove_punct", "--do_lowercase --do_remove_punct"
assoc_array=(
  # "--do_lowercase"
  "--do_remove_punct"
  "--do_lowercase --do_remove_punct"
)
# Loop through the associative array
for i in "${assoc_array[@]}"
do
  # Construct the command
  cmd="python3 wtpsplit/evaluation/intrinsic.py --model_path $MODEL_PATH $i"
  
  
  # Execute the command
  echo "Executing: $cmd"
  eval $cmd
done

echo "All evaluations completed."

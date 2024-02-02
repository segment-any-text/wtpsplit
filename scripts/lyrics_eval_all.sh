#!/bin/bash

# This script takes one argument: --model_path
MODEL_PATH="$1"

# Define an associative array for eval_data_path and their corresponding save_suffixes
declare -A eval_data_paths=(
  ["data/lyrics_lines.pt"]="lines"
  ["data/lyrics_lines_lower.pt"]="lines_lower"
  ["data/lyrics_lines_rmp_lower.pt"]="lines_lower"
  ["data/lyrics_lines_rmp_lower.pt"]="lines_lower_rmp"
  ["data/lyrics_lines_rmp.pt"]="lines_rmp"
  ["data/lyrics_verses_strip_n.pt"]="verses"
  ["data/lyrics_verses_lower_strip_n.pt"]="verses_lower"
  ["data/lyrics_verses_rmp_strip_n.pt"]="verses_rmp"
  ["data/lyrics_verses_rmp_lower_strip_n.pt"]="verses_lower_rmp"
)

# Path to the custom_language_list file
CUSTOM_LANG_LIST="data/lyrics_langs.csv"

# Loop through the associative array
for eval_data_path in "${!eval_data_paths[@]}"; do
  save_suffix="${eval_data_paths[$eval_data_path]}"

  # Construct the command
  cmd="python3 wtpsplit/evaluation/intrinsic.py --model_path $MODEL_PATH --eval_data_path $eval_data_path --custom_language_list $CUSTOM_LANG_LIST"
  
  # Append --save_suffix if it's not empty
  if [[ -n $save_suffix ]]; then
    cmd="$cmd --save_suffix $save_suffix"
  fi
  
  # Execute the command
  echo "Executing: $cmd"
  eval $cmd
done

echo "All evaluations completed."

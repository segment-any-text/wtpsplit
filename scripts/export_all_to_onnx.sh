# all models in manually defined array of models
models=(
    "sat-1l-sm"
    "sat-3l-sm"
    "sat-6l-sm"
    "sat-9l-sm"
    "sat-12l-sm"
    "sat-1l"
    "sat-3l"
    "sat-6l"
    "sat-9l"
    "sat-12l"
    "sat-1l-no-limited-lookahead"
    "sat-3l-no-limited-lookahead"
    "sat-6l-no-limited-lookahead"
    "sat-9l-no-limited-lookahead"
    "sat-12l-no-limited-lookahead"
)

for model in "${models[@]}"
do
    python scripts/export_to_onnx_sat.py --model_name_or_path=segment-any-text/$model --output_dir=output_onnx_exports/$model --upload_to_hub=True
done
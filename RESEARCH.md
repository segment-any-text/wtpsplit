# Research environment

Training, evaluation, and data-acquisition code live in this repo but are left
out of the `wtpsplit` wheel so inference and few-shot adaptation installs stay
small.

Source checkout:

```bash
git clone https://github.com/segment-any-text/wtpsplit
cd wtpsplit
uv sync --locked --group research --extra legacy
```

Add `--extra onnx-cpu` for the full API test suite (ONNX-backed tests need the
runtime even when research work uses PyTorch).

Cluster train/eval: [docs/OPERATOR.md](docs/OPERATOR.md).  
Stage sequence and blockers: [docs/TRAINING_CURRICULUM.md](docs/TRAINING_CURRICULUM.md).

```bash
uv run python wtpsplit/train/train.py configs/your_config.json
uv run python wtpsplit/train/train_SM.py configs/your_config.json
```

Few-shot LoRA adaptation is in the public API (`sat.adapt(...)`, optional
`sat.save_adapter(path)`). `train_lora.py` and AdapterHub configs remain as
references to the old transformers 4 setup; transformers 5 uses the native API.

Held-out 10/50/100-shot BOUQuET adaptation comparison:

```bash
uv run --group research python scripts/evaluate_adaptation.py \
  --language de \
  --bouquet-language deu_Latn \
  --shots 10,50,100 \
  --eval-sentences 200
```

The `research` group is not a PyPI extra on purpose. Extras can add dependencies
but cannot put `wtpsplit/train`, `wtpsplit/evaluation`, or
`wtpsplit/data_acquisition` back into a wheel that excluded them, and a second
distribution writing into the same package would own files ambiguously on
uninstall.

When changing this environment: `uv add --group research <dependency>` and commit
the `pyproject.toml` / `uv.lock` updates with the change.

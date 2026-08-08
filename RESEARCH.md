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

Stage 1 data, training, evaluation, and open work:
[docs/STAGE1.md](docs/STAGE1.md).

Stage 2 sentence supervision, replay experiments and evaluation:
[docs/STAGE2.md](docs/STAGE2.md). Its source routing and corpus builders are in
[docs/STAGE2_DATA.md](docs/STAGE2_DATA.md). Early results and negative
experiments are summarized in [docs/STAGE2_RESULTS.md](docs/STAGE2_RESULTS.md).

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

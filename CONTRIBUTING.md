# Contributing

wtpsplit uses [uv](https://docs.astral.sh/uv/) for environments, locking, and
builds.

```bash
uv sync --locked --extra legacy --extra onnx-cpu
uv run ruff check .
uv run mypy --strict wtpsplit/__init__.pyi wtpsplit/segmentation.py wtpsplit/constants.py
uv run pytest
uv run python -m doctest README.md
```

Use `uv add` for runtime dependencies and `uv add --dev` for development
dependencies. Include the resulting `pyproject.toml` and `uv.lock` updates in
the same change.

Research and training modules are available through source checkout:

```bash
uv sync --locked --group research --extra legacy
```

Before opening a change that affects inference, compare representative outputs
against the checked baseline using `scripts/capture_outputs.py` and
`scripts/compare_outputs.py`.

Changes to custom Transformers integrations must preserve the registration and
vendoring boundary documented in [`MODEL_BACKENDS.md`](MODEL_BACKENDS.md).

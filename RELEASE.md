# Release process for wtpsplit

1. Review `MIGRATION.md`, then bump the version in `pyproject.toml` (`[project] version`) and `wtpsplit/__init__.py` (`__version__`). Both must match; PyPI will reject a duplicate.
2. Run `uv lock` if dependencies changed, and commit the updated `uv.lock`.
3. Run `uv sync --locked --extra legacy --extra onnx-cpu`, `uv run ruff check .`, `uv run mypy --strict wtpsplit/__init__.pyi wtpsplit/segmentation.py wtpsplit/constants.py`, and `uv run pytest`.
4. Run `uv build` and verify the wheel with `uv run pytest tests/test_packaging.py`.
5. Wait for the GitHub CI Actions to pass.
6. Run `bash release.sh` to build and upload to PyPI.

`release.sh` uses `uv build` and `uv publish`; no separate `build`/`twine` install is needed.
Set `UV_PUBLISH_TOKEN` (or pass `--token`) for authentication.

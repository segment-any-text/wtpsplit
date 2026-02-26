# Release process for wtpsplit

1. Bump version in `wtpsplit/__init__.py` and `setup.py`, e.g. https://github.com/segment-any-text/wtpsplit/commit/0945744ef30420f4982aa9429ebf0908ca0e0666
2. Wait for the Github CI Actions to pass.
3. Install build and twine if needed: `pip install build twine`
4. Run `bash release.sh` to build and upload to PyPI.
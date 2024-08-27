# Release process for wtpsplit

1. Bump version in `wtpsplit/__init__.py` and `setup.py`, e.g. https://github.com/segment-any-text/wtpsplit/commit/0945744ef30420f4982aa9429ebf0908ca0e0666
2. Wait for the Github CI Actions to pass.
3. Run `bash release.sh` to create a new release on PyPI.
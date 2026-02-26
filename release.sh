#!/usr/bin/env bash
set -e
rm -rf dist
pip install -q build twine
python -m build
twine upload dist/*
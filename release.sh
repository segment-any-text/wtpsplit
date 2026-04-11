#!/usr/bin/env bash
set -e
rm -rf dist
# setuptools 77+ emits PEP 639 "License-File" in wheel METADATA; twine needs
# packaging>=25 to validate those uploads (older packaging errors on license-file).
pip install -q -U build twine "packaging>=25"
python -m build
twine upload dist/*
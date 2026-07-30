#!/usr/bin/env bash
set -e
rm -rf dist
uv build
uvx twine check dist/*
uv publish

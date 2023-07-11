rm -r dist
python -m build
twine upload dist/*
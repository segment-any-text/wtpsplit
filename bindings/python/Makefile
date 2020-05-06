install:
	pip install -r dev_requirements.txt

build:
	# --manylinux 1-unchecked is needed because the extension-module feature in PyO3 does not work properly, see Cargo.toml
	maturin build --release --manylinux 1-unchecked
	pip install --ignore-installed --upgrade ../../target/wheels/*.whl

test:
	pytest -s
install:
	pip install -r dev_requirements.txt

build:
	# --manylinux 1-unchecked is needed because the extension-module feature in PyO3 does not work properly, see Cargo.toml
	maturin build --release --manylinux 1-unchecked
	# select the least recently modified wheel
	# needed for matrix build, this could theoretically be a race condition but very likely good enough
	# pip install --ignore-installed --upgrade $(ls -1t ../../target/wheels/* | head -1)

test:
	pytest -s
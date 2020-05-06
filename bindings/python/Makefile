install:
	pip install -r dev_requirements.txt

build:
	# --manylinux 1-unchecked is needed because the extension-module feature in PyO3 does not work properly, see Cargo.toml
	wheelname=temp-$$(date +'%Y%m%d-%H%M%S%N')-wheels && \
	maturin build --release --manylinux 1-unchecked --out $$wheelname && \
	pip install --ignore-installed --upgrade $$wheelname/* && \
	rm -r $$wheelname

test:
	pytest -s
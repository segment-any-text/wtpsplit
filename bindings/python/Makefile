install:
	pip install -r dev_requirements.txt

build:
	# --manylinux 1-unchecked is needed because the extension-module feature in PyO3 does not work properly
	maturin build --release --manylinux 1-unchecked
	version=`python -c "import sys; print(str(sys.version_info.major) + str(sys.version_info.minor))"` && \
	pip install --ignore-installed --upgrade ../../target/wheels/*cp$$version*.whl

test:
	pytest -s
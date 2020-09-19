install:
	pip install -r dev_requirements.txt

build:
	maturin build --release --manylinux 1
	version=`python -c "import sys; print(str(sys.version_info.major) + str(sys.version_info.minor))"` && \
	pip install --use-feature=2020-resolver --ignore-installed --upgrade ../../target/wheels/*cp$$version*.whl

test:
	pytest -s
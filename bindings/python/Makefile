install:
	pip install -r dev_requirements.txt

build:
	maturin build --release
	version=`python -c "import sys; print(str(sys.version_info.major) + str(sys.version_info.minor))"` && \
	pip install --ignore-installed --upgrade ../../target/wheels/*cp$$version*.whl
	mkdir -p build
	cp -a ../../target/wheels/*cp$$version*.whl build/ 

test:
	pytest -s
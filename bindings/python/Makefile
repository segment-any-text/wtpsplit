install:
	pip install -r dev_requirements.txt

build:
	# add extension-module feature to PyO3 and remove from workspace to change name
	mv Cargo.toml Cargo.temp && cp Cargo.build.toml Cargo.toml
	maturin build --interpreter python --release --manylinux 1
	mv Cargo.temp Cargo.toml
	version=`python -c "import sys; print(str(sys.version_info.major) + str(sys.version_info.minor))"` && \
	python -m pip install --ignore-installed --upgrade target/wheels/*cp$$version*.whl

test:
	pytest -s
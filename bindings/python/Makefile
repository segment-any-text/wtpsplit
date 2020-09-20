install:
	pip install -r dev_requirements.txt

build:
	# add extension-module feature to PyO3
	echo 'features = ["extension-module"]' >> Cargo.toml
	maturin build --release --manylinux 1
	# remove extension-module feature, see https://stackoverflow.com/a/4881962
	# use "| true" to not fail in CI on e. g. windows, we don't really care if its not restored there
	(head -c -33 Cargo.toml > out && mv out Cargo.toml) | true
	version=`python -c "import sys; print(str(sys.version_info.major) + str(sys.version_info.minor))"` && \
	pip install --use-feature=2020-resolver --ignore-installed --upgrade ../../target/wheels/*cp$$version*.whl

test:
	pytest -s
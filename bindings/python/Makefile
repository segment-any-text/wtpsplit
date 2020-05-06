build:
	cargo build --release

test:
	cp -a ../../target/release/libnnsplit.so nnsplit.so
	pytest -s tests/
	rm nnsplit.so
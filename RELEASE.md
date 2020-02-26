# Release Instructions

Before starting, make sure the latest commit passes CI (https://github.com/bminixhofer/nnsplit/actions).
Versions are kept in sync between Rust, Python and Javascript e. g. a bugfix in JS also causes version bumps for Rust and Python. 
First, clean the repository: `git clean -xdn` to ensure no wrong files are deleted, then `git clean -xdf`.

## Releasing Javascript Bindings

In the directory `nnsplit/js_lib`:

```bash
make install
make test
npm version <new-version>
npm publish
```

## Releasing Rust Bindings

In the directory `nnsplit/rust_lib`:
- bump the version in the `Cargo.toml` file:
- test the bindings: `cargo test`
- release with Cargo:

```bash
 # make sure this fails because of dirty files. The dirty files must ONLY be the files in the `nnsplit/rust_lib/data` directory which are also in `nnsplit/data`
cargo package
cargo publish --allow-dirty
```

## Releasing Python Bindings

In the directory `nnsplit/python_lib`:
- bump the version in `./nnsplit/__init__.py` and in `pyproject.toml`
- test the bindings: `make test`
- build and publish: `make publish`
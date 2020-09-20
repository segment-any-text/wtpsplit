set -e

# login to cargo
cargo login $CARGO_KEY

# install the TOML cli tool
cargo install toml-cli

function update_cargo_toml_version {
    VERSION=$1
    FILE=$2

    toml set $2 package.version $1 > out && mv out $2
}

function update_version {
    VERSION=$1

    update_cargo_toml_version $1 nnsplit/Cargo.toml
    update_cargo_toml_version $1 bindings/python/Cargo.toml
    update_cargo_toml_version $1-python bindings/python/Cargo.build.toml

    npm version $1 --prefix bindings/javascript --allow-same-version
}

update_version $NEW_VERSION
cp -a README.md nnsplit/README.md
cd nnsplit
cargo publish --allow-dirty
cd ..

cp -a README.md bindings/python/README.md
cd bindings/python
twine upload $WHEEL_DIR/*
cd ../..

cd bindings/javascript
npm run build
cp -a ../../README.md pkg/README.md
cd pkg
npm publish
cd ..
cd ../../

update_version $NEW_VERSION-post
rm nnsplit/README.md
rm bindings/javascript/pkg/README.md
rm bindings/python/README.md
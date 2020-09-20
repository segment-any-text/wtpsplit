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

    npm version $1 --prefix bindings/javascript --allow-same-version
}

update_version $NEW_VERSION
cp -a README.md nnsplit/README.md
cd nnsplit
cargo publish --allow-dirty
cd ..

# update core version to avoid clash in Cargo.lock, all of this is VERY hacky, see https://github.com/PyO3/maturin/issues/313
update_cargo_toml_version $1-post nnsplit/Cargo.toml

cp -a README.md bindings/python/README.md
cd bindings/python
twine upload $WHEEL_DIR/*
cd ../..

# change it back
head -n -2 bindings/python/Cargo.toml > out && mv out bindings/python/Cargo.toml
toml set bindings/python/Cargo.toml package.name $NAME > out && mv out bindings/python/Cargo.toml

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
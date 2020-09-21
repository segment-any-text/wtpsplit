set -e

# login to cargo
cargo login $CARGO_KEY

cp -a README.md nnsplit/README.md
cd nnsplit
cargo publish --allow-dirty
cd ..

cp -a README.md bindings/python/README.md
twine upload $WHEEL_DIR/*

cd bindings/javascript
npm run build
cp -a ../../README.md pkg/README.md
npm publish pkg
cd ../../

rm nnsplit/README.md
rm bindings/javascript/pkg/README.md
rm bindings/python/README.md
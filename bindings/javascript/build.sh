wasm-pack build --target bundler --out-dir pkg/nnsplit.bundle
wasm-pack build --target nodejs --out-dir pkg/nnsplit.node

cp -a package.json pkg/package.json
# NNSplit Rust Bindings

![Crates.io](https://img.shields.io/crates/v/nnsplit)
![CI](https://github.com/bminixhofer/nnsplit/workflows/CI/badge.svg)
![License](https://img.shields.io/github/license/bminixhofer/nnsplit)

Fast, robust sentence splitting with bindings for Python, Rust and Javascript and pretrained models for English and German.

## Installation

Add NNSplit as a dependency to your `Cargo.toml`:

```toml
[dependencies]
# ...
nnsplit = "<version>"
# ...
```

## Usage

```rust
use nnsplit::NNSplit;

fn main() -> failure::Fallible<()> {
    let splitter = NNSplit::new("en")?;

    let input = vec!["This is a test This is another test."];
    println!("{:#?}", splitter.split(input));

    Ok(())
}
```

Models for German (`NNSplit::new("de")`) and English (`NNSplit::new("en")`) come prepackaged with NNSplit. Alternatively, you can also load your own model with `NNSplit::from_model(model: tch::CModule)`.


## Advanced

Run `cargo test` to test the NNSplit Rust Bindings. The NNSplit Rust Bindings also come with a simple example which splits the text passed via a CLI.

```bash
cargo run --example cli -- <text> <language>
```

for example:

```bash
cargo run --example cli -- "This is a test This is another test." en
```

You can run a benchmark of the Rust Bindings with `cargo bench`.
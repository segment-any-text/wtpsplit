## Installation

NNSplit for Rust uses [tract](https://github.com/sonos/tract) as backend.
Add NNSplit as a dependency to your `Cargo.toml`:

```toml
# ...

[dependencies.nnsplit]
version = "<version>"
 # to automatically download pretrained models and to use tract for inference, respectively
features = ["model-loader", "tract-backend"]

# ...
```
&nbsp;

## Use

In `src/main.rs`:
```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let splitter =
        nnsplit::NNSplit::load("en", nnsplit::NNSplitOptions::default())?;

    let input: Vec<&str> = vec!["This is a test This is another test."];
    let splits = &splitter.split(&input)[0];

    for sentence in splits.iter() {
        println!("{}", sentence.text());
    }

    Ok(())
}
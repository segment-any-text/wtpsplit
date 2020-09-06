# NNSplit

[![PyPI](https://img.shields.io/pypi/v/nnsplit)](https://pypi.org/project/nnsplit/)
[![Crates.io](https://img.shields.io/crates/v/nnsplit)](https://crates.io/crates/nnsplit)
[![npm](https://img.shields.io/npm/v/nnsplit)](https://www.npmjs.com/package/nnsplit)
![CI](https://github.com/bminixhofer/nnsplit/workflows/CI/badge.svg)
![License](https://img.shields.io/github/license/bminixhofer/nnsplit)

Fast, robust sentence splitting with bindings for Python, Rust and Javascript.

## Features

- __Robust__: Does not depend on proper punctuation and casing to split text into sentences.
- __Small__: NNSplit uses a byte-level LSTM, so weights are very small which makes it easy to run in the browser.
- __Portable__: Models are trained in Python, but inference can be done from Javascript, Rust and Python.
- __Fast__: Can run on your GPU to split 10k short texts in less than 400ms in Colab. See [train.ipynb](train/train.ipynb).

## Pretrained models

NNSplit comes with pretrained models. They were evaluated on the [OPUS Open Subtitles](http://opus.nlpl.eu/OpenSubtitles-v2018.php) dataset by concatenating 2 - 4 sentences and measuring the number of concatenations which are split completely correctly vs. the total number of concatenations.

See [evaluate.ipynb](train/evaluate.ipynb) for details.

### [`en`](models/en)

|                              |   NNSplit |   Spacy (Tagger) |   Spacy (Sentencizer) |
|:-----------------------------|----------:|-----------------:|----------------------:|
| Clean                        |  0.754371 |         0.853603 |            0.820934   |
| Partial punctuation          |  0.485907 |         0.517829 |            0.249753   |
| Partial case                 |  0.761754 |         0.825119 |            0.819679   |
| Partial punctuation and case |  0.443704 |         0.458619 |            0.249873   |
| No punctuation and case      |  0.166273 |         0.180859 |            0.00463281 |

### [`de`](models/de)

|                              |   NNSplit |   Spacy (Tagger) |   Spacy (Sentencizer) |
|:-----------------------------|----------:|-----------------:|----------------------:|
| Clean                        |  0.818902 |        0.833368  |            0.878471   |
| Partial punctuation          |  0.463999 |        0.426458  |            0.266312   |
| Partial case                 |  0.823565 |        0.792839  |            0.876678   |
| Partial punctuation and case |  0.447231 |        0.377201  |            0.26697    |
| No punctuation and case      |  0.198165 |        0.0952267 |            0.00756195 |

## Python Usage

### Installation

NNSplit has PyTorch as the only dependency.

Install it with pip: `pip install nnsplit`

### Usage

```python
from nnsplit import NNSplit
splitter = NNSplit.load("en")

# returns `Split` objects
splits = splitter.split(["This is a test This is another test."])[0]

# a `Split` can be iterated over to yield smaller splits or stringified with `str(...)`.
for sentence in splits:
   print(sentence)
```

## Javascript Usage

### Installation

The Javascript bindings for NNSplit have [tractjs](https://github.com/bminixhofer/tractjs) as the only dependency.

Install them with npm: `npm install nnsplit`

### Usage

TBD

## Rust Usage

### Installation

Add NNSplit as a dependency to your `Cargo.toml`:

```toml
# ...

[dependencies.nnsplit]
version = "<version>"
features = ["model-loader", "tch-rs-backend"] # to automatically download pretrained models and to use tch-rs for inference, respectively

# ...
```

### Usage

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let splitter =
        nnsplit::NNSplit::load("en", tch::Device::Cpu, nnsplit::NNSplitOptions::default())?;

    let input: Vec<&str> = vec!["This is a test! This is another test."];
    let splits = &splitter.split(&input)[0];

    for sentence in splits.iter() {
        println!("{}", sentence.text());
    }

    Ok(())
}
```

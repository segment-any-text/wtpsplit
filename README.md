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

<details>
<summary>en - English</summary>
<p>
 
__Location__: [`models/en`](models/en)

__Metrics__:

|                              |   NNSplit |   Spacy (Tagger) |   Spacy (Sentencizer) |
|:-----------------------------|----------:|-----------------:|----------------------:|
| Clean                        |  0.763458 |         0.859013 |            0.820934   |
| Partial punctuation          |  0.486206 |         0.486595 |            0.249753   |
| Partial case                 |  0.768897 |         0.831067 |            0.819679   |
| Partial punctuation and case |  0.438204 |         0.4339   |            0.249873   |
| No punctuation and case      |  0.141555 |         0.151777 |            0.00463281 |

</p>
</details>

<details>
<summary>de - German</summary>
<p>
   
__Location__: [`models/de`](models/de)

__Metrics__:

|                              |   NNSplit |   Spacy (Tagger) |   Spacy (Sentencizer) |
|:-----------------------------|----------:|-----------------:|----------------------:|
| Clean                        |  0.795678 |        0.833368  |            0.878471   |
| Partial punctuation          |  0.532773 |        0.426458  |            0.266312   |
| Partial case                 |  0.803658 |        0.792839  |            0.876678   |
| Partial punctuation and case |  0.527035 |        0.377201  |            0.26697    |
| No punctuation and case      |  0.293959 |        0.0952267 |            0.00756195 |

</p>
</details>

<details>
   <summary>tr - Turkish</summary>
<p>

__Location__: [`models/tr`](models/tr)

__Metrics__:

|                              |   NNSplit |   Spacy (Tagger) |   Spacy (Sentencizer) |
|:-----------------------------|----------:|-----------------:|----------------------:|
| Clean                        |  0.8733   |                - |            0.918164   |
| Partial punctuation          |  0.632185 |                - |            0.274083   |
| Partial case                 |  0.877694 |                - |            0.917446   |
| Partial punctuation and case |  0.573482 |                - |            0.274352   |
| No punctuation and case      |  0.243955 |                - |            0.00364647 |

</p>
</details>

<details>
<summary>fr - French</summary>
<p>
 
__Location__: [`models/fr`](models/fr)

__Metrics__:

|                              |   NNSplit |   Spacy (Tagger) |   Spacy (Sentencizer) |
|:-----------------------------|----------:|-----------------:|----------------------:|
| Clean                        |  0.885584 |        0.903697  |            0.896942   |
| Partial punctuation          |  0.66587  |        0.382312  |            0.267478   |
| Partial case                 |  0.887438 |        0.876797  |            0.897211   |
| Partial punctuation and case |  0.580686 |        0.34492   |            0.267926   |
| No punctuation and case      |  0.251696 |        0.0473742 |            0.00298891 |

</p>
</details>

<details>
<summary>no - Norwegian</summary>
<p>
   
__Location__: [`models/no`](models/no)

__Metrics__:

|                              |   NNSplit |   Spacy (Tagger) |   Spacy (Sentencizer) |
|:-----------------------------|----------:|-----------------:|----------------------:|
| Clean                        |  0.850256 |         0.93792  |            0.878859   |
| Partial punctuation          |  0.623128 |         0.442299 |            0.263921   |
| Partial case                 |  0.847655 |         0.910273 |            0.877395   |
| Partial punctuation and case |  0.526556 |         0.377141 |            0.26413    |
| No punctuation and case      |  0.195445 |         0.060107 |            0.00472248 |

</p>
</details>

<details>
<summary>sv - Swedish</summary>
<p>
   
__Location__: [`models/sv`](models/sv)

__Metrics__:

|                              |   NNSplit |   Spacy (Tagger) |   Spacy (Sentencizer) |
|:-----------------------------|----------:|-----------------:|----------------------:|
| Clean                        |  0.831306 |                - |            0.873121   |
| Partial punctuation          |  0.587172 |                - |            0.262038   |
| Partial case                 |  0.836716 |                - |            0.87339    |
| Partial punctuation and case |  0.51484  |                - |            0.262217   |
| No punctuation and case      |  0.206952 |                - |            0.00352692 |

</p>
</details>

<details>
<summary>zh - Chinese</summary>
<p>
   
__Location__: [`models/zh`](models/zh)

__Metrics__:

|                              |   NNSplit |   Spacy (Tagger) |   Spacy (Sentencizer) |
|:-----------------------------|----------:|-----------------:|----------------------:|
| Clean                        |  0.219715 |         0.236004 |              0.186478 |
| Partial punctuation          |  0.184774 |         0.21033  |              0.133903 |
| Partial case                 |  0.221538 |         0.235706 |              0.186568 |
| Partial punctuation and case |  0.185432 |         0.210449 |              0.133993 |
| No punctuation and case      |  0.147383 |         0.198284 |              0.107093 |

</p>
</details>

## Python Usage

### Installation

NNSplit has onnxruntime as the only dependency.

Install NNSplit with pip: `pip install nnsplit`

To enable GPU support, install onnxruntime-gpu: `pip install onnxruntime-gpu`.

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

The Javascript API has no method `.load(model_name)` to load a pretrained model. Instead the path to a model in your file system (in Node.js) or accessable via `fetch` (in the browser) has to be given as first argument to `NNSplit.new`. See [models](models) to download the `model.onnx` files for the pretrained models.

#### Node.js

```js
const nnsplit = require("nnsplit");

async function run() {
    const splitter = await nnsplit.NNSplit.new("path/to/model.onnx");

    let splits = (await splitter.split(["This is a test This is another test."]))[0];
    console.log(splits.parts.map((x) => x.text)); // to log sentences, or x.parts to get the smaller subsplits
}

run()
```

#### Browser

NNSplit in the browser currently only works with a bundler and has to be imported asynchronously. API is the same as in Node.js. See [bindings/javascript/dev_server](bindings/javascript/dev_server) for a full example.

## Rust Usage

### Installation

Add NNSplit as a dependency to your `Cargo.toml`:

```toml
# ...

[dependencies.nnsplit]
version = "<version>"
features = ["model-loader", "tract-backend"] # to automatically download pretrained models and to use tract for inference, respectively

# ...
```

### Usage

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
```

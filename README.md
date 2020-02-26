# NNSplit

[![PyPI](https://img.shields.io/pypi/v/nnsplit)](https://pypi.org/project/nnsplit/)
[![Crates.io](https://img.shields.io/crates/v/nnsplit)](https://crates.io/crates/nnsplit)
[![npm](https://img.shields.io/npm/v/nnsplit)](https://www.npmjs.com/package/nnsplit)
![CI](https://github.com/bminixhofer/nnsplit/workflows/CI/badge.svg)
![License](https://img.shields.io/github/license/bminixhofer/nnsplit)

Fast, robust sentence splitting with bindings for Python, Rust and Javascript and pretrained models for English and German.

[Browser demo.](https://bminixhofer.github.io/nnsplit/js_lib/example/)

### Features

- __Robust__: Does not depend on proper punctuation and casing to split text into sentences.
- __Small__: NNSplit uses a character-level LSTM, so weights are very small (~ __350 kB__) which makes it easy to run in the browser.
- __Portable__: Models are trained in Python, but inference can be done from Javascript, Rust and Python.
- __Fast__: Can run on your GPU to __split 100k paragraphs__ from wikipedia in __50 seconds__. <sub>With RTX 2080 TI and i5 8700k. Paragraphs have an average length of ~ 800 characters. See `benchmark.ipynb` for the code.</sub>

## Python Usage

### Installation

NNSplit has PyTorch as the only dependency.

Install it with pip: `pip install nnsplit`

### Usage

```python
>>> from nnsplit import NNSplit
>>> splitter = NNSplit("en")
# NNSplit does not depend on proper punctuation and casing to split sentences
>>> splitter.split(["This is a test This is another test."])
[[[Token(text='This', whitespace=' '),
   Token(text='is', whitespace=' '),
   Token(text='a', whitespace=' '),
   Token(text='test', whitespace=' ')],
  [Token(text='This', whitespace=' '),
   Token(text='is', whitespace=' '),
   Token(text='another', whitespace=' '),
   Token(text='test', whitespace=''),
   Token(text='.', whitespace='')]]]
```

Models for German (`NNSplit("de")`) and English (`NNSplit("en")`) come prepackaged with NNSplit. Alternatively, you can also load your own model:

```python
import torch
model = torch.jit.load("/path/to/your/model.pt") # a regular nn.Module works too

splitter = NNSplit(model)
```

See the [Python README](./python_lib/README.md) for more information.

## Javascript Usage

### Installation

The Javascript bindings for NNSplit have TensorFlow.js as the only dependency.

Install them with npm: `npm install nnsplit`

### Usage


```javascript
>>> const NNSplit = require("nnsplit");
// pass URL to the model.json, see https://www.tensorflow.org/js/tutorials/conversion/import_keras#step_2_load_the_model_into_tensorflowjs for details
>>> const splitter = NNSplit("path/to/model.json");
>>> await splitter.split(["This is a test This is another test."]);
[
  [
    {
      "text": "This",
      "whitespace": " "
    },
    {
      "text": "is",
      "whitespace": " "
    },
    {
      "text": "a",
      "whitespace": " "
    },
    {
      "text": "test",
      "whitespace": " "
    }
  ],
  [
    {
      "text": "This",
      "whitespace": " "
    },
    {
      "text": "is",
      "whitespace": " "
    },
    {
      "text": "another",
      "whitespace": " "
    },
    {
      "text": "test",
      "whitespace": ""
    },
    {
      "text": ".",
      "whitespace": ""
    }
  ]
]
```

Note: when running NNSplit from Node.js, you'll have to manually import `@tensorflow/tfjs-node` before instantiating `NNSplit`.

```javascript
require("@tensorflow/tfjs-node");
const NNSplit = require("nnsplit");
```

For size reasons, the Javascript bindings do note come prepackaged with any models. Instead, download models from the Github Repo:

| Model Name  |                               |
| ----------- | ----------------------------- |
| __en__      | [Path](./data/en/tfjs_model)  |
| __de__      | [Path](./data/de/tfjs_model)  |


See the [Javascript README](./js_lib/README.md) for more information.

## Rust Usage

### Installation

Add NNSplit as a dependency to your `Cargo.toml`:

```toml
[dependencies]
# ...
nnsplit = "<version>"
# ...
```

### Usage

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

See the [Rust README](./rust_lib/README.md) for more information.

## Why?

I developed NNSplit for a side project where I am working on Neural Text Correction. In Neural Text Correction, it makes the most sense to me to have sentence-level input, rather than entire paragraphs.
In general, NNSplit might be useful for:
- NLP projects where sentence-level input is needed.
- For feature engineering (# of sentences, how sentences start, etc.)
- As inspiration for neural networks that work everywhere. NNSplit has bindings for Python, Rust and Javascript. But the code is really simple, so it is easy to use it as a template for other projects.

## How it works

NNSplit uses wikipedia dumps in the [Linguatools format](https://linguatools.org/tools/corpora/wikipedia-monolingual-corpora/) to train.
1. ) Paragraphs are extracted from the dump.
2. ) Split the paragraphs into tokens and sentences using a very accurate existing rule based sentencizer and tokenizer: [SoMaJo](https://github.com/tsproisl/SoMaJo).
3. ) With some probability, words at the start of a sentence are converted from uppercase to lowercase, and dots at the end of a sentence are removed. __This is the step that allows NNSplit to be more tolerant to errors than SoMaJo. For a rule-based system, it is nearly impossible to split sentences that don't have proper separation in the form of punctuation and casing. NNSplit solves this problem__.
4. ) Multiple "cuts" with a fixed length (default 500 characters) are extracted from a paragraph. This makes NNSplit invariant to input length. 
5. ) A simple sequence labeling RNN is trained to predict the two labels for each character. Because the NN works on character-level, embedding sizes are very small.
6. ) At inference time, the input text is split into multiple cuts with the same length of 500 characters so that the entire text is covered. NNSplit predicts each cut separately. The predictions are then averaged together for the final result.

### Training

![How NNSplit works](https://user-images.githubusercontent.com/13353204/73847685-0f8c5180-4827-11ea-8cfb-9d859715c767.png)

### Inference

![How NNSplit inference works](https://user-images.githubusercontent.com/13353204/75252193-6fe63180-57dc-11ea-85f4-7fe3b2b7ce7b.png)

## Evaluation

It is not trivial to evaluate NNSplit since I do not have a dataset of human-annotated data to use as ground truth. What can be done is just reporting metrics on some held out data from the auto-annotated data used for training.

See F1 Score, Precision and Recall averaged over the predictions for every character at threshold 0.5 below on 1.2M held out text cuts.

__Tokenization__

| Model Name  | F1@0.5 | Precision@0.5 | Recall@0.5 |
| ----------- | ------ | ------------- | ---------- |
| __en__      | 0.999  | 0.999         | 0.999      |
| __de__      | 0.999  | 0.999         | 0.999      |

__Sentence Splitting__

| Model Name  | F1@0.5 | Precision@0.5 | Recall@0.5 |
| ----------- | ------ | ------------- | ---------- |
| __en__      | 0.963  | 0.948         | 0.979      |
| __de__      | 0.978  | 0.969         | 0.988      |

These metrics are __not__ comparable to human-level sentence splitting. [SoMaJo](https://github.com/tsproisl/SoMaJo), the tool used to annotate paragraphs, is a good tool though so I do consider the results to be solid.

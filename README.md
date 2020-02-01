# nnsplit

![PyPI](https://img.shields.io/pypi/v/nnsplit)
![Crates.io](https://img.shields.io/crates/v/nnsplit)
![npm](https://img.shields.io/npm/v/nnsplit)
![GitHub](https://img.shields.io/github/license/bminixhofer/nnsplit)

__This project is a work in progress, I do not recommend using it yet.__

Fast, robust sentence splitting with bindings for Python, Rust and Javascript with pretrained models for English and German.

## Usage

### Python

#### Installation

NNSplit has PyTorch as the only dependency.

```pip install nnsplit```

#### Usage

```python
>>> from nnsplit import NNSplit

>>> splitter = NNSplit("de")
# NNSplit does not depend on proper punctuation and casing to split sentences
>>> splitter.split(["Das ist ein Test Das ist noch ein Test."])
[[[Token(text='Das', whitespace=' '),
   Token(text='ist', whitespace=' '),
   Token(text='ein', whitespace=' '),
   Token(text='Test', whitespace=' ')],
  [Token(text='Das', whitespace=' '),
   Token(text='ist', whitespace=' '),
   Token(text='noch', whitespace=' '),
   Token(text='ein', whitespace=' '),
   Token(text='Test', whitespace=''),
   Token(text='.', whitespace='')]]]
```

### Javascript

__TODO__

### Rust

__TODO__

## Why?

I developed NNSplit for a side project where I am working on Neural Text Correction. In Neural Text Correction, it makes the most sense to me to have sentence-level input, rather than an entire paragraph.
In general, NNSplit might be useful for:
- NLP projects where sentence-level input is needed.
- For feature engineering (# of sentences, how sentences start, etc.)
- As inspiration for neural networks that work everywhere. NNSplit has bindings for Python, Rust and Javascript. But the code is really simple, so it is easy to use it as a template for other projects.

## How it works

NNSplit uses wikipedia dumps in the [Linguatools format](https://linguatools.org/tools/corpora/wikipedia-monolingual-corpora/) to train.
1. ) Paragraphs are extracted from the dump.
2. ) Every character in the paragraph is labeled with 2 numbers: 1. whether it is the end of a token, 2. whether it is the end of a sentence. This sentence annotation is done using a very accurate existing rule based sentencizer and tokenizer: [SoMaJo](https://github.com/tsproisl/SoMaJo).
3. ) With some chance, words at the start of a sentence are converted from uppercase to lowercase, and dots at the end of a sentence are removed. __This is the step that allows NNSplit to be more tolerant to errors than SoMaJo. For a rule-based system, it is nearly impossible to split sentences that don't have proper separation in the form of punctuation and casing. NNSplit solves this problem__.
4. ) Multiple "cuts" with a fixed length (default 100 characters) are extracted from a paragraph. This makes NNSplit invariant to input length. 
5. ) A simple sequence labeling RNN is trained to predict the two labels for each character. Because the NN works on character-level, embedding sizes are very small (entire weights are ~ 300kB).
6. ) At inference time, the input text is split into multiple cuts with the same length of 100 characters so that the entire text is covered. NNSplit predicts each cut separately. The predictions are then averaged together for the final result.

## Training a custom model

__TODO__

# NNSplit Python Bindings

![PyPI](https://img.shields.io/pypi/v/nnsplit)
![CI](https://github.com/bminixhofer/nnsplit/workflows/CI/badge.svg)
![License](https://img.shields.io/github/license/bminixhofer/nnsplit)

Fast, robust sentence splitting with bindings for Python, Rust and Javascript and pretrained models for English and German.

## Installation

NNSplit has PyTorch as the only dependency.

Install it with pip: `pip install nnsplit`

## Usage

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

See `train.ipynb` for the code used to train the pretrained models.

## Development

NNSplit uses [Poetry](https://python-poetry.org/) for dependency management. I made a small `Makefile` to automate some steps. Take a look at the `Makefile` and run `make install`, `make build`, `make test` to install, build and test the library, respectively.
# NNSplit Python Bindings

![npm](https://img.shields.io/npm/v/nnsplit)
![CI](https://github.com/bminixhofer/nnsplit/workflows/CI/badge.svg)
![License](https://img.shields.io/github/license/bminixhofer/nnsplit)

Fast, robust sentence splitting with bindings for Python, Rust and Javascript and pretrained models for English and German.

## Installation

The Javascript Bindings for NNSplit have TensorFlow.js as the only dependency.

Install them with npm: `npm install nnsplit`

## Usage


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

| Model Name  |  |
| ------------- | ------------- |
| __en__  | [Path](https://github.com/bminixhofer/nnsplit/tree/master/data/en/tfjs_model)  |
| __de__ | [Path](https://github.com/bminixhofer/nnsplit/tree/master/data/de/tfjs_model)  |


## Development

NNSplit uses the standard Node.js + NPM stack. I made a small `Makefile` to automate some steps. Take a look at the `Makefile` and run `make install` and `make test` to install the dependencies and test the library, respectively.

There is also a small example browser app included where the text you enter is tokenized in realtime. Run `make develop` to start a local webpack dev server where the example is served or [check it out on Github Pages](https://bminixhofer.github.io/nnsplit/js_lib/example/).
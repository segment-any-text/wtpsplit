## Installation

NNSplit for Javascript has [tractjs](https://github.com/bminixhofer/tractjs) as the only dependency.

Install NNSplit with npm: 


```bash
npm install nnsplit
```

&nbsp;

## Use

The Javascript API has no method *.load(model_name)* to load a pretrained model. Instead the path to a model in your file system (in Node.js) or accessable via *fetch* (in the browser) has to be given as first argument to *NNSplit.new*. See [https://github.com/bminixhofer/nnsplit/tree/master/models](models) to download the `model.onnx` files for the pretrained models.

### Node.js

```js
const nnsplit = require("nnsplit");

async function run() {
    const splitter = await nnsplit.NNSplit.new("path/to/model.onnx");

    let splits = (await splitter.split(["This is a test This is another test."]))[0];
    console.log(splits.parts.map((x) => x.text)); // to log sentences, or x.parts to get the smaller subsplits
}

run()
```

### Browser

NNSplit in the browser currently only works with a bundler and has to be imported asynchronously. API is the same as in Node.js. See [bindings/javascript/dev_server](https://github.com/bminixhofer/nnsplit/tree/master/bindings/javascript/dev_server) for a full example.
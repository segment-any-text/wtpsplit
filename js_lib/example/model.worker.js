const NNSplit = require("../src/nnsplit.js");
const nnsplit = new NNSplit("/data/de/tfjs_model/model.json");

self.addEventListener('message', (event) => {
    nnsplit.split([event.data.text]).then((result) => {
        self.postMessage(result[0]);
    });
});
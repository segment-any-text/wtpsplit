const NNSplit = require("../src/nnsplit.js");
console.log(self.location);
const nnsplit = new NNSplit(`${self.location.origin}/nnsplit/data/de/tfjs_model/model.json`);

self.addEventListener('message', (event) => {
    nnsplit.split([event.data.text]).then((result) => {
        self.postMessage(result[0]);
    });
});
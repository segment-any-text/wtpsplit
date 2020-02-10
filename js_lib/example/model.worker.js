const NNSplit = require("../src/nnsplit.js");
const nnsplit_de = new NNSplit(`${self.location.origin}/nnsplit/data/de/tfjs_model/model.json`);
const nnsplit_en = new NNSplit(`${self.location.origin}/nnsplit/data/en/tfjs_model/model.json`);

self.addEventListener('message', (event) => {
    let nnsplit;
    switch (event.data.language) {
        case "german": nnsplit = nnsplit_de; break;
        case "english": nnsplit = nnsplit_en; break;
    }

    nnsplit.split([event.data.text]).then((result) => {
        self.postMessage(result[0]);
    });
});
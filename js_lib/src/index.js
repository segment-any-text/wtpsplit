import NNSplit from "./nnsplit.js";

async function example() {
    const nnsplit = new NNSplit("/data/de/tfjs_model/model.json");
    console.log(await nnsplit.split(["Das ist ein Test.", "asdf"]));
}

example();
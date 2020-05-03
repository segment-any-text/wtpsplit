import * as nnsplit from "nnsplit";

async function run() {
    let splitter = await new nnsplit.NNSplit("/tensorflowjs_model/model.json");

    console.log(splitter.split(["Das ist ein Test."])[0]);
}

run();
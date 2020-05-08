import * as nnsplit from "nnsplit";
import * as tfl from "@tensorflow/tfjs-layers";
async function run() {
    let splitter = await new nnsplit.NNSplit("/de/model.json");

    let splits = splitter.split(["Das ist ein Test Das ist noch ein Test."])[0];
    let isExpected = splits.parts[0].text == "Das ist ein Test " && splits.parts[1].text == "Das ist noch ein Test.";
    if (!isExpected) {
        console.error("split in unexpected parts.");
    }

    console.log("cypress:success");
}

run();
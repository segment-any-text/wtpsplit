import * as nnsplit from "nnsplit";

async function test() {
  let splitter = await nnsplit.NNSplit.new("/de/model.onnx");

  let splits = await splitter.split([
    "Das ist ein Test Das ist noch ein Test.",
  ]);
  splits = splits[0];
  let isExpected =
    splits.parts[0].text == "Das ist ein Test " &&
    splits.parts[1].text == "Das ist noch ein Test.";
  if (!isExpected) {
    console.error("split in unexpected parts.");
  }

  console.log("cypress:success");
}

async function benchmark() { }

export { test, benchmark };

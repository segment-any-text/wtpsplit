const chai = require("chai");
const NNSplit = require("../src/nnsplit.js");
require("@tensorflow/tfjs-node");

describe("NNSplit", function () {
    it("should load model config correctly", async function () {
        const modelPath = __dirname + "/../example/data/de/tfjs_model/model.json";
        const splitter = new NNSplit("file://" + modelPath);
        await splitter.model;
    });
    it("should return an empty array when given an empty array", async function () {
        const modelPath = __dirname + "/../example/data/de/tfjs_model/model.json";
        const splitter = new NNSplit("file://" + modelPath);
        chai.expect(await splitter.split([])).to.deep.equal([]);
    });
    it("should split german sentences and tokens correctly", async function () {
        const modelPath = __dirname + "/../example/data/de/tfjs_model/model.json";
        const splitter = new NNSplit("file://" + modelPath);

        const result = await splitter.split(["Das ist ein Test Das ist noch ein Test."]);
        chai.expect(result).to.deep.equal(
            [ // texts
                [ // sentences
                    [ // tokens
                        {
                            text: "Das",
                            whitespace: " ",
                        },
                        {
                            text: "ist",
                            whitespace: " ",
                        },
                        {
                            text: "ein",
                            whitespace: " ",
                        },
                        {
                            text: "Test",
                            whitespace: " ",
                        }
                    ],
                    [
                        {
                            text: "Das",
                            whitespace: " ",
                        },
                        {
                            text: "ist",
                            whitespace: " ",
                        },
                        {
                            text: "noch",
                            whitespace: " ",
                        },
                        {
                            text: "ein",
                            whitespace: " ",
                        },
                        {
                            text: "Test",
                            whitespace: "",
                        },
                        {
                            text: ".",
                            whitespace: "",
                        }
                    ]
                ]
            ]
        );
    });
});

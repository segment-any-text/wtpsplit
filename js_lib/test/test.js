const chai = require("chai");
const NNSplit = require("../src/nnsplit.js");
require("@tensorflow/tfjs-node");

describe("NNSplit", function () {
    it("should load model config correctly", async function () {
        let modelPath = __dirname + "/../example/nnsplit/data/de/tfjs_model/model.json";
        let splitter = new NNSplit("file://" + modelPath);
        await splitter.model;

        modelPath = __dirname + "/../example/nnsplit/data/en/tfjs_model/model.json";
        splitter = new NNSplit("file://" + modelPath);
        await splitter.model;
    });
    it("should return an empty array when given an empty array", async function () {
        const modelPath = __dirname + "/../example/nnsplit/data/de/tfjs_model/model.json";
        const splitter = new NNSplit("file://" + modelPath);
        chai.expect(await splitter.split([])).to.deep.equal([]);
    });
    it("should split german sentences and tokens correctly", async function () {
        const modelPath = __dirname + "/../example/nnsplit/data/de/tfjs_model/model.json";
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
                        },
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
    it("should split english sentences and tokens correctly", async function () {
        const modelPath = __dirname + "/../example/nnsplit/data/en/tfjs_model/model.json";
        const splitter = new NNSplit("file://" + modelPath);

        const result = await splitter.split(["This is a test This is another test."]);
        chai.expect(result).to.deep.equal(
            [ // texts
                [ // sentences
                    [ // tokens
                        {
                            text: "This",
                            whitespace: " ",
                        },
                        {
                            text: "is",
                            whitespace: " ",
                        },
                        {
                            text: "a",
                            whitespace: " ",
                        },
                        {
                            text: "test",
                            whitespace: " ",
                        },
                    ],
                    [
                        {
                            text: "This",
                            whitespace: " ",
                        },
                        {
                            text: "is",
                            whitespace: " ",
                        },
                        {
                            text: "another",
                            whitespace: " ",
                        },
                        {
                            text: "test",
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
    it("should split long text correctly", async function () {
        const modelPath = __dirname + "/../example/nnsplit/data/en/tfjs_model/model.json";
        const splitter = new NNSplit("file://" + modelPath);

        const result = await splitter.split(["Fast, robust sentence splitting with bindings for Python, Rust and Javascript Punctuation is not necessary to split sentences correctly sometimes even incorrect case is split correctly."]);
        chai.expect(result).to.deep.equal(
            [ // texts
                [ // sentences
                    [ // tokens
                        {
                            text: "Fast",
                            whitespace: "",
                        },
                        {
                            text: ",",
                            whitespace: " ",
                        },
                        {
                            text: "robust",
                            whitespace: " ",
                        },
                        {
                            text: "sentence",
                            whitespace: " ",
                        },
                        {
                            text: "splitting",
                            whitespace: " ",
                        },
                        {
                            text: "with",
                            whitespace: " ",
                        },
                        {
                            text: "bindings",
                            whitespace: " ",
                        },
                        {
                            text: "for",
                            whitespace: " ",
                        },
                        {
                            text: "Python",
                            whitespace: "",
                        },
                        {
                            text: ",",
                            whitespace: " ",
                        },
                        {
                            text: "Rust",
                            whitespace: " ",
                        },
                        {
                            text: "and",
                            whitespace: " ",
                        },
                        {
                            text: "Javascript",
                            whitespace: " ",
                        }
                    ],
                    [
                        {
                            text: "Punctuation",
                            whitespace: " ",
                        },
                        {
                            text: "is",
                            whitespace: " ",
                        },
                        {
                            text: "not",
                            whitespace: " ",
                        },
                        {
                            text: "necessary",
                            whitespace: " ",
                        },
                        {
                            text: "to",
                            whitespace: " ",
                        },
                        {
                            text: "split",
                            whitespace: " ",
                        },
                        {
                            text: "sentences",
                            whitespace: " ",
                        },
                        {
                            text: "correctly",
                            whitespace: " ",
                        }
                    ],
                    [
                        {
                            text: "sometimes",
                            whitespace: " ",
                        },
                        {
                            text: "even",
                            whitespace: " ",
                        },
                        {
                            text: "incorrect",
                            whitespace: " ",
                        },
                        {
                            text: "case",
                            whitespace: " ",
                        },
                        {
                            text: "is",
                            whitespace: " ",
                        },
                        {
                            text: "split",
                            whitespace: " ",
                        },
                        {
                            text: "correctly",
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

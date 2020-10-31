const nnsplit = require('../pkg');
const assert = require('assert');

describe('NNSplit', function () {
    it('should instantiate given valid .onnx path', async function () {
        let splitter = await nnsplit.NNSplit.new("../../models/de/model.onnx");
    });
    it('should raise an error given an invald model', async function () {
        assert.rejects(nnsplit.NNSplit.new("package.json"));
    });
    it('should split a simple text correctly', async function () {
        let splitter = await nnsplit.NNSplit.new("../../models/de/model.onnx");
        let split = await splitter.split([
            "Das ist ein Test Das ist noch ein Test.",
        ]);

        assert.deepStrictEqual(split[0].parts.map((x) => x.text), ["Das ist ein Test ", "Das ist noch ein Test."]);
    });
    it('should be able to return names of split levels', async function () {
        let splitter = await nnsplit.NNSplit.new("../../models/de/model.onnx");

        assert.deepStrictEqual(splitter.getLevels(), ["Sentence", "Token", "_Whitespace", "Compound constituent"])
    });
    it('should be able to split long text correctly', async function () {
        let splitter = await nnsplit.NNSplit.new("../../models/de/model.onnx");
        let split = await splitter.split([
            "Eine Vernetzung von Neuronen im Nervensystem eines Lebewesens darstellen. ".repeat(20),
        ]);

        assert.deepStrictEqual(split[0].parts.map((x) => x.text), Array(20).fill("Eine Vernetzung von Neuronen im Nervensystem eines Lebewesens darstellen. "));
    });
});

const tf = require("@tensorflow/tfjs-core");
const tfl = require("@tensorflow/tfjs-layers");

const CUT_LENGTH = 100;

function textToId(char) {
    const x = char.charCodeAt(0);
    return x <= 127 ? x + 2 : 1;
}

class Token {
    constructor(text, whitespace) {
        this.text = text;
        this.whitespace = whitespace;
    }
}

function get_token(text) {
    let lastCharIndex = Array.from(text).reverse().findIndex((x) => x.trim().length != 0);
    lastCharIndex = text.length - lastCharIndex;

    return new Token(text.slice(0, lastCharIndex), text.slice(lastCharIndex));
}

class NNSplit {
    constructor(modelPath, threshold = 0.5, stride = 50, cutLength = CUT_LENGTH) {
        this.threshold = threshold;
        this.stride = stride;
        this.cutLength = cutLength;

        this.model = tfl.loadLayersModel(modelPath);
    }

    async split(texts) {
        if (texts.length === 0) {
            return [];
        }

        const allInputs = [];
        const allIdx = [];
        const nCutsPerText = [];

        texts.forEach((text) => {
            const inputs = Array.from(text).map(textToId);

            while (inputs.length < this.cutLength) {
                inputs.push(0);
            }

            let start = 0;
            let end = -1;
            let i = 0;

            while (end != inputs.length) {
                end = Math.min(start + this.cutLength, inputs.length);
                start = end - this.cutLength;

                const idx = [start, end];
                allInputs.push(inputs.slice(start, end));
                allIdx.push(idx);

                start += this.stride;
                i += 1;
            }

            nCutsPerText.push(i);
        });

        const batchedInputs = tf.tensor(allInputs);
        // TODO: batch properly
        let preds = (await this.model).predict(batchedInputs).sigmoid();
        preds = await preds.buffer();

        const allAvgPreds = texts.map((x) => [new Float32Array(x.length), new Float32Array(x.length)]);
        const allAvgPredCounts = texts.map((x) => new Uint8Array(x.length));
        let currentText = 0;
        let currentI = 0;

        for (let i = 0; i < allIdx.length; i++) {
            let [start, end] = allIdx[i];

            for (let j = start; j < end; j++) {
                allAvgPreds[currentText][0][j] += preds.get(i, j, 0);
                allAvgPreds[currentText][1][j] += preds.get(i, j, 1);
                allAvgPredCounts[currentText][j] += 1;
            }

            currentI += 1;

            if (currentI === nCutsPerText[currentText]) {
                for (let j = 0; j < allAvgPredCounts[currentText].length; j++) {
                    allAvgPreds[currentText][0][j] /= allAvgPredCounts[currentText][j];
                }

                currentI = 0;
                currentText += 1;
            }
        }

        const tokenizedTexts = [];

        allAvgPreds.forEach((avgPreds, index) => {
            const sentences = [];
            let tokens = [];
            let token = "";

            for (let i = 0; i < texts[index].length; i++) {
                token += texts[index][i];

                if (avgPreds[0][i] > this.threshold || avgPreds[1][i] > this.threshold) {
                    tokens.push(get_token(token));
                    token = "";
                }

                if (avgPreds[1][i] > this.threshold) {
                    sentences.push(tokens);
                    tokens = [];
                }
            }

            if (token.length > 0) {
                tokens.push(get_token(token));
            }

            if (tokens.length > 0) {
                sentences.push(tokens);
            }

            tokenizedTexts.push(sentences);
        });

        return tokenizedTexts;
    }
}

module.exports = NNSplit;
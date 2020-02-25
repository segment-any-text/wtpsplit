const tf = require("@tensorflow/tfjs-core");
const tfl = require("@tensorflow/tfjs-layers");

const CUT_LENGTH = 500;

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
    /**
     * An NNSplit sentencizer and tokenizer.
     * 
     * @param {string} modelPath - path to the model.json from which to load the TensorFlow.js model.
     * @param {float} [threshold=0.1] - Cutoff above which predictions will be considered as 1. 
     * @param {*} stride - How much to move the window after each prediction. Comparable to stride in a 1d convolution.
     * @param {*} cutLength - The number of characters in each cut.
     */
    constructor(modelPath, threshold = 0.1, stride = Math.floor(CUT_LENGTH / 2), cutLength = CUT_LENGTH) {
        this.threshold = threshold;
        this.stride = stride;
        this.cutLength = cutLength;
        this.padding = 5;
        this.model = tfl.loadLayersModel(modelPath);
    }

    async _get_raw_preds(texts, batchSize) {
        const allInputs = [];
        const allIdx = [];
        const nCutsPerText = [];
        const maxTextLength = this.padding * 2 + Math.max(...texts.map((x) => x.length));
        const optCutLength = Math.min(this.cutLength, maxTextLength);

        texts.forEach((text) => {
            // char -> id and pad on the left and right
            const inputs = Array.from(text).map(textToId);
            const pad = Array(this.padding).fill(0);
            inputs.unshift(...pad);
            inputs.push(...pad);

            let start = 0;
            let end = -1;
            let i = 0;

            while (end != inputs.length) {
                end = Math.min(start + optCutLength, inputs.length);
                start = end - optCutLength;

                const idx = [start, end];
                allInputs.push(inputs.slice(start, end));
                allIdx.push(idx);

                start += this.stride;
                i += 1;
            }

            nCutsPerText.push(i);
        });

        const batchedInputs = tf.tensor(allInputs);
        let preds = (await this.model).predict(batchedInputs, { batchSize }).sigmoid();
        preds = await preds.buffer();

        return [preds, allIdx, nCutsPerText];
    }

    _average_preds(texts, preds, allIdx, nCutsPerText) {
        const allAvgPreds = texts.map((x) => [new Float32Array(x.length), new Float32Array(x.length)]);
        const allAvgPredCounts = texts.map((x) => new Uint8Array(x.length));
        let currentText = 0;
        let currentI = 0;

        for (let i = 0; i < allIdx.length; i++) {
            let [start, end] = allIdx[i];
            for (let j = start; j < end; j++) {
                if (j < this.padding || j >= texts[currentText].length + this.padding) {
                    continue;
                }

                allAvgPreds[currentText][0][j - this.padding] += preds.get(i, j - start, 0);
                allAvgPreds[currentText][1][j - this.padding] += preds.get(i, j - start, 1);
                allAvgPredCounts[currentText][j - this.padding] += 1;
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

        return allAvgPreds;
    }

    _split_text_from_preds(texts, allAvgPreds) {
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

    /**
     * Split texts into sentences and tokens.
     * 
     * @param {string[]} texts - A list of texts to split. Passing multiple texts at once allows for parallelization of the model.
     * @param {int} [batchSize=128] - Batch size with which cuts are processed by the model.
     * 
     * @return {string[][][]}
     *  - A list with the same length as `texts`.
     *  - Each element is a list of sentences.
     *  - Each sentence is a list of tokens.
     *  - Each token is a `Token` class with properties `text` and `whitespace`.
     */
    async split(texts, batchSize = 128) {
        if (texts.length === 0) {
            return [];
        }

        const [preds, allIdx, nCutsPerText] = await this._get_raw_preds(texts, batchSize);
        const allAvgPreds = this._average_preds(texts, preds, allIdx, nCutsPerText);

        return this._split_text_from_preds(texts, allAvgPreds);
    }
}

module.exports = NNSplit;
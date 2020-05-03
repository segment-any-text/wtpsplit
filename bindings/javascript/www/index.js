import * as wasm from "nnsplit";
import * as tf from "@tensorflow/tfjs-core";
import * as tfl from "@tensorflow/tfjs-layers";

async function run() {
    const model = await tfl.loadLayersModel("/tensorflowjs_model/model.json");

    function predict(blob, shape, batchSize) {
        const input = tf.tensor(blob, shape);
        const pred = model.predict(input, { batchSize });

        return [pred.dataSync(), new Uint32Array(pred.shape)];
    }

    wasm.run(predict);
}

run();
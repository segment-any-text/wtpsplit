use js_sys::{Array, Float32Array, Promise, Uint32Array, Uint8Array};
use ndarray::prelude::*;
use std::error::Error;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use nnsplit as core;

#[wasm_bindgen(module = "@tensorflow/tfjs-core")]
extern "C" {
    fn tensor(blob: Uint8Array, shape: Array) -> Tensor;
}

#[wasm_bindgen(module = "@tensorflow/tfjs-layers")]
extern "C" {
    #[wasm_bindgen(js_name = loadLayersModel)]
    fn load_layers_model(path: &str) -> Promise;
}

#[wasm_bindgen]
extern "C" {
    type Tensor;

    #[wasm_bindgen(method)]
    fn dataSync(this: &Tensor) -> Float32Array;

    #[wasm_bindgen(method, getter)]
    fn shape(this: &Tensor) -> Uint32Array;
}

#[wasm_bindgen]
extern "C" {
    type Model;

    #[wasm_bindgen(method)]
    fn predict(this: &Model, tensor: Tensor) -> Tensor;
}
pub struct TensorflowJSBackend {
    model: Model,
}

impl TensorflowJSBackend {
    pub async fn new(model_path: &str) -> Self {
        let model = JsFuture::from(load_layers_model(model_path))
            .await
            .unwrap_throw()
            .into();

        TensorflowJSBackend { model }
    }
}

impl core::Backend for TensorflowJSBackend {
    fn predict(&self, input: Array2<u8>, batch_size: usize) -> Result<Array3<f32>, Box<dyn Error>> {
        let shape: Array = input
            .shape()
            .iter()
            .map(|x| JsValue::from(*x as u32))
            .collect();

        let tensor = tensor(
            input
                .as_slice()
                .ok_or("converting ndarray to slice failed (likely not contiguous)")?
                .into(),
            shape,
        );
        let pred = self.model.predict(tensor);

        let shape = pred.shape();
        let shape = shape.to_vec();
        assert!(shape.len() == 3);
        let shape = (shape[0] as usize, shape[1] as usize, shape[2] as usize);

        let blob = pred.dataSync();
        let preds = Array3::from_shape_vec(shape, blob.to_vec())?;

        Ok(preds)
    }
}

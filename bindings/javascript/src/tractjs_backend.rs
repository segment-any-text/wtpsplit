use js_sys::{Array, Float32Array, Promise, Uint32Array, Uint8Array};
use ndarray::prelude::*;
use serde_derive::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

#[derive(Serialize, Deserialize)]
struct ModelLoadArgs {
    optimize: bool,
}

#[wasm_bindgen(module = "tractjs")]
extern "C" {
    type Model;

    #[wasm_bindgen(static_method_of = Model)]
    fn load(path: &str, options: JsValue) -> Promise;

    #[wasm_bindgen(method)]
    fn predict_one(this: &Model, input: Tensor) -> Promise;
}

#[wasm_bindgen(module = "tractjs")]
extern "C" {
    type Tensor;

    #[wasm_bindgen(constructor)]
    fn new(data: JsValue, shape: Array) -> Tensor;

    #[wasm_bindgen(method, getter)]
    fn data(this: &Tensor) -> JsValue;

    #[wasm_bindgen(method, getter)]
    fn shape(this: &Tensor) -> Uint32Array;
}

pub struct TractJSBackend {
    model: Model,
}

impl TractJSBackend {
    pub async fn new(model_path: &str) -> Result<Self, JsValue> {
        let model: Model = JsFuture::from(Model::load(
            model_path,
            JsValue::from_serde(&ModelLoadArgs { optimize: false }).unwrap(),
        ))
        .await?
        .into();

        Ok(TractJSBackend { model })
    }

    pub async fn predict(&self, input: Array2<u8>) -> Result<Array3<f32>, JsValue> {
        let shape: Array = input
            .shape()
            .iter()
            .map(|x| JsValue::from(*x as u32))
            .collect();

        let tensor = Tensor::new(
            Uint8Array::from(
                input
                    .as_slice()
                    .ok_or("converting ndarray to slice failed (likely not contiguous)")?,
            )
            .into(),
            shape,
        );

        let pred: Tensor = JsFuture::from(self.model.predict_one(tensor)).await?.into();

        let shape = pred.shape();
        let shape = shape.to_vec();
        assert!(shape.len() == 3);
        let shape = (shape[0] as usize, shape[1] as usize, shape[2] as usize);

        let data: Float32Array = pred.data().into();
        let mut preds =
            Array3::from_shape_vec(shape, data.to_vec()).map_err(|_| "Array conversion error")?;

        // sigmoid
        preds.mapv_inplace(|x| 1f32 / (1f32 + (-x).exp()));

        Ok(preds)
    }
}

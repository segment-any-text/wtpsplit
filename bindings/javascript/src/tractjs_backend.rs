use futures::future::join_all;
use js_sys::{Array, Float32Array, Promise, Uint32Array, Uint8Array};
use ndarray::prelude::*;
use serde_derive::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{cmp, collections::HashMap};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

#[derive(Serialize, Deserialize)]
struct ModelLoadArgs {
    #[serde(rename = "inputFacts")]
    input_facts: HashMap<usize, Value>,
}

#[wasm_bindgen(module = "tractjs")]
extern "C" {
    type Model;

    #[wasm_bindgen(static_method_of = Model)]
    fn load(path: &str, options: JsValue) -> Promise;

    #[wasm_bindgen(method)]
    fn predict_one(this: &Model, input: Tensor) -> Promise;

    #[wasm_bindgen(method)]
    fn get_metadata(this: &Model) -> Promise;

    #[wasm_bindgen(method)]
    fn destroy(this: &Model) -> Promise;
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

#[wasm_bindgen(module = "tractjs")]
extern "C" {
    fn terminate() -> ();
}

pub struct TractJSBackend {
    model: Model,
    batch_size: usize,
}

impl TractJSBackend {
    pub async fn new(
        model_path: &str,
        length_divisor: usize,
        batch_size: usize,
    ) -> Result<Self, JsValue> {
        let mut input_facts = HashMap::new();
        input_facts.insert(
            0,
            json!(["uint8", [1, {
                "id": "s",
                "slope": length_divisor,
                "intercept": 0,
            }]]),
        );

        let model: Model = JsFuture::from(Model::load(
            model_path,
            JsValue::from_serde(&ModelLoadArgs { input_facts }).unwrap(),
        ))
        .await?
        .into();

        Ok(TractJSBackend { model, batch_size })
    }

    pub async fn predict(&self, input: Array2<u8>) -> Result<Array3<f32>, JsValue> {
        let preds = (0..input.shape()[0])
            .step_by(self.batch_size)
            .map(|start| {
                let end = cmp::min(start + self.batch_size, input.shape()[0]);
                let actual_batch_size = end - start;

                let shape: Array = vec![
                    JsValue::from(actual_batch_size as u32),
                    JsValue::from(input.shape()[1] as u32),
                ]
                .into_iter()
                .collect();

                let tensor = Tensor::new(
                    Uint8Array::from(input.slice(s![start..end, ..]).as_slice().expect_throw(
                        "converting ndarray to slice failed (likely not contiguous)",
                    ))
                    .into(),
                    shape,
                );

                JsFuture::from(self.model.predict_one(tensor))
            })
            .collect::<Vec<_>>();
        let preds = join_all(preds)
            .await
            .into_iter()
            .map(|x| x.map(|value| value.into()))
            .collect::<Result<Vec<Tensor>, JsValue>>()?;

        let shape = preds[0].shape();
        let shape = shape.to_vec();
        assert!(shape.len() == 3);
        let shape = (input.shape()[0], shape[1] as usize, shape[2] as usize);

        let data_vec = preds.into_iter().fold(Vec::new(), |mut arr, x| {
            let curr: Float32Array = x.data().into();

            arr.extend(curr.to_vec());
            arr
        });

        let mut preds =
            Array3::from_shape_vec(shape, data_vec).map_err(|_| "Array conversion error.")?;

        // sigmoid
        preds.mapv_inplace(|x| 1f32 / (1f32 + (-x).exp()));

        Ok(preds)
    }

    pub async fn get_metadata(&self) -> Result<HashMap<String, String>, JsValue> {
        let metadata: HashMap<String, String> = JsFuture::from(self.model.get_metadata())
            .await?
            .into_serde()
            .map_err(|_| "reading metadata failed")?;

        Ok(metadata)
    }
}

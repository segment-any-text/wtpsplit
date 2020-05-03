mod utils;

use js_sys::{Array, Float32Array, Function, Uint32Array, Uint8Array};
use ndarray::prelude::*;
use nnsplit as core;
use nnsplit::Backend;
use wasm_bindgen::prelude::*;

struct TensorflowJSBackend<'a> {
    predict_closure: &'a Function,
}

impl<'a> TensorflowJSBackend<'a> {
    fn new(predict_closure: &'a Function) -> TensorflowJSBackend<'a> {
        TensorflowJSBackend { predict_closure }
    }
}

impl<'a> core::Backend for TensorflowJSBackend<'a> {
    fn predict(&self, input: Array2<u8>, batch_size: usize) -> Array3<f32> {
        let this = JsValue::NULL;
        let data: Uint8Array = input.as_slice().unwrap().into();
        let shape: Array = input
            .shape()
            .iter()
            .map(|x| JsValue::from(*x as u32))
            .collect::<Array>();

        let data: Array = self
            .predict_closure
            .call3(&this, &data, &shape, &(batch_size as u32).into())
            .unwrap()
            .into();

        let shape: Uint32Array = data.pop().into();
        let shape = shape.to_vec();
        assert!(shape.len() == 3);
        let shape = (shape[0] as usize, shape[1] as usize, shape[2] as usize);

        let blob: Float32Array = data.pop().into();
        let preds = Array3::from_shape_vec(shape, blob.to_vec()).unwrap();

        preds
    }
}

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
extern "C" {
    fn alert(input: &str);
}

#[wasm_bindgen]
pub fn run(f: &Function) {
    let backend = TensorflowJSBackend::new(f);

    backend.predict(Array2::zeros([10, 1]), 32);
}

mod tensorflowjs_backend;
mod utils;

use tensorflowjs_backend::TensorflowJSBackend;
use wasm_bindgen::prelude::*;

use nnsplit as core;

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
struct NNSplit {
    inner: core::NNSplit,
}

#[wasm_bindgen]
impl NNSplit {
    #[wasm_bindgen(constructor)]
    pub async fn new(path: String) -> Self {
        let backend = TensorflowJSBackend::new(&path).await;

        NNSplit {
            inner: core::NNSplit::from_backend(
                Box::new(backend) as Box<dyn core::Backend>,
                core::NNSplitOptions::default(),
            ),
        }
    }

    pub fn split(&self, texts: Vec<JsValue>) {
        let texts: Vec<String> = texts.into_iter().map(|x| x.as_string().unwrap()).collect();
        let texts = texts.iter().map(|x| x.as_ref()).collect();
        let splits = self.inner.split(texts);

        alert(&format!("{:#?}", splits));
    }
}

mod tensorflowjs_backend;
mod utils;

use tensorflowjs_backend::TensorflowJSBackend;
use wasm_bindgen::prelude::*;

use nnsplit as core;

#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen(inspectable)]
struct Split {
    text: String,
    parts: Vec<JsValue>,
}

#[wasm_bindgen]
impl Split {
    #[wasm_bindgen(getter)]
    pub fn parts(&self) -> Vec<JsValue> {
        self.parts.iter().map(|x| x.clone()).collect()
    }

    #[wasm_bindgen(getter)]
    pub fn text(&self) -> String {
        self.text.clone()
    }
}

impl<'a> From<core::Split<'a>> for Split {
    fn from(split: core::Split) -> Self {
        match split {
            core::Split::Text(_) => panic!("text can not be converted to a Split"),
            core::Split::Split((text, split_parts)) => {
                let parts = split_parts
                    .into_iter()
                    .map(|x| match &x {
                        core::Split::Split(_) => {
                            let split: Split = x.into();
                            split.into()
                        }
                        core::Split::Text(text) => text.to_owned().into(),
                    })
                    .collect();

                Split {
                    text: text.to_owned(),
                    parts,
                }
            }
        }
    }
}

#[wasm_bindgen]
struct NNSplit {
    inner: core::NNSplit,
}

#[wasm_bindgen]
impl NNSplit {
    #[wasm_bindgen(constructor)]
    pub async fn new(path: String) -> Self {
        utils::set_panic_hook();
        let backend = TensorflowJSBackend::new(&path).await;

        NNSplit {
            inner: core::NNSplit::from_backend(
                Box::new(backend) as Box<dyn core::Backend>,
                core::NNSplitOptions::default(),
            ),
        }
    }

    pub fn split(&self, texts: Vec<JsValue>) -> Vec<JsValue> {
        let texts: Vec<String> = texts.into_iter().map(|x| x.as_string().unwrap()).collect();
        let texts = texts.iter().map(|x| x.as_ref()).collect();
        let splits = self.inner.split(texts);

        splits
            .into_iter()
            .map(|x| {
                let split: Split = x.into();
                split.into()
            })
            .collect()
    }
}

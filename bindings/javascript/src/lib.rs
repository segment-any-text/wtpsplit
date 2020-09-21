mod tractjs_backend;
mod utils;

use js_sys::Array;
use tractjs_backend::TractJSBackend;
use wasm_bindgen::prelude::*;

use nnsplit as core;

#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen(inspectable)]
pub struct Split {
    text: String,
    parts: Vec<JsValue>,
}

#[wasm_bindgen]
impl Split {
    #[wasm_bindgen(getter)]
    pub fn parts(&self) -> Vec<JsValue> {
        self.parts.to_vec()
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
pub struct NNSplit {
    backend: TractJSBackend,
    inner: core::NNSplitLogic,
}

#[wasm_bindgen]
impl NNSplit {
    #[wasm_bindgen(constructor)]
    pub fn invalid_new() -> Result<(), JsValue> {
        Err("NNSplit can't be construced directly because it is asynchronous! Please use NNSplit.new.".into())
    }

    pub async fn new(path: String, options: JsValue) -> Result<NNSplit, JsValue> {
        utils::set_panic_hook();
        let backend = TractJSBackend::new(&path).await?;

        Ok(NNSplit {
            backend,
            inner: core::NNSplitLogic::new(if options.is_undefined() || options.is_null() {
                core::NNSplitOptions::default()
            } else {
                options.into_serde().unwrap()
            }),
        })
    }

    pub async fn split(self, texts: Vec<JsValue>) -> Result<JsValue, JsValue> {
        let texts: Vec<String> = texts
            .into_iter()
            .map(|x| x.as_string().unwrap_throw())
            .collect();
        let texts: Vec<&str> = texts.iter().map(|x| x.as_ref()).collect();

        let (inputs, indices) = self.inner.get_inputs_and_indices(&texts);
        let slice_preds = self.backend.predict(inputs).await?;

        let splits = self.inner.split(&texts, slice_preds, indices);
        let splits = splits
            .into_iter()
            .map(|x| {
                let split: Split = x.into();
                split.into()
            })
            .collect::<Vec<JsValue>>();

        let array = Array::new();
        for split in &splits {
            array.push(split);
        }

        Ok(array.into())
    }
}

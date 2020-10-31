use crate::{NNSplitLogic, NNSplitOptions};
use ndarray::prelude::*;
use std::error::Error;
use tract_onnx::prelude::*;

struct TractBackend {
    model: TypedModel,
    n_outputs: usize,
    length_divisor: usize,
}

impl TractBackend {
    fn new(model: TypedModel, length_divisor: usize) -> TractResult<Self> {
        let n_outputs = if let TDim::Val(value) = model.outlet_fact(model.outputs[0])?.shape[2] {
            value as usize
        } else {
            0 // TODO: raise error here
        };

        Ok(TractBackend {
            model,
            n_outputs,
            length_divisor,
        })
    }

    fn predict(
        &self,
        input: Array2<u8>,
        _batch_size: usize,
    ) -> Result<Array3<f32>, Box<dyn Error>> {
        let input_shape = input.shape();
        let opt_model = self
            .model
            .concretize_dims(&SymbolValues::default().with(
                's'.into(),
                input_shape[1] as i64 / self.length_divisor as i64,
            ))?
            .optimize()?
            .into_runnable()?;

        let mut preds = Array3::<f32>::zeros((input_shape[0], input_shape[1], self.n_outputs));

        // currently batch size is always 1, tract does not yet support more than one streaming dimension well
        for i in 0..input_shape[0] {
            let batch_inputs: Tensor = input.slice(s![i..(i + 1), ..]).to_owned().into();

            let batch_preds = opt_model.run(tvec![batch_inputs])?.remove(0);
            let mut batch_preds: ArrayD<f32> = (*batch_preds).clone().into_array()?;

            // sigmoid
            batch_preds.mapv_inplace(|x| 1f32 / (1f32 + (-x).exp()));

            preds.slice_mut(s![i..(i + 1), .., ..]).assign(&batch_preds);
        }

        Ok(preds)
    }
}

/// Complete splitter using tract as backend.
pub struct NNSplit {
    backend: TractBackend,
    logic: NNSplitLogic,
}

impl NNSplit {
    fn type_model(model: InferenceModel, length_divisor: usize) -> TractResult<TypedModel> {
        model
            .with_input_fact(
                0,
                InferenceFact::dt_shape(
                    u8::datum_type(),
                    tvec!(1.into(), TDim::from(Symbol::from('s')) * length_divisor),
                ),
            )?
            .into_typed()?
            .declutter()
    }

    fn from_model(
        model_proto: tract_onnx::pb::ModelProto,
        options: NNSplitOptions,
    ) -> Result<Self, Box<dyn Error>> {
        let model = NNSplit::type_model(
            onnx().model_for_proto_model(&model_proto)?,
            options.length_divisor,
        )?;

        let split_sequence_string = model_proto
            .metadata_props
            .into_iter()
            .find_map(|x| {
                if x.key == "split_sequence" {
                    Some(x.value)
                } else {
                    None
                }
            })
            .ok_or("Model must contain `split_sequence` metadata key")?;

        let backend = TractBackend::new(model, options.length_divisor)?;

        Ok(NNSplit {
            backend,
            logic: NNSplitLogic::new(options, serde_json::from_str(&split_sequence_string)?),
        })
    }

    /// Create a new splitter from the given model location.
    /// # Errors
    /// * If the file can not be loaded as tract model.
    pub fn new<P: AsRef<std::path::Path>>(
        model_path: P,
        options: NNSplitOptions,
    ) -> Result<Self, Box<dyn Error>> {
        let model_proto = onnx().proto_model_for_path(model_path)?;

        NNSplit::from_model(model_proto, options)
    }

    /// Loads a built-in model. From the local cache or from the internet if it is not cached.
    #[cfg(feature = "model-loader")]
    pub fn load(model_name: &str, options: NNSplitOptions) -> Result<Self, Box<dyn Error>> {
        let mut model_data = crate::model_loader::get_resource(model_name, "model.onnx")?.0;
        let model_proto = onnx().proto_model_for_read(&mut model_data)?;

        NNSplit::from_model(model_proto, options)
    }

    /// Split a list of texts into a list of `Split` objects.
    pub fn split<'a>(&self, texts: &[&'a str]) -> Vec<crate::Split<'a>> {
        let (inputs, indices) = self.logic.get_inputs_and_indices(texts);

        let slice_preds = self
            .backend
            .predict(inputs, self.logic.options().batch_size)
            .expect("model failure.");
        self.logic.split(texts, slice_preds, indices)
    }

    /// Gets the underlying NNSplitLogic.
    pub fn logic(&self) -> &NNSplitLogic {
        &self.logic
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Level;

    #[test]
    fn splitter_model_works() {
        let splitter = NNSplit::new(
            concat!(env!("CARGO_MANIFEST_DIR"), "/../models/de/model.onnx"),
            NNSplitOptions::default(),
        )
        .unwrap();
        let splits = &splitter.split(&["Das ist ein Test Das ist noch ein Test."])[0];

        assert_eq!(
            splits.flatten(0),
            vec!["Das ist ein Test ", "Das ist noch ein Test."]
        );
    }

    #[test]
    fn splitter_model_works_on_long_texts() {
        let splitter = NNSplit::new(
            concat!(env!("CARGO_MANIFEST_DIR"), "/../models/de/model.onnx"),
            NNSplitOptions::default(),
        )
        .unwrap();
        let text =
            "Eine Vernetzung von Neuronen im Nervensystem eines Lebewesens darstellen. ".repeat(20);
        let splits = &splitter.split(&[&text])[0];

        assert_eq!(
            splits.flatten(0),
            vec!["Eine Vernetzung von Neuronen im Nervensystem eines Lebewesens darstellen. "; 20]
        );
    }

    #[test]
    fn getting_levels_works() {
        let splitter = NNSplit::new(
            concat!(env!("CARGO_MANIFEST_DIR"), "/../models/de/model.onnx"),
            NNSplitOptions::default(),
        )
        .unwrap();

        assert_eq!(
            splitter.logic().split_sequence().get_levels(),
            vec![
                &Level("Sentence".into()),
                &Level("Token".into()),
                &Level("_Whitespace".into()),
                &Level("Compound constituent".into())
            ]
        );
    }
}

use crate::{NNSplitLogic, NNSplitOptions};
use ndarray::prelude::*;
use std::cmp;
use std::convert::TryInto;
use std::error::Error;

struct TchRsBackend {
    model: tch::CModule,
    device: tch::Device,
    n_outputs: usize,
}

impl TchRsBackend {
    fn new(model: tch::CModule, device: tch::Device) -> Self {
        let dummy_data = tch::Tensor::zeros(&[1, 1], (tch::Kind::Uint8, device));
        let n_outputs = model.forward_ts(&[dummy_data]).unwrap().size()[2] as usize;

        TchRsBackend {
            model,
            device,
            n_outputs,
        }
    }

    fn predict(&self, input: Array2<u8>, batch_size: usize) -> Result<Array3<f32>, Box<dyn Error>> {
        let input_shape = input.shape();

        let mut preds = Array3::<f32>::zeros((input_shape[0], input_shape[1], self.n_outputs));

        for i in (0..input_shape[0]).step_by(batch_size) {
            let start = i;
            let end = cmp::min(i + batch_size, input_shape[0]);

            let batch_inputs = input
                .slice(s![start..end, ..])
                .to_slice()
                .ok_or("converting ndarray to slice failed (likely not contiguous)")?;
            let batch_inputs = tch::Tensor::of_slice(batch_inputs)
                .view((-1, input_shape[1] as i64))
                .to_device(self.device);

            let batch_preds = self.model.forward_ts(&[batch_inputs])?.sigmoid();
            let batch_preds: ArrayD<f32> = (&batch_preds).try_into()?;

            preds.slice_mut(s![start..end, .., ..]).assign(&batch_preds);
        }

        Ok(preds)
    }
}

/// Complete splitter using tch-rs as backend.
pub struct NNSplit {
    backend: TchRsBackend,
    logic: NNSplitLogic,
}

impl NNSplit {
    /// Create a new splitter from the given model location.
    /// # Errors
    /// * If the file can not be loaded as TorchScript module.
    pub fn new<P: AsRef<std::path::Path>>(
        model_path: P,
        device: tch::Device,
        options: NNSplitOptions,
    ) -> Result<Self, Box<dyn Error>> {
        let model = tch::CModule::load(model_path)?;
        let backend = TchRsBackend::new(model, device);

        Ok(NNSplit {
            backend,
            logic: NNSplitLogic::new(options),
        })
    }

    /// Loads a built-in model. From the local cache or from the interent if it is not cached.
    #[cfg(feature = "model-loader")]
    pub fn load(
        model_name: &str,
        device: tch::Device,
        options: NNSplitOptions,
    ) -> Result<Self, Box<dyn Error>> {
        let filename = match device {
            tch::Device::Cpu => "torchscript_cpu_model.pt",
            tch::Device::Cuda(_) => "torchscript_cuda_model.pt",
        };
        let mut model_data = crate::model_loader::get_resource(model_name, filename)?.0;
        let model = tch::CModule::load_data(&mut model_data)?;
        let backend = TchRsBackend::new(model, device);

        Ok(NNSplit {
            backend,
            logic: NNSplitLogic::new(options),
        })
    }

    /// Split a list of texts into a list of `Split` objects.
    pub fn split<'a>(&self, texts: &[&'a str]) -> Vec<crate::Split<'a>> {
        let (inputs, indices) = self.logic.get_inputs_and_indices(texts);

        let slice_preds = self
            .backend
            .predict(inputs, self.logic.options.batch_size)
            .expect("model failure.");
        self.logic.split(texts, slice_preds, indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "model-loader")]
    #[test]
    fn splitter_model_works() {
        let splitter = NNSplit::load("de", tch::Device::Cpu, NNSplitOptions::default()).unwrap();
        let splits = &splitter.split(&["Das ist ein Test Das ist noch ein Test."])[0];

        assert_eq!(
            splits.flatten(0),
            vec!["Das ist ein Test ", "Das ist noch ein Test."]
        );
    }
}

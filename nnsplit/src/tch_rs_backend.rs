use crate::Backend;
use ndarray::prelude::*;
use std::cmp;
use std::convert::TryInto;

pub struct TchRsBackend {
    model: tch::CModule,
    device: tch::Device,
    n_outputs: usize,
}

impl TchRsBackend {
    pub fn new(model: tch::CModule, device: tch::Device) -> Self {
        let dummy_data = tch::Tensor::zeros(&[1, 1], (tch::Kind::Uint8, device));
        let n_outputs = model.forward_ts(&[dummy_data]).unwrap().size()[2] as usize;

        TchRsBackend {
            model,
            device,
            n_outputs,
        }
    }
}

impl Backend for TchRsBackend {
    fn predict(&self, input: Array2<u8>, batch_size: usize) -> Array3<f32> {
        let input_shape = input.shape();

        let mut preds = Array3::<f32>::zeros((input_shape[0], input_shape[1], self.n_outputs));

        for i in (0..input_shape[0]).step_by(batch_size) {
            let start = i;
            let end = cmp::min(i + batch_size, input_shape[0]);

            let batch_inputs = input.slice(s![start..end, ..]).to_slice().unwrap();
            let batch_inputs = tch::Tensor::of_slice(batch_inputs)
                .view((-1, input_shape[1] as i64))
                .to_device(self.device);

            let batch_preds = self.model.forward_ts(&[batch_inputs]).unwrap().sigmoid();
            let batch_preds: ArrayD<f32> = (&batch_preds).try_into().unwrap();

            preds.slice_mut(s![start..end, .., ..]).assign(&batch_preds);
        }

        return preds;
    }
}

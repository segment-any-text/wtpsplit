use ndarray::prelude::*;
use numpy::{PyArray3, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use std::cmp;

pub struct PytorchBackend {
    model: PyObject,
    device: PyObject,
    n_outputs: usize,
}

impl PytorchBackend {
    fn predict_batch(
        data: ArrayView2<u8>,
        model: &PyObject,
        device: &PyObject,
    ) -> PyResult<Array3<f32>> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let input = data.to_pyarray(py);

        let locals = [
            ("torch", &py.import("torch")?.to_object(py)),
            ("input", &input.to_object(py)),
            ("model", model),
            ("device", device),
        ]
        .into_py_dict(py);

        let prediction = py.eval(
            "model(torch.from_numpy(input).to(device)).detach().float().sigmoid().cpu().numpy()",
            None,
            Some(locals),
        )?;
        let prediction: &PyArray3<f32> = prediction.extract()?;

        Ok(prediction.to_owned_array())
    }

    pub fn new(model: PyObject, device: PyObject) -> PyResult<Self> {
        let dummy_data = Array2::<u8>::zeros((1, 1));
        let n_outputs = PytorchBackend::predict_batch((&dummy_data).into(), &model, &device)?.len();

        Ok(PytorchBackend {
            model,
            device,
            n_outputs,
        })
    }

    pub fn from_path(path: String, device: PyObject) -> PyResult<Self> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let model = py
            .import("torch.jit")?
            .call_method1("load", (path,))?
            .to_object(py);

        PytorchBackend::new(model, device)
    }

    pub fn predict(&self, input: Array2<u8>, batch_size: usize) -> PyResult<Array3<f32>> {
        let input_shape = input.shape();

        let mut preds = Array3::<f32>::zeros((input_shape[0], input_shape[1], self.n_outputs));

        for i in (0..input_shape[0]).step_by(batch_size) {
            let start = i;
            let end = cmp::min(i + batch_size, input_shape[0]);

            let batch_inputs = input.slice(s![start..end, ..]);
            let batch_preds =
                PytorchBackend::predict_batch(batch_inputs, &self.model, &self.device)?;

            preds.slice_mut(s![start..end, .., ..]).assign(&batch_preds);
        }

        Ok(preds)
    }
}

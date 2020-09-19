use lazy_static::lazy_static;
use ndarray::prelude::*;
use numpy::{PyArray3, ToPyArray};
use pyo3::prelude::*;
use std::cmp;

// only load the module once, otherwise onnxruntime gets newly imported for each splitter instance
lazy_static! {
    static ref MODULE: Py<PyModule> = {
        let guard = Python::acquire_gil();
        let py = guard.python();

        PyModule::from_code(py, include_str!("./backend.py"), "backend.py", "backend")
            .unwrap()
            .into()
    };
}

pub struct ONNXRuntimeBackend {
    session: PyObject,
    n_outputs: usize,
}

impl ONNXRuntimeBackend {
    fn predict_batch(
        py: Python,
        data: ArrayView2<u8>,
        session: &PyObject,
    ) -> PyResult<Array3<f32>> {
        let prediction: &PyArray3<f32> = MODULE
            .as_ref(py)
            .call1("predict_batch", (session, data.to_pyarray(py)))?
            .extract()?;

        let mut prediction = prediction.to_owned_array();

        // sigmoid
        prediction.mapv_inplace(|x| 1f32 / (1f32 + (-x).exp()));

        Ok(prediction)
    }

    pub fn new<P: AsRef<str>>(py: Python, model_path: P, use_cuda: Option<bool>) -> PyResult<Self> {
        let session = MODULE
            .as_ref(py)
            .call1("create_session", (model_path.as_ref(), use_cuda))?
            .into();

        let dummy_data = Array2::<u8>::zeros((1, 1));
        let n_outputs = ONNXRuntimeBackend::predict_batch(py, dummy_data.view(), &session)?.len();

        Ok(ONNXRuntimeBackend { session, n_outputs })
    }

    pub fn predict(
        &self,
        py: Python,
        input: Array2<u8>,
        batch_size: usize,
    ) -> PyResult<Array3<f32>> {
        let input_shape = input.shape();

        let mut preds = Array3::<f32>::zeros((input_shape[0], input_shape[1], self.n_outputs));

        for i in (0..input_shape[0]).step_by(batch_size) {
            let start = i;
            let end = cmp::min(i + batch_size, input_shape[0]);

            let batch_inputs = input.slice(s![start..end, ..]);
            let batch_preds = ONNXRuntimeBackend::predict_batch(py, batch_inputs, &self.session)?;

            preds.slice_mut(s![start..end, .., ..]).assign(&batch_preds);
        }

        Ok(preds)
    }
}

use lazy_static::lazy_static;
use ndarray::prelude::*;
use numpy::{PyArray3, ToPyArray};
use pyo3::prelude::*;
use std::cmp;
use std::collections::HashMap;

// only load the module once, otherwise onnxruntime gets newly imported for each splitter instance
lazy_static! {
    static ref MODULE: Py<PyModule> = {
        let guard = Python::acquire_gil();
        let py = guard.python();

        PyModule::from_code(py, include_str!("./backend.py"), "backend.py", "backend")
            .map_err(|x| x.print(py))
            .expect("error loading backend.py")
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

        let dummy_data = Array2::<u8>::zeros((1, 12));
        let dummy_out = ONNXRuntimeBackend::predict_batch(py, dummy_data.view(), &session)?;
        let shape = dummy_out.shape();
        let n_outputs = shape[shape.len() - 1];

        Ok(ONNXRuntimeBackend { session, n_outputs })
    }

    pub fn predict(
        &self,
        py: Python,
        input: Array2<u8>,
        batch_size: usize,
        verbose: bool,
    ) -> PyResult<Array3<f32>> {
        let input_shape = input.shape();

        let mut preds = Array3::<f32>::zeros((input_shape[0], input_shape[1], self.n_outputs));

        let bar = if verbose {
            Some(
                MODULE
                    .as_ref(py)
                    .call1("get_progress_bar", (input_shape[0],))?,
            )
        } else {
            None
        };

        for i in (0..input_shape[0]).step_by(batch_size) {
            let start = i;
            let end = cmp::min(i + batch_size, input_shape[0]);

            let batch_inputs = input.slice(s![start..end, ..]);
            let batch_preds = ONNXRuntimeBackend::predict_batch(py, batch_inputs, &self.session)?;

            preds.slice_mut(s![start..end, .., ..]).assign(&batch_preds);

            if let Some(bar) = bar {
                bar.call_method1("update", (end - start,))?;
            }
        }

        if let Some(bar) = bar {
            bar.call_method0("close")?;
        }

        Ok(preds)
    }

    pub fn get_metadata(&self, py: Python) -> PyResult<HashMap<String, String>> {
        MODULE
            .as_ref(py)
            .call1("get_metadata", (&self.session,))?
            .extract()
    }
}

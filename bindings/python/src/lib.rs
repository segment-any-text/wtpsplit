mod onnxruntime_backend;

use onnxruntime_backend::ONNXRuntimeBackend;
use pyo3::class::sequence::PySequenceProtocol;
use pyo3::conversion::IntoPy;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::{class::basic::PyObjectProtocol, exceptions::PyIndexError};

use nnsplit as core;

create_exception!(nnsplit, ResourceError, PyException);

/// Represents a splitted text. Can be iterated over to yield either:
///     * `Split` objects representing smaller parts of this split.
///     * `str` objects if at the lowest split level.
///
/// Can also be stringifed with `str(...)` to get the full text this split contains.
#[pyclass]
pub struct Split {
    parts: Vec<PyObject>,
}

fn join_method_output(items: &[PyObject], method: &str, joiner: &str) -> PyResult<String> {
    let gil = Python::acquire_gil();
    let py = gil.python();

    let string_parts = items.iter().map(|x| {
        x.call_method0(py, method)
            .and_then(|x| x.extract::<String>(py))
    });
    let joined = string_parts.collect::<PyResult<Vec<_>>>()?[..].join(joiner);
    Ok(joined)
}

fn to_options(maybe_py_dict: Option<&PyDict>) -> PyResult<core::NNSplitOptions> {
    let mut options = core::NNSplitOptions::default();

    if let Some(py_dict) = maybe_py_dict {
        if let Some(obj) = py_dict.get_item("threshold") {
            options.threshold = obj.extract()?;
        }

        if let Some(obj) = py_dict.get_item("stride") {
            options.stride = obj.extract()?;
        }

        if let Some(obj) = py_dict.get_item("max_length") {
            options.max_length = obj.extract()?;
        }

        if let Some(obj) = py_dict.get_item("padding") {
            options.padding = obj.extract()?;
        }

        if let Some(obj) = py_dict.get_item("batch_size") {
            options.batch_size = obj.extract()?;
        }
    }

    Ok(options)
}

#[pyproto]
impl PyObjectProtocol for Split {
    fn __str__(&self) -> PyResult<String> {
        join_method_output(&self.parts, "__str__", "")
    }

    fn __repr__(&self) -> PyResult<String> {
        let joined = join_method_output(&self.parts, "__repr__", ", ")?;
        Ok(format!("Split({})", joined))
    }
}

#[pyproto]
impl PySequenceProtocol for Split {
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.parts.len())
    }

    fn __getitem__(&self, idx: isize) -> PyResult<PyObject> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        if idx >= 0 && (idx as usize) < self.parts.len() {
            Ok(self.parts[idx as usize].clone_ref(py))
        } else {
            Err(PyIndexError::new_err("list index out of range"))
        }
    }
}

impl<'a> IntoPy<Split> for core::Split<'a> {
    fn into_py(self, py: Python) -> Split {
        match self {
            core::Split::Text(_) => panic!("text can not be converted to a Split"),
            core::Split::Split((_, split_parts)) => {
                let parts = split_parts
                    .into_iter()
                    .map(|x| match &x {
                        core::Split::Split(_) => {
                            let split: Split = x.into_py(py);
                            PyCell::new(py, split).unwrap().to_object(py)
                        }
                        core::Split::Text(text) => text.to_object(py),
                    })
                    .collect();

                Split { parts }
            }
        }
    }
}

/// Complete Splitter using ONNXRuntime as backend.
///
/// Args:
///     model_path (str or pathlib.Path): Path to a .onnx model to use for prediction.
///     use_cuda (bool): Whether to use CUDA to run the model on GPU. If None, will use CUDA if available, otherwise CPU.
///     **kwargs: Additional options. Can be:
///         * threshold (float): Threshold from 0 to 1 above which predictions will be considered positive.
///         * stride (int): How much to move the window after each prediction (comparable to stride of 1d convolution).
///         * max_length (int): The maximum length of each cut (comparable to kernel size of 1d convolution).
///         * padding (int): How much to zero pad the text on both sides.
///         * batch_size (int): Batch size to use.
///         * length_divisor (int): Total length will be padded until it is divisible by this number. Allows some additional optimizations.
#[pyclass]
#[pyo3(text_signature = "(model_path, use_cuda=None, **kwargs)")]
pub struct NNSplit {
    backend: ONNXRuntimeBackend,
    logic: core::NNSplitLogic,
}

impl NNSplit {
    fn from_backend_and_kwargs(
        py: Python,
        backend: ONNXRuntimeBackend,
        kwargs: Option<&PyDict>,
    ) -> PyResult<Self> {
        let options = to_options(kwargs)?;
        let metadata = backend.get_metadata(py)?;

        Ok(NNSplit {
            backend,
            logic: core::NNSplitLogic::new(
                options,
                serde_json::from_str(metadata.get("split_sequence").ok_or_else(|| {
                    PyException::new_err("Model must contain `split_sequence` metadata key")
                })?)
                .map_err(|_| PyException::new_err("split_sequence must be valid JSON."))?,
            ),
        })
    }
}

#[pymethods]
impl NNSplit {
    #[new]
    #[args(kwargs = "**")]
    pub fn new(
        py: Python,
        model_path: PyObject,
        use_cuda: Option<bool>,
        kwargs: Option<&PyDict>,
    ) -> PyResult<Self> {
        // explicitly call str(..) to handle pathlib.Path etc. correctly
        let path = model_path.as_ref(py).str()?.to_string();

        let backend = ONNXRuntimeBackend::new(py, path, use_cuda)?;
        NNSplit::from_backend_and_kwargs(py, backend, kwargs)
    }

    /// Loads a built-in model. From the local cache or from the interent if it is not cached.
    ///
    /// Args:
    ///     model_name (str): Name of the model.
    ///     use_cuda (bool): Whether to use CUDA to run the model on GPU. If None, will use CUDA if available, otherwise CPU.
    ///     **kwargs: Additional options. Can be:
    ///         * threshold (float): Threshold from 0 to 1 above which predictions will be considered positive.
    ///         * stride (int): How much to move the window after each prediction (comparable to stride of 1d convolution).
    ///         * max_length (int): The maximum length of each cut (comparable to kernel size of 1d convolution).
    ///         * padding (int): How much to zero pad the text on both sides.
    ///         * batch_size (int): Batch size to use.
    ///         * length_divisor (int): Total length will be padded until it is divisible by this number. Allows some additional optimizations.
    #[pyo3(text_signature = "(model_name, use_cuda=None, **kwargs)")]
    #[args(kwargs = "**")]
    #[staticmethod]
    pub fn load(
        py: Python,
        model_name: &str,
        use_cuda: Option<bool>,
        kwargs: Option<&PyDict>,
    ) -> PyResult<Self> {
        let (_, resource_path) = core::model_loader::get_resource(&model_name, "model.onnx")
            .map_err(|error| ResourceError::new_err(error.to_string()))?;

        let backend = ONNXRuntimeBackend::new(
            py,
            resource_path
                .expect("could not cache model.")
                .into_os_string()
                .into_string()
                .unwrap(),
            use_cuda,
        )?;

        NNSplit::from_backend_and_kwargs(py, backend, kwargs)
    }

    /// Splits text into `Split` objects.
    ///
    /// Args:
    ///     texts (List[str]): List of texts to split.
    ///     verbose (bool): Whether to display a progress bar.
    /// Returns:
    ///     splits (List[Split]): A list of `Split` objects with the same length as the input text list.
    #[pyo3(text_signature = "(texts, verbose=False)")]
    pub fn split(
        &self,
        py: Python,
        texts: Vec<&str>,
        verbose: Option<bool>,
    ) -> PyResult<Vec<Split>> {
        let (inputs, indices) = self.logic.get_inputs_and_indices(&texts);

        let slice_preds = self.backend.predict(
            py,
            inputs,
            self.logic.options().batch_size,
            verbose.unwrap_or(false),
        )?;

        let splits = self.logic.split(&texts, slice_preds, indices);
        Ok(splits.into_iter().map(|x| x.into_py(py)).collect())
    }

    /// Gets names of the levels of this splitter.
    ///
    /// Returns:
    ///     levels (List[str]): A list of strings describing the split levels, from top (largest split) to bottom (smallest split).
    #[pyo3(text_signature = "()")]
    pub fn get_levels(&self) -> PyResult<Vec<String>> {
        Ok(self
            .logic
            .split_sequence()
            .get_levels()
            .iter()
            .map(|x| x.0.clone())
            .collect())
    }
}

#[pymodule]
fn nnsplit(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<NNSplit>()?;
    m.add_class::<Split>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}

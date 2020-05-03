use pyo3::class::basic::PyObjectProtocol;
use pyo3::class::gc::{PyGCProtocol, PyTraverseError, PyVisit};
use pyo3::class::sequence::PySequenceProtocol;
use pyo3::conversion::FromPy;
use pyo3::prelude::*;

mod pytorch_backend;
use pytorch_backend::PytorchBackend;

#[pyclass(gc)]
pub struct Split {
    parts: Vec<PyObject>,
}

fn join_method_result(items: &Vec<PyObject>, method: &str, joiner: &str) -> PyResult<String> {
    let gil = Python::acquire_gil();
    let py = gil.python();

    let string_parts = items.iter().map(|x| {
        x.call_method0(py, method)
            .and_then(|x| x.extract::<String>(py))
    });
    let joined = string_parts.collect::<PyResult<Vec<_>>>()?[..].join(joiner);
    Ok(joined)
}

#[pyproto]
impl PyObjectProtocol for Split {
    fn __str__(&self) -> PyResult<String> {
        join_method_result(&self.parts, "__str__", "")
    }

    fn __repr__(&self) -> PyResult<String> {
        let joined = join_method_result(&self.parts, "__repr__", ", ")?;
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
            Err(PyErr::new::<pyo3::exceptions::IndexError, _>(
                "list index out of range",
            ))
        }
    }
}

#[pyproto]
impl PyGCProtocol for Split {
    fn __traverse__(&'p self, visit: PyVisit) -> Result<(), PyTraverseError> {
        let gil = Python::acquire_gil();
        let python = gil.python();

        self.parts
            .iter()
            .map(|x| {
                // TODO: clarify if this is required
                if let Ok(split) = x.extract::<&PyCell<Split>>(python) {
                    for part in &split.borrow().parts {
                        visit.call(part)?;
                    }
                }

                visit.call(x)
            })
            .collect()
    }

    fn __clear__(&'p mut self) {
        let gil = Python::acquire_gil();
        let python = gil.python();

        for part in self.parts.drain(..) {
            // TODO: clarify if this is required
            if let Ok(split) = part.extract::<&PyCell<Split>>(python) {
                for subpart in &split.borrow().parts {
                    python.release(subpart);
                }
            }

            python.release(part);
        }
    }
}

impl<'a> FromPy<nnsplit::Split<'a>> for Split {
    fn from_py(split: nnsplit::Split, py: Python) -> Self {
        match split {
            nnsplit::Split::Text(_) => unreachable!(),
            nnsplit::Split::Split((_, split_parts)) => {
                let parts = split_parts
                    .into_iter()
                    .map(|x| match &x {
                        nnsplit::Split::Split(_) => {
                            let split: Split = x.into_py(py);
                            PyCell::new(py, split).unwrap().to_object(py)
                        }
                        nnsplit::Split::Text(text) => text.to_object(py),
                    })
                    .collect();

                Split { parts }
            }
        }
    }
}

#[pyclass]
pub struct NNSplit {
    inner: nnsplit::NNSplit,
}

#[pymethods]
impl NNSplit {
    #[staticmethod]
    pub fn new(model: PyObject, device: PyObject, batch_size: usize) -> Self {
        let backend = PytorchBackend::new(model, device, batch_size).unwrap();

        NNSplit {
            inner: nnsplit::NNSplit::new(
                Box::new(backend) as Box<dyn nnsplit::Backend>,
                nnsplit::NNSplitOptions::default(),
            )
            .unwrap(),
        }
    }

    pub fn split<'a>(&self, py: Python, texts: Vec<&'a str>) -> Vec<Split> {
        let splits = self.inner.split(texts);

        splits.into_iter().map(|x| x.into_py(py)).collect()
    }
}

#[pymodule]
fn nnsplit(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<NNSplit>()?;

    Ok(())
}

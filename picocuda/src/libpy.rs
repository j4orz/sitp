use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyList};

use crate::{nn, Device, Dtype, DtypeVal, Layout, Tensor};

#[pymethods]
impl Tensor {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self))
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.ndim
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    #[getter]
    fn stride(&self) -> Vec<usize> {
        self.stride.clone()
    }

    #[getter]
    fn device(&self) -> Device {
        self.device.clone()
    }

    #[getter]
    fn layout(&self) -> Layout {
        self.layout.clone()
    }

    #[getter]
    fn dtype(&self) -> Dtype {
        self.dtype.clone()
    }
}

#[pyfunction]
fn tensor<'py>(data: Bound<'py, PyList>) -> PyResult<Tensor> {
    let data = data.extract::<Vec<DtypeVal>>()?;
    Ok(crate::new(data))
}

#[pyfunction]
fn tanh(x: Tensor) -> PyResult<Tensor> {
    x.tanh().map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// A Python module implemented in Rust.
#[pymodule]
fn picograd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // constructors
    m.add_function(wrap_pyfunction!(tensor, m)?)?;
    m.add_function(wrap_pyfunction!(crate::zeros, m)?)?;
    m.add_function(wrap_pyfunction!(crate::ones, m)?)?;
    m.add_function(wrap_pyfunction!(crate::randn, m)?)?;
    m.add_function(wrap_pyfunction!(crate::arange, m)?)?;

    // ops
    m.add_function(wrap_pyfunction!(tanh, m)?)?;
    nn_module(m)?;

    // F.cross_entropy()
    // F.softmax()

    // inference (rng)
    // - picograd.randint()
    // - picograd.multinomial()
    // - picograd.Generator().manual_seed()
    // - https://pytorch.org/docs/stable/notes/randomness.html
    Ok(())
}

fn nn_module(pg_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let nn_module = PyModule::new(pg_module.py(), "nn")?;
    functional_module(&nn_module)?;
    pg_module.add_submodule(&nn_module)
}

fn functional_module(nn_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let functional_module = PyModule::new(nn_module.py(), "functional")?;
    functional_module.add_function(wrap_pyfunction!(nn::cross_entropy, &functional_module)?)?;
    functional_module.add_function(wrap_pyfunction!(nn::softmax, &functional_module)?)?;
    nn_module.add_submodule(&functional_module)
}

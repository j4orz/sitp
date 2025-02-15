use pyo3::prelude::*;

use crate::{Device, Dtype, Layout, Tensor};

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

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn picograd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(crate::zeros, m)?)?;
    m.add_function(wrap_pyfunction!(crate::ones, m)?)?;
    m.add_function(wrap_pyfunction!(crate::randn, m)?)?;
    m.add_function(wrap_pyfunction!(crate::arange, m)?)?;
    // tanh
    Ok(())
}

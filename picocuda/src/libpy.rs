use pyo3::{prelude::*, types::PyList};

use crate::{Device, Dtype, DtypeVal, Layout, Tensor};

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

/// A Python module implemented in Rust.
#[pymodule]
fn picograd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // constructors
    m.add_function(wrap_pyfunction!(tensor, m)?)?;
    m.add_function(wrap_pyfunction!(crate::zeros, m)?)?;
    m.add_function(wrap_pyfunction!(crate::ones, m)?)?;
    m.add_function(wrap_pyfunction!(crate::randn, m)?)?;
    m.add_function(wrap_pyfunction!(crate::arange, m)?)?;

    // tanh
    // F.cross_entropy()
    // F.softmax()

    // inference (rng)
    // - picograd.randint()
    // - picograd.multinomial()
    // - picograd.Generator().manual_seed()
    // - https://pytorch.org/docs/stable/notes/randomness.html
    Ok(())
}

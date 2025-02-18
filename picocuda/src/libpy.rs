use crate::{nn, Device, Dtype, DtypeVal, Layout, Tensor};
use numpy::{PyArrayMethods, PyUntypedArrayMethods};
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyList};
use std::ops::{Add, Div, Mul, Sub};

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

    fn __add__(&self, other: &Tensor) -> PyResult<Tensor> {
        self.add(other)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn __sub__(&self, other: &Tensor) -> PyResult<Tensor> {
        self.sub(other)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn __mul__(&self, other: &Tensor) -> PyResult<Tensor> {
        self.mul(other)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn __div__(&self, other: &Tensor) -> PyResult<Tensor> {
        self.div(other)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn __matmul__(&self, other: &Tensor) -> PyResult<Tensor> {
        self.matmul(other)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

#[pyfunction]
fn tensor<'py>(data: Bound<'py, PyAny>) -> PyResult<Tensor> {
    if let Ok(np) = data.downcast::<numpy::PyArrayDyn<f32>>() {
        // cast once to PyArrayDyn<f32>
        let np = np.try_readonly()?;
        let (data, shape) = (
            np.as_slice()?
                .to_vec()
                .into_iter()
                .map(DtypeVal::Float32) // cast twice to picograd::DtypeVal
                .collect(),
            np.shape().to_vec(),
        );
        Ok(crate::alloc(&shape, data))
    } else if let Ok(pylist) = data.downcast::<PyList>() {
        let data_vec = pylist.extract::<Vec<DtypeVal>>()?;
        Ok(crate::new(data_vec))
    } else {
        Err(PyRuntimeError::new_err(
            "Unsupported type: Expected NumPy ndarray or list",
        ))
    }
}

#[pyfunction]
fn tanh(x: Tensor) -> PyResult<Tensor> {
    x.tanh().map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// A Python module implemented in Rust.
#[pymodule]
fn picograd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // m.add_class::<Tensor>()?;

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

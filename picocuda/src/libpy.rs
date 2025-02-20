use crate::{nn, Device, Dtype, DtypeVal, Layout, Tensor};
use numpy::{PyArrayMethods, PyUntypedArrayMethods};
use pyo3::{
    exceptions::PyRuntimeError,
    prelude::*,
    types::{PyList, PyTuple},
};
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
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.shape)
    }

    #[getter]
    fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    #[setter]
    fn set_requires_grad(&mut self, value: bool) {
        self.requires_grad = value;
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

    fn __add__(&self, other: Bound<'_, PyAny>) -> PyResult<Tensor> {
        if let Ok(val) = other.extract::<f32>() {
            self.add(val)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        } else if let Ok(t2) = other.extract::<Tensor>() {
            self.add(&t2)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        } else {
            Err(PyRuntimeError::new_err(
                "Expected a Tensor or a float32 scalar.",
            ))
        }
    }

    fn __sub__(&self, other: Bound<'_, PyAny>) -> PyResult<Tensor> {
        if let Ok(val) = other.extract::<f32>() {
            self.sub(val)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        } else if let Ok(t2) = other.extract::<Tensor>() {
            self.sub(&t2)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        } else {
            Err(PyRuntimeError::new_err(
                "Expected a Tensor or a float32 scalar.",
            ))
        }
    }
    fn __mul__(&self, other: Bound<'_, PyAny>) -> PyResult<Tensor> {
        if let Ok(val) = other.extract::<f32>() {
            self.mul(val)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        } else if let Ok(t2) = other.extract::<Tensor>() {
            self.mul(&t2)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        } else {
            Err(PyRuntimeError::new_err(
                "Expected a Tensor or a float32 scalar.",
            ))
        }
    }

    fn __truediv__(&self, other: Bound<'_, PyAny>) -> PyResult<Tensor> {
        if let Ok(val) = other.extract::<f32>() {
            self.div(val)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        } else if let Ok(t2) = other.extract::<Tensor>() {
            self.div(&t2)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        } else {
            Err(PyRuntimeError::new_err(
                "Expected a Tensor or a float32 scalar.",
            ))
        }
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

#[pyfunction]
fn exp(x: Tensor) -> PyResult<Tensor> {
    x.exp().map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn log(x: Tensor) -> PyResult<Tensor> {
    x.log().map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn sum(x: Tensor, dim: usize, keepdim: bool) -> PyResult<Tensor> {
    x.sum(dim, keepdim)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
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
    m.add_function(wrap_pyfunction!(exp, m)?)?;
    m.add_function(wrap_pyfunction!(log, m)?)?;
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    nn_module(m)?;

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
    // functional_module.add_function(wrap_pyfunction!(nn::softmax, &functional_module)?)?;
    nn_module.add_submodule(&functional_module)
}

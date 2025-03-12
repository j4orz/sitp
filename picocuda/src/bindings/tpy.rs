use crate::ops::cpu_ops::OpForwardError;
use crate::trs::Tensor;
use crate::{Device, Dtype, Layout};
use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyTuple};
use std::ops::{Add, Div, Mul, Sub};

// forward---
// 2. indexing on trs (TEST)
// 3. view on trs (TEST)
// 4. softmax (TEST)
// 5. multinomial <--- home free (forward pass)

// backward---

impl From<OpForwardError> for PyErr {
    fn from(e: OpForwardError) -> Self {
        PyRuntimeError::new_err(e.to_string())
    }
}

#[pymethods]
impl Tensor {
    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
        // todo: why does pytorch need to detatch first?
        let data = self
            .storage
            .borrow()
            .data
            .iter()
            .map(|&x| x.into())
            .collect::<Vec<f32>>();

        let np_flat = data.into_pyarray(py).to_dyn().to_owned();
        let np_shaped = np_flat.reshape(self.shape.clone())?;
        Ok(np_shaped)
    }

    pub fn detach(&self) -> Self {
        Self {
            ndim: self.ndim,
            shape: self.shape.clone(),
            stride: self.stride.clone(),
            input_op: None, // detach
            requires_grad: self.requires_grad,
            storage: self.storage.clone(),
            device: self.device.clone(),
            layout: self.layout.clone(),
            dtype: self.dtype.clone(),
        }
    }

    // fn view(&self, shape: Bound<'_, PyList>) -> PyResult<Tensor> {
    //     Ok(self.no_alloc(shape))
    // }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self))
    }

    fn __getitem__(&self, I: Tensor) -> PyResult<Self> {
        todo!()
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

    // VIEW OPS
    fn reshape(&self, shape: Bound<'_, PyTuple>) -> PyResult<Tensor> {
        let shape = shape.extract::<Vec<usize>>()?;
        Ok(self._reshape(&shape)?)
    }

    // POINTWISE OPS
    fn __add__(&self, other: Bound<'_, PyAny>) -> PyResult<Tensor> {
        if let Ok(val) = other.extract::<f32>() {
            Ok(self.add(val)?)
        } else if let Ok(t2) = other.extract::<Tensor>() {
            Ok(self.add(&t2)?)
        } else {
            Err(PyRuntimeError::new_err("expected a tensor or scalar"))
        }
    }

    fn __sub__(&self, other: Bound<'_, PyAny>) -> PyResult<Tensor> {
        if let Ok(val) = other.extract::<f32>() {
            Ok(self.sub(val)?)
        } else if let Ok(t2) = other.extract::<Tensor>() {
            Ok(self.sub(&t2)?)
        } else {
            Err(PyRuntimeError::new_err("expected a tensor or scalar"))
        }
    }

    fn __mul__(&self, other: Bound<'_, PyAny>) -> PyResult<Tensor> {
        if let Ok(val) = other.extract::<f32>() {
            Ok(self.mul(val)?)
        } else if let Ok(t2) = other.extract::<Tensor>() {
            Ok(self.mul(&t2)?)
        } else {
            Err(PyRuntimeError::new_err("expected a tensor or scalar"))
        }
    }

    fn __truediv__(&self, other: Bound<'_, PyAny>) -> PyResult<Tensor> {
        if let Ok(val) = other.extract::<f32>() {
            Ok(self.div(val)?)
        } else if let Ok(t2) = other.extract::<Tensor>() {
            Ok(self.div(&t2)?)
        } else {
            Err(PyRuntimeError::new_err("expected a tensor or scalar"))
        }
    }

    // PROCESSING OPS
    fn __matmul__(&self, other: &Tensor) -> PyResult<Tensor> {
        Ok(self.matmul(other)?)
    }
}

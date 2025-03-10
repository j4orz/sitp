use crate::ops::cpu_ops::OpForwardError;
use crate::trs::Tensor;
use crate::{Device, Dtype, Layout};
use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyTuple};
use std::ops::{Add, Div, Mul, Sub};

// forward---
// 1. error mapping
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

    // TODO: partial indexing return Tensor with one less dim
    // fn __getitem__(&self, I: Tensor) -> PyResult<Self> {
    //     let output_shape = I
    //         .shape
    //         .iter()
    //         .chain(self.shape.iter().skip(1)) // collapse the first dim of self via indexing
    //         .copied()
    //         .collect::<Vec<_>>();
    //     let output = trs::zeros(output_shape, Dtype::Float32);

    //     {
    //         let I_storage = I.storage.borrow();
    //         let input_storage = self.storage.borrow();
    //         let mut output_storage = output.storage.borrow_mut();

    //         for phy_I in 0..I_storage.data.len() {
    //             let i = usize::from(I_storage.data[phy_I]);
    //             let (l, r) = (self.stride[0] * i, (self.stride[0] * i) + self.stride[0]);
    //             let plucked_tensor = &input_storage.data[l..r];
    //             // place plucked_tensor (nested ndarray) in output_storage

    //             let log_I = Self::encode(phy_I, &I.shape); // where we slot the plucked input in the output tensor
    //             let log_output = log_I
    //                 .iter()
    //                 .chain(iter::repeat(&0).take(self.shape.len() - 1)) // input.shape.len()
    //                 .copied()
    //                 .collect::<Vec<_>>();

    //             let phys_output = Self::decode(&log_output, &output.shape);
    //             output_storage.data[phys_output..phys_output + plucked_tensor.len()]
    //                 .copy_from_slice(plucked_tensor);
    //         }
    //     }
    //     Ok(output)
    // }

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

    fn __matmul__(&self, other: &Tensor) -> PyResult<Tensor> {
        Ok(self.matmul(other)?)
    }
}

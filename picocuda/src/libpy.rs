use crate::{
    nn,
    tensor::{self, Device, Dtype, DtypeVal, Layout, Tensor},
};
use numpy::{PyArrayMethods, PyUntypedArrayMethods};
use pyo3::{
    exceptions::PyRuntimeError,
    prelude::*,
    types::{PyList, PyTuple},
};
use std::{
    iter,
    ops::{Add, Div, Mul, Sub},
};

#[pymethods]
impl Tensor {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self))
    }

    // TODO: partial indexing return Tensor with one less dim
    fn __getitem__(&self, I: Tensor) -> PyResult<Self> {
        let output_shape = I
            .shape
            .iter()
            .chain(self.shape.iter().skip(1)) // collapse the first dim of self via indexing
            .copied()
            .collect::<Vec<_>>();
        let output = tensor::zeros(output_shape, Dtype::Float32);

        {
            let I_storage = I.storage.borrow();
            let input_storage = self.storage.borrow();
            let mut output_storage = output.storage.borrow_mut();

            for phy_I in 0..I_storage.data.len() {
                let i = usize::from(I_storage.data[phy_I]);
                let (l, r) = (self.stride[0] * i, (self.stride[0] * i) + self.stride[0]);
                let plucked_tensor = &input_storage.data[l..r];
                // place plucked_tensor (nested ndarray) in output_storage

                let log_I = Self::encode(phy_I, &I.shape); // where we slot the plucked input in the output tensor
                let log_output = log_I
                    .iter()
                    .chain(iter::repeat(&0).take(self.shape.len() - 1)) // input.shape.len()
                    .copied()
                    .collect::<Vec<_>>();

                let phys_output = Self::decode(&log_output, &output.shape);
                output_storage.data[phys_output..phys_output + plucked_tensor.len()]
                    .copy_from_slice(plucked_tensor);
            }
        }
        Ok(output)
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

// #[pyfunction]
// fn tensor<'py>(data: Bound<'py, PyAny>) -> PyResult<Tensor> {
//     fn alloc_from_np<T: numpy::Element, 'py>(
//         np: &Bound<'py, numpy::PyArrayDyn<T>>,
//     ) -> PyResult<Tensor>
//     where
//         T: Into<DtypeVal> + Copy,
//     {
//         let np = np.try_readonly()?;
//         let (data, shape) = (
//             np.as_slice()?
//                 .iter()
//                 .map(|x| (*x).into())
//                 .collect::<Vec<DtypeVal>>(),
//             np.shape().to_vec(),
//         );
//         Ok(tensor::alloc(&shape, data))
//     }

//     if let Ok(np) = data.downcast::<numpy::PyArrayDyn<i32>>() {
//         return alloc_from_np(np);
//     } else if let Ok(np) = data.downcast::<numpy::PyArrayDyn<i64>>() {
//         return alloc_from_np(np);
//     } else if let Ok(np) = data.downcast::<numpy::PyArrayDyn<f32>>() {
//         return alloc_from_np(np);
//     } else if let Ok(pylist) = data.downcast::<PyList>() {
//         let (mut data, mut shape) = (Vec::new(), Vec::new());
//         flatten_pylist(pylist.as_any(), &mut data, &mut shape, 1)?;
//         // println!("moose {:?}", data);
//         // println!("moose {:?}", shape);
//         Ok(tensor::alloc(&shape, data))
//     } else {
//         Err(PyRuntimeError::new_err(
//             "Unsupported type: Expected NumPy ndarray or list",
//         ))
//     }
// }

fn flatten_pylist<'py>(
    pyobj: &Bound<'py, PyAny>,
    data: &mut Vec<DtypeVal>,
    shape: &mut Vec<usize>,
    i: usize, // 1-based
) -> PyResult<()> {
    match pyobj.downcast::<PyList>() {
        Ok(pylist) => {
            if shape.len() < i {
                shape.push(pylist.len());
            }
            for item in pylist.iter() {
                flatten_pylist(&item, data, shape, i + 1)?;
            }
        }
        Err(_) => data.push(pyobj.extract::<DtypeVal>()?),
    };

    Ok(())
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
fn picograd(py: Python, pg_m: &Bound<'_, PyModule>) -> PyResult<()> {
    // m.add_class::<Tensor>()?;

    // constructors
    // pg_m.add_function(wrap_pyfunction!(tensor::tensor, pg_m)?)?;
    pg_m.add_function(wrap_pyfunction!(tensor::zeros, pg_m)?)?;
    pg_m.add_function(wrap_pyfunction!(tensor::ones, pg_m)?)?;
    pg_m.add_function(wrap_pyfunction!(tensor::randn, pg_m)?)?;
    pg_m.add_function(wrap_pyfunction!(tensor::arange, pg_m)?)?;

    // ops
    pg_m.add_function(wrap_pyfunction!(tanh, pg_m)?)?;
    pg_m.add_function(wrap_pyfunction!(exp, pg_m)?)?;
    pg_m.add_function(wrap_pyfunction!(log, pg_m)?)?;
    pg_m.add_function(wrap_pyfunction!(sum, pg_m)?)?;

    // nn.functional
    let ff_m = PyModule::new(py, "functional")?;
    ff_m.add_function(wrap_pyfunction!(nn::cross_entropy, &ff_m)?)?;

    // nn
    let nn_m = PyModule::new(py, "nn")?;
    nn_m.add_submodule(&ff_m)?;
    pg_m.add_submodule(&nn_m)?;

    // BUG: see pyo3/issues/759: https://github.com/PyO3/pyo3/issues/759#issuecomment-977835119
    py.import("sys")?
        .getattr("modules")?
        .set_item("picograd.nn", nn_m)?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("picograd.nn.functional", ff_m)?;

    Ok(())
}

// fn register_nn_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
//     let nn_m = PyModule::new(parent_module.py(), "nn")?;
//     register_nn_functional_module(&nn_m)?;
//     parent_module.add_submodule(&nn_m)
// }

// fn register_nn_functional_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
//     let functional_m = PyModule::new(parent_module.py(), "functional")?;
//     functional_m.add_function(wrap_pyfunction!(nn::cross_entropy, &functional_m)?)?;
//     parent_module.add_submodule(&functional_m)
// }

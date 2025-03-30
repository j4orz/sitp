use std::ops::{Div, Sub};

use pyo3::{PyResult, exceptions::PyRuntimeError, pyfunction};

use crate::trs::Tensor;

#[pyfunction]
pub fn cross_entropy(p: Tensor, q: Tensor) -> PyResult<Tensor> {
    todo!()
}

#[pyfunction]
pub fn softmax(x: Tensor, dim: usize) -> PyResult<Tensor> {
    let shifted = x.sub(&x.max(dim, true)?)?;
    let exp = shifted.exp()?;
    let expsum = exp.sum(dim, true)?;
    let softmax = exp.div(&expsum)?;
    Ok(softmax)
}

use pyo3::{PyResult, exceptions::PyRuntimeError, pyfunction};

use crate::trs::Tensor;

#[pyfunction]
pub fn cross_entropy(p: Tensor, q: Tensor) -> PyResult<Tensor> {
    // H(p,q) := -Σ(pᵢ log qᵢ)
    // let (p, q) = (softmax(p, -1), softmax(q, -1));
    // (&p * &q.log()).sum() * -1 // 1. broadcast
    let logq = q
        .log()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let plogq = (&p * &logq).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    // sum across classes:
    let sum_class = plogq
        .sum(1, false)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    // sum across batch (or mean, etc.) if you want a scalar:
    let sum_all = sum_class
        .sum(0, false)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let negsum = (-&sum_all).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(negsum)
}

#[pyfunction]
pub fn softmax(x: Tensor, dim: usize) -> PyResult<Tensor> {
    // softmax(x) := exp(xᵢ) / Σⱼ exp(xⱼ)
    let exp = x
        .exp()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let expsum = exp
        .sum(dim, true)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let softmax = (&exp / &expsum).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(softmax)
}

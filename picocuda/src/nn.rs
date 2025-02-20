use pyo3::{exceptions::PyRuntimeError, pyfunction, PyResult};

use crate::Tensor;

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

// #[pyfunction]
// pub fn softmax(x: Tensor, dim: usize) -> Tensor {
//     // softmax(x)_i := exp(xᵢ) / Σⱼ exp(xⱼ)
//     let exp = x.exp();
//     let expsum = exp.sum(dim, true); // keepdim=true
//     &exp / &expsum
// }

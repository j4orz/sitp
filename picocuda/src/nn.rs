use pyo3::pyfunction;

use crate::Tensor;

#[pyfunction]
pub fn cross_entropy(p: Tensor, q: Tensor) -> Tensor {
    // H(p,q) := -Σ(pᵢ log qᵢ)
    let (p, q) = (softmax(p, -1), softmax(q, -1));
    -&(&p * &q.log()).sum(1, true) // 1. unary
                                   // (&p * &q.log()).sum() * -1 // 2. broadcast
}

#[pyfunction]
pub fn softmax(x: Tensor, dim: usize) -> Tensor {
    // softmax(x)_i := exp(xᵢ) / Σⱼ exp(xⱼ)
    let exp = x.exp();
    let expsum = exp.sum(dim, true); // keepdim=true
    &exp / &expsum
}

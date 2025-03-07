use crate::ops::{Op, OpForwardError};
use crate::Tensor;
use cudarc::driver;
use thiserror::Error;

use cudarc::driver;
use once_cell::sync::Lazy;
use std::sync::Mutex;

static DEVICE: Lazy<Mutex<driver::CudaDevice>> =
    Lazy::new(|| Mutex::new(driver::CudaDevice::new(0).expect("Failed to initialize CUDA Device")));

#[pyfunction]
fn forward(op: &Op) -> PyResult<Tensor> {
    let device = DEVICE.lock().unwrap();
    // ...
    Ok(todo!())
}

pub fn forward_cuda(op: &Op) -> Result<Tensor, OpForwardError> {
    match op {
        Op::Add(x, y) => todo!(),
        Op::Sub(x, y) => todo!(),
        Op::Mul(x, y) => todo!(),
        Op::Div(x, y) => todo!(),
        Op::Matmul(x, y) => todo!(),
        Op::Neg(x) => todo!(),
        Op::Exp(x) => todo!(),
        Op::Log(x) => todo!(),
        Op::Sinh(x) => todo!(),
        Op::Cosh(x) => todo!(),
        Op::Tanh(x) => todo!(),
        Op::Sum(x, _, _) => todo!(),
    }
}

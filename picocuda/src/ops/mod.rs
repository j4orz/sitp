pub mod cpu_ops;
pub mod dfdx;
// pub mod cuda_ops;

use crate::{
    Device, DtypeVal,
    tpy::{self, ones},
    trs::Tensor,
};
use cpu_ops::{OpForwardError, ReduceDimInput, forward_cpu};
// use cuda_ops::forward_cuda;
use std::{
    collections::HashSet,
    ops::{Add, Div, Mul, Neg, Sub},
};

#[rustfmt::skip]
#[derive(Clone)]
pub enum Op {
    Add(Tensor, Tensor), Sub(Tensor, Tensor), Mul(Tensor, Tensor), Div(Tensor, Tensor), Matmul(Tensor, Tensor), // binops (zip)
    Neg(Tensor), Exp(Tensor), Log(Tensor), Sinh(Tensor), Cosh(Tensor), Tanh(Tensor), // uops (map)
    Sum(Tensor, Option<ReduceDimInput>), Max(Tensor, Option<ReduceDimInput>) // reduce
}

impl Op {
    pub fn inputs(&self) -> Vec<&Tensor> {
        // flatten inputs: tup -> vec
        match self {
            Op::Add(x, y) | Op::Sub(x, y) | Op::Mul(x, y) | Op::Div(x, y) | Op::Matmul(x, y) => {
                vec![x, y]
            }
            Op::Neg(x)
            | Op::Exp(x)
            | Op::Log(x)
            | Op::Sinh(x)
            | Op::Cosh(x)
            | Op::Tanh(x)
            // | Op::Mean(x) | Op::Var(x)
            => {
                vec![x]
            }
            Op::Sum(x, _) => vec![x],
            Op::Max(x, _) => vec![x]
        }
    }
}

impl Tensor {
    fn forward(&self, op: &Op) -> Result<Tensor, OpForwardError> {
        match self.device {
            Device::Cpu => forward_cpu(op),
            // Device::Gpu => forward_wgsl(op),
            // Device::Cuda => forward_cuda(op),
            _ => unimplemented!("picograd only supports cpu, gpu(opencl) or nv(cuda)"),
        }
    }

    pub fn backward(&self) -> () {
        self.storage.borrow_mut().grad = Some(ones(self.shape.clone()));
        for tensor in self.topo().iter().rev() {
            tensor._backward();
        }
    }

    fn _backward(&self) -> () {
        if self.input_op.is_none() {
            return;
        }

        let op = self.input_op.as_ref().unwrap();
        let grads = op.backward(self.storage.borrow().grad.as_ref().unwrap());

        // assuming grads.len() == op.inputs().len()
        for (input, dfdx_next) in op.inputs().into_iter().zip(grads.iter()) {
            let mut storage = input.storage.borrow_mut();
            match storage.grad {
                Some(ref mut dfdx_prev) => {
                    *dfdx_prev = (&*dfdx_prev + dfdx_next).unwrap();
                }
                None => {
                    storage.grad = Some(dfdx_next.clone());
                }
            }
        }
    }

    fn topo(&self) -> Vec<Tensor> {
        let (mut output, mut seen) = (Vec::new(), HashSet::new());
        Self::_topo(self, &mut output, &mut seen);
        output
    }

    fn _topo(tensor: &Tensor, output: &mut Vec<Tensor>, seen: &mut HashSet<Tensor>) {
        if seen.contains(&tensor) {
            return;
        }

        seen.insert(tensor.clone());
        if let Some(ref op) = tensor.input_op {
            for input in op.inputs() {
                Self::_topo(input, output, seen);
            }
        }
        output.push(tensor.clone());
    }
}

impl Neg for &Tensor {
    type Output = Result<Tensor, OpForwardError>;

    fn neg(self) -> Self::Output {
        let op = Op::Neg(self.clone());
        let output = self.forward(&op);
        output
    }
}

impl Add for &Tensor {
    type Output = Result<Tensor, OpForwardError>;

    fn add(self, input_other: &Tensor) -> Self::Output {
        let op = Op::Add(self.clone(), input_other.clone());
        let output = self.forward(&op);
        output
    }
}

impl Add<f32> for &Tensor {
    type Output = Result<Tensor, OpForwardError>;

    fn add(self, other: f32) -> Self::Output {
        let op = Op::Add(self.clone(), tpy::new(vec![DtypeVal::Float32(other)]));
        let output = self.forward(&op);
        output
    }
}

impl Sub for &Tensor {
    type Output = Result<Tensor, OpForwardError>;

    fn sub(self, input_other: &Tensor) -> Self::Output {
        let op = Op::Sub(self.clone(), input_other.clone());
        let output = self.forward(&op);
        output
    }
}

impl Sub<f32> for &Tensor {
    type Output = Result<Tensor, OpForwardError>;

    fn sub(self, other: f32) -> Self::Output {
        let op = Op::Sub(self.clone(), tpy::new(vec![DtypeVal::Float32(other)]));
        let output = self.forward(&op);
        output
    }
}
impl Mul for &Tensor {
    type Output = Result<Tensor, OpForwardError>;

    fn mul(self, other: &Tensor) -> Self::Output {
        let op = Op::Mul(self.clone(), other.clone());
        let output = self.forward(&op);
        output
    }
}

impl Mul<f32> for &Tensor {
    type Output = Result<Tensor, OpForwardError>;

    fn mul(self, other: f32) -> Self::Output {
        let op = Op::Mul(self.clone(), tpy::new(vec![DtypeVal::Float32(other)]));
        let output = self.forward(&op);
        output
    }
}

impl Div for &Tensor {
    type Output = Result<Tensor, OpForwardError>;

    fn div(self, other: &Tensor) -> Self::Output {
        let op = Op::Div(self.clone(), other.clone());
        let output = self.forward(&op);
        output
    }
}

impl Div<f32> for &Tensor {
    type Output = Result<Tensor, OpForwardError>;

    fn div(self, other: f32) -> Self::Output {
        let op = Op::Div(self.clone(), tpy::new(vec![DtypeVal::Float32(other)]));
        let output = self.forward(&op);
        output
    }
}
// note: picograd operations do not support `out` arg for "return oriented programming"

impl Tensor {
    // ***transcendental***
    pub fn exp(&self) -> Result<Tensor, OpForwardError> {
        let op = Op::Exp(self.clone());
        let output = self.forward(&op);
        output
    }

    pub fn log(&self) -> Result<Tensor, OpForwardError> {
        let op = Op::Log(self.clone());
        let output = self.forward(&op);
        output
    }

    // ***linear/non-linear***
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, OpForwardError> {
        let op = Op::Matmul(self.clone(), other.clone());
        let output = self.forward(&op);
        output
    }

    pub fn sinh(&self) -> Result<Tensor, OpForwardError> {
        let op = Op::Sinh(self.clone());
        let output = self.forward(&op);
        output
    }

    pub fn cosh(&self) -> Result<Tensor, OpForwardError> {
        let op = Op::Cosh(self.clone());
        let output = self.forward(&op);
        output
    }

    pub fn tanh(&self) -> Result<Tensor, OpForwardError> {
        let op = Op::Tanh(self.clone());
        let output = self.forward(&op);
        output
    }

    // ***reductions***
    pub fn _sum(&self, rdi: Option<ReduceDimInput>) -> Result<Tensor, OpForwardError> {
        let op = Op::Sum(self.clone(), rdi);
        let output = self.forward(&op);
        output
    }

    pub fn max(&self, dim: usize, keepdim: bool) -> Result<Tensor, OpForwardError> {
        let rdi = ReduceDimInput { dim, keepdim };
        let op = Op::Max(self.clone(), Some(rdi));
        let output = self.forward(&op);
        output
    }
}

pub mod cpu;
pub mod cuda;
pub mod gpu;

use crate::{DtypeVal, Tensor};
use cpu::{forward_cpu, OpForwardError};
use std::{
    hash,
    ops::{Add, Div, Mul, Neg, Sub},
};

#[rustfmt::skip]
#[derive(Clone, Debug)]
pub enum Op {
    // ***desugared core***
    Add(Tensor, Tensor), Sub(Tensor, Tensor), Mul(Tensor, Tensor), Div(Tensor, Tensor), Matmul(Tensor, Tensor), // binary maps: algebraic
    // ***sugar***
    // (can be desugared to algebraic via power series — contra calc teachers taylor series)
    Neg(Tensor), Exp(Tensor), Log(Tensor), Sinh(Tensor), Cosh(Tensor), Tanh(Tensor), // unary zips: transcendental
    Sum(Tensor, usize, bool),
    // Mean(Tensor), Var(Tensor), // statistics
    // Dot
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
            Op::Sum(x, _, _) => vec![x],
        }
    }
}

impl PartialEq for Op {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other) // reference comparison instead of value since f32 is not Eq.
    }
}

impl Eq for Op {}

impl hash::Hash for Op {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        (self as *const Self).hash(state);
    }
}

impl Tensor {
    fn forward(&self, op: &Op) -> Result<Tensor, OpForwardError> {
        match self.device {
            crate::Device::Cpu => forward_cpu(op),
            _ => todo!(), // crate::Device::Gpu => forward_gpu(op),
                          // crate::Device::Cuda => forward_cuda(op),
        }
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
        let op = Op::Add(self.clone(), crate::new(vec![DtypeVal::Float32(other)]));
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
        let op = Op::Sub(self.clone(), crate::new(vec![DtypeVal::Float32(other)]));
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
        let op = Op::Mul(self.clone(), crate::new(vec![DtypeVal::Float32(other)]));
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
        let op = Op::Div(self.clone(), crate::new(vec![DtypeVal::Float32(other)]));
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
    pub fn sum(&self, dim: usize, keepdim: bool) -> Result<Tensor, OpForwardError> {
        let op = Op::Sum(self.clone(), dim, keepdim);
        let output = self.forward(&op);
        output
    }

    // pub fn max(&self, dim: i32, keepdim: bool) -> Result<Tensor, OpForwardError> {
    //     todo!()
    // }

    // pub fn min(&self, dim: i32, keepdim: bool) -> Result<Tensor, OpForwardError> {
    //     todo!()
    // }

    // pub fn mean(&self, dim: usize) -> Result<Tensor, OpForwardError> {
    //     todo!()
    // }

    // pub fn var(&self, dim: usize) -> Result<Tensor, OpForwardError> {
    //     todo!()
    // }
}

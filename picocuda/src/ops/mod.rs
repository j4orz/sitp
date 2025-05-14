//! ops module:
//!
//! - cpu_ops: impl <...> for DTypeVal (map/zip/reduce)

pub mod cpu_ops;
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
        self._backward();
        for tensor in self.topo().iter().rev() {
            tensor._backward();
        }
    }

    fn _backward(&self) -> () {
        // NB: autodifferentiation's .backward() is defined on Ops, not Tensors.
        // since the gradient gives us the perturbation sensitivty a function's
        // input has on the final loss, it would be clearer mathematically if
        // .grad lived on Op, not Tensor.
        if self.input_op.is_none() { return }
        let op = self.input_op.as_ref().unwrap();
        let storage_ref = self.storage.borrow(); // lifetime needs to be extended
        let dfdx_cached = storage_ref.grad.as_ref().unwrap();

        // evaluate local derivatives via chain rule
        let local_grads = backward_cpu(&op, &self, dfdx_cached);

        // propagate derivative to inputs assuming grads.len() == op.inputs().len()
        for (x, dfdx_next) in op.inputs().into_iter().zip(local_grads.iter()) {
            let mut storage = x.storage.borrow_mut();
            match storage.grad {
                Some(ref mut dfdx_prev) => *dfdx_prev = (&*dfdx_prev + dfdx_next).unwrap(),
                None => storage.grad = Some(dfdx_next.clone()),
            }
        }
    }

    fn topo(&self) -> Vec<Tensor> {
        let (mut output, mut seen) = (Vec::new(), HashSet::new());
        Self::_visit(self, &mut output, &mut seen);
        output
    }

    fn _visit(tensor: &Tensor, output: &mut Vec<Tensor>, seen: &mut HashSet<Tensor>) {
        if seen.contains(&tensor) { return }
        seen.insert(tensor.clone());
        if let Some(ref op) = tensor.input_op {
            for input in op.inputs() { Self::_visit(input, output, seen) }
        }
        output.push(tensor.clone());
    }
}

// dFdx is the global derivative (backpropagated to the current op)
// dfdx is the local derivative
pub fn backward_cpu(op: &Op, opout: &Tensor, dFdx: &Tensor) -> Vec<Tensor> {
    match op {
        Op::Add(_x, _y) => vec![
            (1.0 * &dFdx.clone()).unwrap(),
            (1.0 * &dFdx.clone()).unwrap(),
        ],
        Op::Sub(x, y) => todo!(),
        Op::Mul(x, y) => vec![(y * &dFdx.clone()).unwrap(), (x * &dFdx.clone()).unwrap()],
        Op::Div(x, y) => todo!(),
        Op::Matmul(x, y) => todo!(),
        Op::Neg(x) => todo!(),
        Op::Exp(x) => todo!(),
        Op::Log(x) => todo!(),
        Op::Sinh(x) => todo!(),
        Op::Cosh(x) => todo!(),
        Op::Tanh(x) => {
            // d(tanh)dx: 1-tanh(x)^2
            let tanh_x = opout;
            let (ones_tensor, tanh_squared) =
                (ones(tanh_x.shape.clone()), (tanh_x * tanh_x).unwrap());
            let dfdx = (&ones_tensor - &tanh_squared).unwrap();

            vec![(&dfdx * &dFdx.clone()).unwrap()] // chain rule
        }
        Op::Sum(x, reduce_dim_input) => todo!(),
        Op::Max(x, reduce_dim_input) => todo!(),
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

    fn add(self, rhs: &Tensor) -> Self::Output {
        let op = Op::Add(self.clone(), rhs.clone());
        let output = self.forward(&op);
        output
    }
}

impl Add<f32> for &Tensor {
    type Output = Result<Tensor, OpForwardError>;

    fn add(self, rhs: f32) -> Self::Output {
        let op = Op::Add(self.clone(), tpy::new(vec![DtypeVal::Float32(rhs)]));
        let output = self.forward(&op);
        output
    }
}

impl Sub for &Tensor {
    type Output = Result<Tensor, OpForwardError>;

    fn sub(self, rhs: &Tensor) -> Self::Output {
        let op = Op::Sub(self.clone(), rhs.clone());
        let output = self.forward(&op);
        output
    }
}

impl Sub<f32> for &Tensor {
    type Output = Result<Tensor, OpForwardError>;

    fn sub(self, rhs: f32) -> Self::Output {
        let op = Op::Sub(self.clone(), tpy::new(vec![DtypeVal::Float32(rhs)]));
        let output = self.forward(&op);
        output
    }
}
impl Mul for &Tensor {
    type Output = Result<Tensor, OpForwardError>;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        let op = Op::Mul(self.clone(), rhs.clone());
        let output = self.forward(&op);
        output
    }
}

impl Mul<f32> for &Tensor {
    type Output = Result<Tensor, OpForwardError>;

    fn mul(self, rhs: f32) -> Self::Output {
        let op = Op::Mul(self.clone(), tpy::new(vec![DtypeVal::Float32(rhs)]));
        let output = self.forward(&op);
        output
    }
}

impl Mul<&Tensor> for f32 {
    type Output = Result<Tensor, OpForwardError>;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        let op = Op::Mul(tpy::new(vec![DtypeVal::Float32(self)]), rhs.clone());
        let output = rhs.forward(&op);
        output
    }
}

impl Div for &Tensor {
    type Output = Result<Tensor, OpForwardError>;

    fn div(self, rhs: &Tensor) -> Self::Output {
        let op = Op::Div(self.clone(), rhs.clone());
        let output = self.forward(&op);
        output
    }
}

impl Div<f32> for &Tensor {
    type Output = Result<Tensor, OpForwardError>;

    fn div(self, rhs: f32) -> Self::Output {
        let op = Op::Div(self.clone(), tpy::new(vec![DtypeVal::Float32(rhs)]));
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

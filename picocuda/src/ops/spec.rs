use crate::{
    Device, DtypeVal,
    ops::{Op, cpu_ops},
    tpy::{self, ones},
    trs::Tensor,
};
use cpu_ops::{OpForwardError, ReduceDimInput, backward_cpu, forward_cpu};
// use cuda_ops::forward_cuda;
use std::{
    collections::HashSet,
    ops::{Add, Div, Mul, Neg, Sub},
};

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
        // NB: backward_cpu and autodifferentiation is defined on Ops, not Tensors.
        // since the gradient gives us the perturbation sensitivty a function's
        // input has on the final loss. it would be clearer mathematically if
        // .grad lived on Op, not Tensor.
        if self.input_op.is_none() {
            return;
        }
        let op = self.input_op.as_ref().unwrap();
        let storage_ref = self.storage.borrow(); // lifetime needs to be extended
        let dfdx_cached = storage_ref.grad.as_ref().unwrap();

        // evaluate local derivatives via chain rule
        let local_grads = backward_cpu(&op, &self, dfdx_cached);

        // propagate derivative to inputs assuming grads.len() == op.inputs().len()
        for (x, dfdx_next) in op.inputs().into_iter().zip(local_grads.iter()) {
            let mut storage = x.storage.borrow_mut();
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

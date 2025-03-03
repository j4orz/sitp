use crate::{
    ops::{Op, OpForwardError},
    Tensor,
};
use wgpu::{self, Instance, InstanceDescriptor};

pub fn forward_wgsl(op: &Op) -> Result<Tensor, OpForwardError> {
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

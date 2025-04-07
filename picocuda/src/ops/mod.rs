//! ops module:
//!
//! - mod.rs: contains the definition (IR) of Op
//! - spec.rs: impl <...> for Tensor
//! - cpu_ops: impl <...> for DTypeVal (map/zip/reduce)

pub mod cpu_ops;
pub mod spec;
use crate::{ops::cpu_ops::ReduceDimInput, trs::Tensor};

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

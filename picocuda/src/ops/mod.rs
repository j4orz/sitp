pub mod cpu;
pub mod cuda;
pub mod gpu;

use crate::Tensor;
use std::hash;

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

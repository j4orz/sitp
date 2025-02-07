use std::{
    cell::RefCell,
    collections::HashSet,
    hash,
    ops::{Add, Mul},
    rc::Rc,
};

use crate::{Storage, Tensor};

#[rustfmt::skip]
#[derive(Clone, Debug)]
pub enum Op {
    // ***desugared core*** (trascendental ops can be desugared to algebraic via power series (not taylor, contra calc teachers))
    Add(Tensor, Tensor), Sub(Tensor, Tensor), Mul(Tensor, Tensor), Div(Tensor, Tensor), // algebraic
    // ***sugar***
    Sin(Tensor), Cos(Tensor), Exp(Tensor), Log(Tensor), // transcendental
    Matmul(Tensor, Tensor), Tanh(Tensor), // linear/nonlinear
    Mean(Tensor), Var(Tensor), // statistics
}

impl Op {
    fn inputs(&self) -> Vec<&Tensor> {
        // flatten inputs: tup -> vec
        match self {
            Op::Add(x, y) | Op::Sub(x, y) | Op::Mul(x, y) | Op::Div(x, y) | Op::Matmul(x, y) => {
                vec![x, y]
            }
            Op::Sin(x)
            | Op::Cos(x)
            | Op::Exp(x)
            | Op::Log(x)
            | Op::Tanh(x)
            | Op::Mean(x)
            | Op::Var(x) => {
                vec![x]
            }
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

// *****************************************************************************************************************
// ******************************************** .forward() *********************************************
// *****************************************************************************************************************

impl Op {
    fn forward(&self) -> Tensor {
        match &self {
            Op::Add(x, y) => self.apply_binary_op(|xi, yi| xi + yi, x, y),
            Op::Sub(x, y) => self.apply_binary_op(|xi, yi| xi - yi, x, y),
            Op::Mul(x, y) => self.apply_binary_op(|xi, yi| xi * yi, x, y),
            Op::Div(x, y) => self.apply_binary_op(|xi, yi| xi / yi, x, y),
            Op::Sin(x) => todo!(),
            Op::Cos(x) => todo!(),
            Op::Exp(x) => todo!(),
            Op::Log(x) => todo!(),
            Op::Matmul(x, y) => todo!(),
            Op::Tanh(x) => todo!(),
            Op::Mean(x) => todo!(),
            Op::Var(x) => todo!(),
        }
    }

    fn apply_binary_op<F>(&self, f: F, x: &Tensor, y: &Tensor) -> Tensor
    where
        F: Fn(f32, f32) -> f32,
    {
        assert_eq!(x.shape, y.shape, "Shape mismatch in operation");

        Tensor {
            shape: x.shape.clone(),
            stride: x.stride.clone(),
            input_op: Some(Box::new(self.clone())), // Box since Op owns Tensors
            // alloc new storage
            storage: Rc::new(RefCell::new(Storage {
                data: x
                    .storage
                    .borrow()
                    .data
                    .iter()
                    .zip(y.storage.borrow().data.iter())
                    .map(|(&xi, &yi)| f(xi, yi))
                    .collect(),
                grad: None,
            })),
            device: x.device.clone(),
            layout: x.layout.clone(),
            dtype: x.dtype.clone(),
        }
    }
}

impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, input_other: &Tensor) -> Self::Output {
        let op = Op::Add(self.clone(), input_other.clone());
        let output = op.forward();
        output
    }
}

impl Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, other: &Tensor) -> Self::Output {
        let op = Op::Mul(self.clone(), other.clone());
        let output = op.forward();
        output
    }
}

impl Tensor {
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        todo!()
    }

    pub fn sin(&self) -> Tensor {
        todo!()
    }

    pub fn cos(&self) -> Tensor {
        todo!()
    }

    pub fn exp(&self) -> Tensor {
        todo!()
    }

    pub fn log(&self) -> Tensor {
        todo!()
    }

    pub fn tanh(&self) -> Tensor {
        todo!()
    }

    pub fn mean(&self) -> Tensor {
        todo!()
    }

    pub fn var(&self) -> Tensor {
        todo!()
    }
}

// *****************************************************************************************************************
// ******************************************** .backward() *********************************************
// *****************************************************************************************************************

// autodifferentiation is algorithmic but uses numeric objects, not symbolic.
impl Tensor {
    // forward mode: implicit graph: (V,E) = [f(x)=TODO, (f'(x)=path_i)edge_i]
    // backward mode: explicit graph 1. .forward() 2. .backward()
    // -> deep learning uses reverse mode because the function class of neural networks
    // usually has more inputs (sources) than outputs (sinks) due to the lifting of representations/motifs/embeddings
    // reverse mode lets us reuse the outputs and keeping the computation for the massive input fan in shallow insteaad
    // of forwards deep computation
    // -> forward mode for neural networks is naive recursion
    // -> backward mode for neural networks is reuse of dependencies (dp)

    // .grad is sum over product of path weights
    //          \sigma_{path0...path1}(\prod_{pathi} ei)

    // reversemode/forwardmode is same for \mathbb{R} because of associativity
    // but with \mahtbb{R^{nxm}} you have to make sure matrices are associative
    pub fn backward(&mut self) -> () {
        self.storage.borrow_mut().grad = Some(Tensor::ones(&self.shape)); // base case dfdx
        let (mut topo, mut visited) = (Vec::new(), HashSet::new());
        self.topo(&mut topo, &mut visited);

        for tensor in topo.into_iter() {
            if let Some(ref input_op) = tensor.input_op {
                let (inputs, input_grads) = (input_op.inputs(), input_op.backward(&tensor));

                for (x, dfdx_next) in inputs.into_iter().zip(input_grads.iter()) {
                    let mut storage = x.storage.borrow_mut();
                    match storage.grad {
                        Some(ref mut dfdx_prev) => {
                            *dfdx_prev = &*dfdx_prev + dfdx_next;
                        }
                        None => {
                            storage.grad = Some(dfdx_next.clone());
                        }
                    }
                }
            }
        }
    }

    fn topo(&self, topo: &mut Vec<Tensor>, visited: &mut HashSet<Op>) {
        match self.input_op {
            Some(ref op) => {
                if visited.contains(&op) {
                    // todo: op or tensor?
                    return;
                }
                visited.insert(*op.clone());
                for input in op.inputs() {
                    input.topo(topo, visited);
                }
            }
            None => {}
        }

        topo.push(self.clone());
    }
}

impl Op {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        match self {
            Op::Add(_x, _y) => {
                // d/dx(x + y) = 1, d/dy(x + y) = 1
                let (dx, dy) = (Tensor::ones(&grad.shape), Tensor::ones(&grad.shape));
                vec![&dx * &grad.clone(), &dy * &grad.clone()]
            }
            Op::Sub(_x, _y) => {
                // d/dx(x - y) = 1, d/dy(x - y) = -1
                let (dx, dy) = (
                    Tensor::ones(&grad.shape),
                    Tensor::new(vec![-1.0; grad.numel()]),
                );
                vec![&dx * &grad.clone(), &dy * &grad.clone()]
            }
            Op::Mul(x, y) => {
                // d/dx(x * y) = y, d/dy(x * y) = x
                let (dx, dy) = (y, x);
                vec![dx * &grad.clone(), dy * &grad.clone()]
            }
            Op::Div(x, y) => todo!(),
            Op::Sin(x) => todo!(),
            Op::Cos(x) => todo!(),
            Op::Exp(x) => todo!(),
            Op::Log(x) => todo!(),
            Op::Matmul(x, y) => todo!(),
            Op::Tanh(x) => todo!(),
            Op::Mean(x) => todo!(),
            Op::Var(x) => todo!(),
        }
    }
}

// impl Index<(usize, usize)> for Tensor {
//     type Output = f32; // fixed to fp32 for now

//     fn index(&self, index: (usize, usize)) -> &Self::Output {
//         let (i, j) = index;
//         let idx = i * self.shape[1] + j; // Row-major ordering
//         &self.data[idx]
//     }
// }

// impl IndexMut<(usize, usize)> for Tensor {
//     fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
//         let (i, j) = index;
//         let idx = i * self.shape[1] + j; // Row-major ordering
//         &mut self.data[idx]
//     }
// }

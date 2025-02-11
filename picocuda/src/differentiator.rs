use std::{
    cell::RefCell,
    collections::HashSet,
    hash,
    ops::{Add, Div, Mul, Neg},
    rc::Rc,
};

use crate::{Storage, Tensor};

#[rustfmt::skip]
#[derive(Clone, Debug)]
pub enum Op {
    // ***desugared core***
    Add(Tensor, Tensor), Sub(Tensor, Tensor), Mul(Tensor, Tensor), Div(Tensor, Tensor), Matmul(Tensor, Tensor), // algebraic
    // ***sugar***
    Neg(Tensor), Exp(Tensor), Log(Tensor), Sinh(Tensor), Cosh(Tensor), Tanh(Tensor), // transcendental (can be desugared to algebraic via power serie — not taylor, contra calc teachers)
    // Mean(Tensor), Var(Tensor), // statistics
    // Dot
}

impl Op {
    fn inputs(&self) -> Vec<&Tensor> {
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
            Op::Matmul(X, Y) => {
                // 1. def O(n^3)
                // 2. data oriented(cache)/pthreads/SIMD
                let (n, m1, m2, p) = (X.shape[0], X.shape[1], Y.shape[0], Y.shape[1]);
                assert_eq!(m1, m2, "Shape mismatch in operation");
                assert_eq!(X.ndim, 2, "X must be a 2D tensor");
                assert_eq!(Y.ndim, 2, "Y must be a 2D tensor");

                let Z = Tensor::zeros(&[n, p]);

                {
                    let (X_storage, Y_storage, mut Z_storage) = (
                        X.storage.borrow(),
                        Y.storage.borrow(),
                        Z.storage.borrow_mut(),
                    );

                    for i in 0..n {
                        for j in 0..p {
                            // linear combination of p basis vectors in R^m mapped to
                            // X[n][m] * Y[m][p]

                            // [n][m]: m basis vectors in R^n
                            // [m][p]: p basis vectors in R^m
                            for k in 0..m1 {
                                let x = X_storage.data[i * X.stride[0] + k * X.stride[1]];
                                let y = Y_storage.data[k * Y.stride[0] + j * Y.stride[1]];
                                Z_storage.data[i * Z.stride[0] + j * Z.stride[1]] += x * y;
                            }
                        }
                    }
                }

                Z
            }
            Op::Neg(x) => {
                // TODO?: can desugar to mul, just like tanh(x) := div(sinh(x), cosh(x))
                // let op = Op::Mul(x.clone(), Tensor::new(vec![-1.0; x.numel()]));
                // let y = op.forward();
                // y
                let y = Tensor::zeros(&x.shape);
                {
                    let (x_storage, mut z_storage) = (x.storage.borrow(), y.storage.borrow_mut());

                    for (i, &xi) in x_storage.data.iter().enumerate() {
                        z_storage.data[i] = -xi;
                    }
                }
                y
            }
            Op::Exp(x) => {
                let y = Tensor::zeros(&x.shape);
                {
                    let (x_storage, mut z_storage) = (x.storage.borrow(), y.storage.borrow_mut());

                    for (i, &xi) in x_storage.data.iter().enumerate() {
                        z_storage.data[i] = xi.exp();
                    }
                }
                y
            }
            Op::Log(x) => {
                let y = Tensor::zeros(&x.shape);
                {
                    let (x_storage, mut z_storage) = (x.storage.borrow(), y.storage.borrow_mut());

                    for (i, &xi) in x_storage.data.iter().enumerate() {
                        z_storage.data[i] = xi.ln();
                    }
                }
                y
            }
            Op::Sinh(x) => {
                let y = Tensor::zeros(&x.shape);
                {
                    let (x_storage, mut z_storage) = (x.storage.borrow(), y.storage.borrow_mut());

                    for (i, &xi) in x_storage.data.iter().enumerate() {
                        z_storage.data[i] = xi.sinh();
                    }
                }
                y
            }
            Op::Cosh(x) => {
                let y = Tensor::zeros(&x.shape);
                {
                    let (x_storage, mut z_storage) = (x.storage.borrow(), y.storage.borrow_mut());

                    for (i, &xi) in x_storage.data.iter().enumerate() {
                        z_storage.data[i] = xi.cosh();
                    }
                }
                y
            }
            Op::Tanh(x) => {
                // let op = Op::Div(Op::Sinh(x.clone()).forward(), Op::Cosh(x.clone()).forward());
                // op.forward()
                let y = Tensor::zeros(&x.shape);
                {
                    let (x_storage, mut z_storage) = (x.storage.borrow(), y.storage.borrow_mut());

                    for (i, &xi) in x_storage.data.iter().enumerate() {
                        z_storage.data[i] = xi.tanh();
                    }
                }
                y
            } // Op::Mean(x) => todo!(),
              // Op::Var(x) => todo!(),
        }
    }

    fn apply_binary_op<F>(&self, f: F, x: &Tensor, y: &Tensor) -> Tensor
    where
        F: Fn(f32, f32) -> f32,
    {
        assert_eq!(x.shape, y.shape, "Shape mismatch in operation");

        Tensor {
            ndim: x.ndim,
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

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        let op = Op::Neg(self.clone());
        let output = op.forward();
        output
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

impl Div for &Tensor {
    type Output = Tensor;

    fn div(self, other: &Tensor) -> Self::Output {
        let op = Op::Div(self.clone(), other.clone());
        let output = op.forward();
        output
    }
}

impl Tensor {
    // ***transcendental***
    pub fn exp(&self) -> Tensor {
        let op = Op::Exp(self.clone());
        let output = op.forward();
        output
    }

    pub fn log(&self) -> Tensor {
        let op = Op::Log(self.clone());
        let output = op.forward();
        output
    }

    // ***linear/non-linear***
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let op = Op::Matmul(self.clone(), other.clone());
        let output = op.forward();
        output
    }

    pub fn sinh(&self) -> Tensor {
        let op = Op::Sinh(self.clone());
        let output = op.forward();
        output
    }

    pub fn cosh(&self) -> Tensor {
        let op = Op::Cosh(self.clone());
        let output = op.forward();
        output
    }

    pub fn tanh(&self) -> Tensor {
        let op = Op::Tanh(self.clone());
        let output = op.forward();
        output
    }

    // ***reductions***
    pub fn sum(&self, dim: i32, keepdim: bool) -> Tensor {
        todo!()
    }

    // pub fn max(&self, dim: i32, keepdim: bool) -> Tensor {
    //     todo!()
    // }

    // pub fn min(&self, dim: i32, keepdim: bool) -> Tensor {
    //     todo!()
    // }

    // pub fn mean(&self, dim: usize) -> Tensor {
    //     todo!()
    // }

    // pub fn var(&self, dim: usize) -> Tensor {
    //     todo!()
    // }
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
                let (inputs, input_grads) = (
                    input_op.inputs(),
                    input_op.backward(&tensor.storage.borrow().grad.as_ref().unwrap()),
                );

                for (x, dfdx_next) in inputs.into_iter().zip(input_grads.iter()) {
                    let mut storage = x.storage.borrow_mut();
                    match storage.grad {
                        Some(ref mut dfdx_prev) => {
                            *dfdx_prev = &*dfdx_prev + dfdx_next; // todo: dfdx_next.detatch()?
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
            Op::Matmul(x, y) => todo!(),
            Op::Neg(x) => todo!(),
            Op::Exp(x) => todo!(),
            Op::Log(x) => todo!(),
            Op::Sinh(x) => todo!(),
            Op::Cosh(x) => todo!(),
            Op::Tanh(x) => todo!(),
            // Op::Mean(x) => todo!(),
            // Op::Var(x) => todo!(),
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

use crate::{Dtype, DtypeVal, Storage, Tensor, TensorError};
use std::{
    cell::RefCell,
    collections::HashSet,
    hash,
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};
use thiserror::Error;

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

impl Add for DtypeVal {
    type Output = DtypeVal;

    fn add(self, other: Self) -> Self::Output {
        match (self, other) {
            (DtypeVal::Float32(x), DtypeVal::Float32(y)) => DtypeVal::Float32(x + y),
            (DtypeVal::Int32(x), DtypeVal::Int32(y)) => DtypeVal::Int32(x + y),
            _ => todo!(),
        }
    }
}

impl Sub for DtypeVal {
    type Output = DtypeVal;

    fn sub(self, other: Self) -> Self::Output {
        match (self, other) {
            (DtypeVal::Float32(x), DtypeVal::Float32(y)) => DtypeVal::Float32(x - y),
            (DtypeVal::Int32(x), DtypeVal::Int32(y)) => DtypeVal::Int32(x - y),
            _ => todo!(),
        }
    }
}

impl Mul for DtypeVal {
    type Output = DtypeVal;

    fn mul(self, other: Self) -> Self::Output {
        match (self, other) {
            (DtypeVal::Float32(x), DtypeVal::Float32(y)) => DtypeVal::Float32(x * y),
            (DtypeVal::Int32(x), DtypeVal::Int32(y)) => DtypeVal::Int32(x * y),
            _ => todo!(),
        }
    }
}

impl Div for DtypeVal {
    type Output = DtypeVal;

    fn div(self, other: Self) -> Self::Output {
        match (self, other) {
            (DtypeVal::Float32(x), DtypeVal::Float32(y)) => DtypeVal::Float32(x / y),
            (DtypeVal::Int32(x), DtypeVal::Int32(y)) => DtypeVal::Int32(x / y),
            _ => todo!(),
        }
    }
}

impl Neg for DtypeVal {
    type Output = DtypeVal;

    fn neg(self) -> Self::Output {
        match self {
            DtypeVal::Float32(x) => DtypeVal::Float32(-x),
            DtypeVal::Int32(x) => DtypeVal::Int32(-x),
            _ => todo!(),
        }
    }
}

// *****************************************************************************************************************
// ******************************************** .forward() *********************************************
// *****************************************************************************************************************

#[derive(Error, Debug)]
pub enum OpForwardError {
    #[error(transparent)]
    TensorError(#[from] TensorError),
    #[error("unknown operation error")]
    Unknown,
}

impl Op {
    fn forward(&self) -> Result<Tensor, OpForwardError> {
        match &self {
            Op::Add(x, y) => self.zip(|xi, yi| xi + yi, x, y),
            Op::Sub(x, y) => self.zip(|xi, yi| xi - yi, x, y),
            Op::Mul(x, y) => self.zip(|xi, yi| xi * yi, x, y),
            Op::Div(x, y) => self.zip(|xi, yi| xi / yi, x, y),
            // TODO?: can desugar to mul, just like tanh(x) := div(sinh(x), cosh(x))
            // let op = Op::Mul(x.clone(), Tensor::new(vec![-1.0; x.numel()]));
            // let y = op.forward();
            // y
            Op::Neg(x) => self.map(|xi| -xi, x),
            Op::Exp(x) => self.map(|xi| DtypeVal::Float32(f32::from(xi).exp()), x),
            Op::Log(x) => self.map(|xi| DtypeVal::Float32(f32::from(xi).ln()), x),
            Op::Sinh(x) => self.map(|xi| DtypeVal::Float32(f32::from(xi).sinh()), x),
            Op::Cosh(x) => self.map(|xi| DtypeVal::Float32(f32::from(xi).cosh()), x),
            Op::Tanh(x) => self.map(|xi| DtypeVal::Float32(f32::from(xi).tanh()), x),
            // sigmoid
            // relu
            Op::Sum(x, dim, keepdim) => self.reduce(|xi, yi| xi + yi, x, *dim, *keepdim),
            // max
            // min
            // mean
            // var
            Op::Matmul(X, Y) => {
                // 1. def O(n^3)
                // 2. data oriented(cache)/pthreads/SIMD
                let (n, m1, m2, p) = (X.shape[0], X.shape[1], Y.shape[0], Y.shape[1]);
                assert_eq!(m1, m2, "Shape mismatch in operation");
                assert_eq!(X.ndim, 2, "X must be a 2D tensor");
                assert_eq!(Y.ndim, 2, "Y must be a 2D tensor");

                // let Z = Tensor::zeros(&[n, p]);

                // {
                //     let (X_storage, Y_storage, mut Z_storage) = (
                //         X.storage.borrow(),
                //         Y.storage.borrow(),
                //         Z.storage.borrow_mut(),
                //     );

                //     for i in 0..n {
                //         for j in 0..p {
                //             // linear combination of p basis vectors in R^m mapped to
                //             // X[n][m] * Y[m][p]

                //             // [n][m]: m basis vectors in R^n
                //             // [m][p]: p basis vectors in R^m
                //             for k in 0..m1 {
                //                 let x = X_storage.data[i * X.stride[0] + k * X.stride[1]];
                //                 let y = Y_storage.data[k * Y.stride[0] + j * Y.stride[1]];
                //                 Z_storage.data[i * Z.stride[0] + j * Z.stride[1]] += x * y;
                //             }
                //         }
                //     }
                // }

                // Ok(Z)
                todo!()
            }
        }
    }

    // picograd does not support broadcastable map implementation with return-oriented "out" arg
    fn map<F>(&self, f: F, x: &Tensor) -> Result<Tensor, OpForwardError>
    where
        F: Fn(DtypeVal) -> DtypeVal,
    {
        Ok(Tensor {
            ndim: x.ndim,
            shape: x.shape.clone(),
            stride: x.stride.clone(),
            input_op: Some(Box::new(self.clone())), // Box since Op owns Tensors
            // alloc new storage
            storage: Rc::new(RefCell::new(Storage {
                data: x.storage.borrow().data.iter().map(|&xi| f(xi)).collect(),
                grad: None,
            })),
            device: x.device.clone(),
            layout: x.layout.clone(),
            dtype: x.dtype.clone(),
        })
    }

    // 1. z_shape <- broadcast_shape(x.shape, y.shape)
    // 2. for i in z.numel():

    //      **A: find the logical nd index of x and y**
    //      3. logz <- encode(i)
    //      4. (logx, logy) <- broadcast_logidx(logx, logz), broadcast_logidx(logy, logz)

    //      **B: perform update**
    //      5. physx, physy, physz <- decode(logx), decode(logy), decode(logz)
    //      6. z[physz] <- f(x[physx], y[physy])
    fn zip<F>(&self, f: F, x: &Tensor, y: &Tensor) -> Result<Tensor, OpForwardError>
    where
        F: Fn(DtypeVal, DtypeVal) -> DtypeVal,
    {
        let z_shape = if x.shape != y.shape {
            Tensor::broadcast_shape(&x.shape, &y.shape)?
        } else {
            x.shape.clone()
        };
        let z = Tensor::zeros(&z_shape, Dtype::Float32);

        {
            let (x_storage, y_storage, mut z_storage) = (
                x.storage.borrow(),
                y.storage.borrow(),
                z.storage.borrow_mut(),
            );

            let (logx, logy) = (vec![0 as usize; x.numel()], vec![0 as usize; y.numel()]);
            for phy in 0..z.numel() {
                // map logz -> (logx, logy)
                let logz = Tensor::encode(phy, &z.shape);
                let (logx, logy) = (
                    Tensor::broadcast_logidx(&logx, &logz)?,
                    Tensor::broadcast_logidx(&logy, &logz)?,
                );

                let (physx, physy, physz) = (
                    Tensor::decode(&logx, &x.stride),
                    Tensor::decode(&logy, &y.stride),
                    Tensor::decode(&logz, &z.stride),
                );

                z_storage.data[physz] = f(x_storage.data[physx], y_storage.data[physy]);
            }
        }

        Ok(z)
    }

    fn reduce<F>(
        &self,
        f: F,
        x: &Tensor,
        dim: usize,
        keepdim: bool,
    ) -> Result<Tensor, OpForwardError>
    where
        F: Fn(DtypeVal, DtypeVal) -> DtypeVal,
    {
        todo!()
    }
}

impl Neg for &Tensor {
    type Output = Result<Tensor, OpForwardError>;

    fn neg(self) -> Self::Output {
        let op = Op::Neg(self.clone());
        let output = op.forward();
        output
    }
}

impl Add for &Tensor {
    type Output = Result<Tensor, OpForwardError>;

    fn add(self, input_other: &Tensor) -> Self::Output {
        let op = Op::Add(self.clone(), input_other.clone());
        let output = op.forward();
        output
    }
}

impl Mul for &Tensor {
    type Output = Result<Tensor, OpForwardError>;

    fn mul(self, other: &Tensor) -> Self::Output {
        let op = Op::Mul(self.clone(), other.clone());
        let output = op.forward();
        output
    }
}

// #[derive(Clone, Copy, Debug)]
// pub struct Scalar(pub f32); // how does wrapper type interop with pyo3 bindings?

// impl Mul<&Tensor> for Scalar {
//     type Output = Tensor;

//     fn mul(self, rhs: Self) -> Self::Output {
//         todo!()
//     }
// }

impl Div for &Tensor {
    type Output = Result<Tensor, OpForwardError>;

    fn div(self, other: &Tensor) -> Self::Output {
        let op = Op::Div(self.clone(), other.clone());
        let output = op.forward();
        output
    }
}

// note: picograd operations do not support `out` arg for "return oriented programming"
impl Tensor {
    // ***transcendental***
    pub fn exp(&self) -> Result<Tensor, OpForwardError> {
        let op = Op::Exp(self.clone());
        let output = op.forward();
        output
    }

    pub fn log(&self) -> Result<Tensor, OpForwardError> {
        let op = Op::Log(self.clone());
        let output = op.forward();
        output
    }

    // ***linear/non-linear***
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, OpForwardError> {
        let op = Op::Matmul(self.clone(), other.clone());
        let output = op.forward();
        output
    }

    pub fn sinh(&self) -> Result<Tensor, OpForwardError> {
        let op = Op::Sinh(self.clone());
        let output = op.forward();
        output
    }

    pub fn cosh(&self) -> Result<Tensor, OpForwardError> {
        let op = Op::Cosh(self.clone());
        let output = op.forward();
        output
    }

    pub fn tanh(&self) -> Result<Tensor, OpForwardError> {
        let op = Op::Tanh(self.clone());
        let output = op.forward();
        output
    }

    // ***reductions***
    pub fn sum(&self, dim: usize, keepdim: bool) -> Result<Tensor, OpForwardError> {
        let op = Op::Sum(self.clone(), dim, keepdim);
        let output = op.forward();
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

// *****************************************************************************************************************
// ******************************************** .backward() *********************************************
// *****************************************************************************************************************

// autodifferentiation is algorithmic but uses numerical objects, not symbolic.
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
                            *dfdx_prev = (&*dfdx_prev + dfdx_next).unwrap(); // todo: dfdx_next.detatch()?
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
                vec![
                    (&dx * &grad.clone()).unwrap(),
                    (&dy * &grad.clone()).unwrap(),
                ]
            }
            Op::Sub(_x, _y) => {
                // d/dx(x - y) = 1, d/dy(x - y) = -1
                let (dx, dy) = (
                    Tensor::ones(&grad.shape),
                    Tensor::new(vec![DtypeVal::Float32(-1.0); grad.numel()]),
                );
                vec![
                    (&dx * &grad.clone()).unwrap(),
                    (&dy * &grad.clone()).unwrap(),
                ]
            }
            Op::Mul(x, y) => {
                // d/dx(x * y) = y, d/dy(x * y) = x
                let (dx, dy) = (y, x);
                vec![(dx * &grad.clone()).unwrap(), (dy * &grad.clone()).unwrap()]
            }
            Op::Div(x, y) => todo!(),
            Op::Neg(x) => todo!(),
            Op::Exp(x) => todo!(),
            Op::Log(x) => todo!(),
            Op::Sinh(x) => todo!(),
            Op::Cosh(x) => todo!(),
            Op::Tanh(x) => todo!(),
            // Op::Mean(x) => todo!(),
            // Op::Var(x) => todo!(),
            Op::Matmul(x, y) => todo!(),
            Op::Sum(tensor, _, _) => todo!(),
        }
    }
}

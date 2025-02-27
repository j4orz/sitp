use crate::{ops::Op, Dtype, DtypeVal, Storage, Tensor, TensorError};
use std::{
    cell::RefCell,
    ops::{Add, AddAssign, Div, Mul, Neg, Sub},
    rc::Rc,
};
use thiserror::Error;

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

impl AddAssign for DtypeVal {
    fn add_assign(&mut self, other: Self) {
        *self = self.add(other);
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

pub fn forward_cpu(op: &Op) -> Result<Tensor, OpForwardError> {
    match op {
        Op::Add(x, y) => zip_cpu(|xi, yi| xi + yi, x, y),
        Op::Sub(x, y) => zip_cpu(|xi, yi| xi - yi, x, y),
        Op::Mul(x, y) => zip_cpu(|xi, yi| xi * yi, x, y),
        Op::Div(x, y) => zip_cpu(|xi, yi| xi / yi, x, y),
        // TODO?: can desugar to mul, just like tanh(x) := div(sinh(x), cosh(x))
        // let op = Op::Mul(x.clone(), Tensor::new(vec![-1.0; x.numel()]));
        // let y = forward(&op, &self.device);
        // y

        // thread op through to maintain compute graph (output.inputs)
        Op::Neg(x) => map_cpu(&op, |xi| -xi, x),
        Op::Exp(x) => map_cpu(&op, |xi| DtypeVal::Float32(f32::from(xi).exp()), x),
        Op::Log(x) => map_cpu(&op, |xi| DtypeVal::Float32(f32::from(xi).ln()), x),
        Op::Sinh(x) => map_cpu(&op, |xi| DtypeVal::Float32(f32::from(xi).sinh()), x),
        Op::Cosh(x) => map_cpu(&op, |xi| DtypeVal::Float32(f32::from(xi).cosh()), x),
        Op::Tanh(x) => map_cpu(&op, |xi| DtypeVal::Float32(f32::from(xi).tanh()), x),
        // sigmoid
        // relu
        Op::Sum(x, dim, keepdim) => reduce_cpu(|xi, yi| xi + yi, x, *dim, *keepdim),
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

            let Z = crate::zeros(vec![n, p], Dtype::Float32);

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

            Ok(Z)
        }
    }
}

// picograd does not support broadcastable map implementation with return-oriented "out" arg
fn map_cpu<F>(op: &Op, f: F, x: &Tensor) -> Result<Tensor, OpForwardError>
where
    F: Fn(DtypeVal) -> DtypeVal,
{
    Ok(Tensor {
        ndim: x.ndim,
        shape: x.shape.clone(),
        stride: x.stride.clone(),
        input_op: Some(Box::new(op.clone())), // Box since Op owns Tensors
        requires_grad: x.requires_grad,
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
// 2. for physz in z.numel():

//      **a. find the logical nd index of x and y**
//      3. logz <- encode(i) # lift physical z to logical z
//      4. (logx, logy) <- broadcast_logidx(logx, logz), broadcast_logidx(logy, logz) # map logical z over to logical x and y
//      5. physx, physy <- decode(logx), decode(logy) # lower x and y to physical

//      6. z[physz] <- f(x[physx], y[physy])
fn zip_cpu<F>(f: F, x: &Tensor, y: &Tensor) -> Result<Tensor, OpForwardError>
where
    F: Fn(DtypeVal, DtypeVal) -> DtypeVal,
{
    let (x, y, z) = if x.shape != y.shape {
        let z_shape = Tensor::broadcast_shape(&x.shape, &y.shape)?;
        let z = crate::zeros(z_shape, Dtype::Float32); // clone because of python/rust memory mismatch

        let (x_stride, y_stride) = (
            Tensor::clamp_stride(&x.shape),
            Tensor::clamp_stride(&y.shape),
        );

        let (mut x, mut y) = (x.clone(), y.clone());
        x.stride = x_stride;
        y.stride = y_stride;

        (x, y, z)
    } else {
        let z = crate::zeros(x.shape.clone(), Dtype::Float32); // clone because of python/rust memory mismatch
        (x.clone(), y.clone(), z) // TODO: possibly remove taking view just to satisfy typechecker
    };

    // println!("x.shape: {:?}, x.stride: {:?}", x.shape, x.stride);
    // println!("y.shape: {:?}, y.stride: {:?}", y.shape, y.stride);
    // println!("z.shape: {:?}, z.stride: {:?}", z.shape, z.stride);

    {
        let (x_storage, y_storage, mut z_storage) = (
            x.storage.borrow(),
            y.storage.borrow(),
            z.storage.borrow_mut(),
        );
        let (logx, logy) = (vec![0; x.ndim], vec![0; y.ndim]);

        for phyz in 0..z.numel() {
            let logz = Tensor::encode(phyz, &z.shape);

            let (logx, logy) = (
                Tensor::broadcast_logidx(&logx, &logz)?,
                Tensor::broadcast_logidx(&logy, &logz)?,
            );
            let (phyx, phyy) = (
                Tensor::decode(&logx, &x.stride),
                Tensor::decode(&logy, &y.stride),
            );

            // println!("phy: {:?}", phyz);
            // println!("logx {:?} -> physx: {:?}", logx, phyx);
            // println!("logy {:?} -> physy: {:?}", logy, phyy);
            // println!("logz {:?} -> physz: {:?}", logz, phyz);

            z_storage.data[phyz] = f(x_storage.data[phyx], y_storage.data[phyy]);
        }
    }

    Ok(z)
}

fn reduce_cpu<F>(
    f: F,
    x: &Tensor,
    dim_reduce: usize,
    keepdim: bool,
) -> Result<Tensor, OpForwardError>
where
    F: Fn(DtypeVal, DtypeVal) -> DtypeVal,
{
    let y_shape = x
        .shape
        .iter()
        .enumerate()
        .map(|(i, &dim_size)| if i == dim_reduce { 1 } else { dim_size })
        .collect();
    let mut y = crate::zeros(y_shape, Dtype::Float32); // clone because of python/rust memory mismatch

    {
        let (x_storage, mut y_storage) = (x.storage.borrow(), y.storage.borrow_mut());
        for physy in 0..y.numel() {
            let mut logy = Tensor::encode(physy, &y.shape);

            for d in 0..x.shape[dim_reduce] {
                logy[dim_reduce] = d;
                let physx = Tensor::decode(&logy, &x.stride);

                // accumulate/reduce
                y_storage.data[physy] = f(y_storage.data[physy], x_storage.data[physx]);
            }
        }
    }

    if !keepdim {
        y.shape.remove(dim_reduce);
        y.ndim -= 1;
    }

    Ok(y)
}

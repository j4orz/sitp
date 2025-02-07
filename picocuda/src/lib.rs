use pyo3::prelude::*;
use rand::distr::StandardUniform;
use rand::Rng;
use std::{
    collections::HashSet,
    fmt::{self, Display},
    ops::{Add, Div, Index, IndexMut, Mul, Sub},
    rc::Rc,
};

// pub mod nn;

#[rustfmt::skip]
#[derive(Clone, Debug)]
pub enum Device { Cpu, Cuda, Mps }

#[rustfmt::skip]
#[derive(Clone, Debug)]
pub enum Layout { Strided  } // Sparse, // MklDnn

#[rustfmt::skip]
#[derive(Clone, Debug)]
pub enum Dtype { Bool, Float16, Float32, Float64, Int16, Int32, Int64 }

// - lifetime: deallocation semantics
//   - Box<_>: N/A (exclusive ownership)
//   - Rc<_>: reference counting
//   - arena:
// - mutation: needed for .backward()
//   - RefCell<_>: safe
//   - UnsafeCell<_>: efficient

#[derive(Clone, Debug)]
pub struct Tensor {
    // logical
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    pub grad: Option<Box<Vec<Tensor>>>,
    pub input: Option<Box<Op>>, // todo: weak for cyclic? NN's should be DAG's though

    // physical
    pub device: Device,
    pub layout: Layout,
    pub dtype: Dtype,
    pub data: Vec<f32>, // picograd fixed on fp32 to bootstrap
}

// TODO: impl .item() for pythonic pytorch api?
impl Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.format(f, &self.shape, &self.stride, 0)
    }
}

impl Tensor {
    // *****************************************************************************************************************
    // ********************************************* CONSTRUCTORS (alloc) **********************************************
    // *****************************************************************************************************************

    fn numel(&self) -> usize {
        self.shape.iter().product::<usize>()
    }

    fn stride(shape: &[usize]) -> Vec<usize> {
        let stride = shape
            .iter()
            .rev()
            .fold((vec![], 1), |(mut strides, acc), &dim| {
                strides.push(acc);
                (strides, acc * dim)
            })
            .0
            .into_iter()
            .rev()
            .collect::<Vec<_>>();

        stride
    }

    // todo: requires_grad: bool
    pub fn new(data: Vec<f32>) -> Self {
        Tensor {
            shape: vec![data.len()],
            stride: Self::stride(&vec![data.len()]),
            grad: None,
            input: None,
            device: Device::Cpu,
            layout: Layout::Strided,
            dtype: Dtype::Float32,
            data,
        }
    }

    /// randn returns a tensor filled with random numbers from a normal distribution
    /// with mean 0 and variance 1 (also called the standard normal distribution)
    ///
    /// Examples
    /// ```rust
    /// let x = Tensor::randn(4);
    /// println!("{:?}", x);
    ///
    /// let y = Tensor::randn(&[2, 3]);
    /// println!("{:?}", y);
    /// ```
    pub fn randn(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product::<usize>();
        let data = (0..size)
            .map(|_| rand::rng().sample(StandardUniform))
            .collect::<Vec<f32>>();

        Tensor {
            shape: shape.to_owned(),
            stride: Self::stride(shape),
            grad: None,
            input: None,
            device: Device::Cpu,
            layout: Layout::Strided,
            dtype: Dtype::Float32,
            data,
        }
    }

    /// returns a tensor filled with the scalar value 0, with the shape defined by the variable argument size
    ///
    /// Examples
    /// ```rust
    /// let x = Tensor::zeros(&[2, 3]);
    /// println!("{:?}", x);
    ///
    /// let y = Tensor::zeros(5);
    /// println!("{:?}", y);
    /// ```
    pub fn zeros(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product::<usize>();
        Tensor {
            shape: shape.to_owned(),
            stride: Self::stride(shape),
            grad: None,
            input: None,
            device: Device::Cpu,
            layout: Layout::Strided,
            dtype: Dtype::Float32,
            data: vec![0.0; size],
        }
    }

    /// returns a tensor filled with the scalar value 1, with the shape defined by the variable argument size
    ///
    /// Examples
    /// ```rust
    /// let x = Tensor::ones(&[2, 3]);
    /// println!("{:?}", x);
    ///
    /// let y = Tensor::ones(5);
    /// println!("{:?}", y);
    /// ```
    pub fn ones(shape: &[usize]) -> Self {
        let size = shape.iter().product::<usize>();
        Tensor {
            shape: shape.to_owned(),
            stride: Self::stride(shape),
            grad: None,
            input: None,
            device: Device::Cpu,
            layout: Layout::Strided,
            dtype: Dtype::Float32,
            data: vec![1.0; size],
        }
    }

    // *****************************************************************************************************************
    // ************************************************** VIEWS ***************************************************
    // *****************************************************************************************************************

    /// returns a new tensor with the same data as the self tensor but of a different shape.
    ///
    /// The returned tensor shares the same data and must have the same number of elements, but may have a different size.
    /// For a tensor to be viewed, the new view size must be compatible with its original size and stride
    /// i.e., each new view dimension must either be a subspace of an original dimension, or only span across original dimensions
    ///
    /// Examples
    /// ```rust
    /// let x = Tensor::randn(&[4, 4]);
    /// let y = x.view(&[16]);
    /// let z = y.view(&[-1, 8]);
    /// ```
    pub fn view(&self, shape: &[i32]) -> Self {
        todo!()
    }

    pub fn permute(&self) -> Self {
        todo!()
    }

    pub fn reshape(&self) -> Self {
        todo!()
    }

    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        shape: &[usize],
        stride: &[usize],
        offset: usize,
    ) -> fmt::Result {
        match (shape, stride) {
            ([], []) => {
                write!(fmt, "{:.4}", self.data[offset])?;
                Ok(())
            }
            // basecase: indexed ndarray all the way through
            // ([_dimf], [_stridef]) => {
            //     write!(fmt, "{:.4}", self.data[offset])?;
            //     Ok(())
            // }
            ([D, dimr @ ..], [stridef, strider @ ..]) => {
                write!(fmt, "[")?;
                for d in 0..*D {
                    // rolling out dimf
                    self.format(fmt, dimr, strider, offset + stridef * d)?;
                    if d != *D - 1 {
                        write!(fmt, ", ")?;
                    }
                }
                write!(fmt, "]\n")?;
                Ok(())
            }
            _ => panic!(),
        }
    }
}

// *****************************************************************************************************************
// ******************************************** .forward()/.backward() *********************************************
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
    pub fn backward(mut self) -> () {
        self.grad = Some(Box::new(vec![Tensor::ones(&self.shape)])); // base case dfdx
        let (mut topo, mut visited) = (Vec::new(), HashSet::new());
        self.topo(&mut topo, &mut visited);
        for mut tensor in topo.into_iter().rev() {
            if let Some(ref op) = tensor.input {
                tensor.grad = Some(Box::new(op.backward(&tensor)));
            }
        }
    }

    fn topo(&self, topo: &mut Vec<Tensor>, visited: &mut HashSet<*const Op>) {
        match self.input {
            Some(ref input) => {
                let input_ptr = Rc::as_ptr(input);
                if visited.contains(&input_ptr) {
                    return;
                }
                visited.insert(input_ptr);

                // flatten inputs: tup -> vec
                let inputs = match &**input {
                    Op::Add(x, y)
                    | Op::Sub(x, y)
                    | Op::Mul(x, y)
                    | Op::Div(x, y)
                    | Op::Matmul(x, y) => vec![x, y],
                    Op::Sin(x)
                    | Op::Cos(x)
                    | Op::Exp(x)
                    | Op::Log(x)
                    | Op::Tanh(x)
                    | Op::Mean(x)
                    | Op::Var(x) => vec![x],
                };

                for input in inputs {
                    input.topo(topo, visited);
                }
                topo.push(self.clone());
            }
            None => todo!(),
        }
    }
}

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

    fn apply_binary_op<F>(&self, f: F, x: &Tensor, y: &Tensor) -> Tensor
    where
        F: Fn(f32, f32) -> f32,
    {
        assert_eq!(x.shape, y.shape, "Shape mismatch in operation");

        Tensor {
            shape: x.shape.clone(),
            stride: x.stride.clone(),
            grad: None,
            input: Some(Box::new(self.clone())), // todo: avoid alloc?
            device: x.device.clone(),
            layout: x.layout.clone(),
            dtype: x.dtype.clone(),
            data: x
                .data
                .iter()
                .zip(y.data.iter())
                .map(|(&a_val, &b_val)| f(a_val, b_val))
                .collect(),
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

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn picograd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}

pub mod differentiator;
use differentiator::Op;
use pyo3::prelude::*;
use rand::distr::StandardUniform;
use rand::Rng;
use std::{
    cell::RefCell,
    fmt::{self, Display},
    rc::Rc,
};

// pub mod nn;

// - lifetime: deallocation semantics
//   - Box<_>: N/A (exclusive ownership)
//   - Rc<_>: reference counting
//   - arena:
// - mutation: needed for .backward()
//   - RefCell<_>: safe
//   - UnsafeCell<_>: efficient

#[derive(Debug)]
pub struct Tensor {
    // logical
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    pub input_op: Option<Box<Op>>, // need indirection since Op owns a Tensor

    // physical
    pub storage: Rc<RefCell<Storage>>,
    pub device: Device,
    pub layout: Layout,
    pub dtype: Dtype,
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            stride: self.stride.clone(),
            input_op: self.input_op.clone(),
            storage: self.storage.clone(), // Rc::clone()
            device: self.device.clone(),
            layout: self.layout.clone(),
            dtype: self.dtype.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Storage {
    pub data: Vec<f32>, // picograd fixed on fp32 to bootstrap
    pub grad: Option<Tensor>,
}

// TODO: impl .item() for pythonic pytorch api?
impl Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.format(f, &self.shape, &self.stride, 0)
    }
}

// ********************************************* physical types **********************************************

#[rustfmt::skip]
#[derive(Clone, Debug)]
pub enum Device { Cpu, Cuda, Mps }

#[rustfmt::skip]
#[derive(Clone, Debug)]
pub enum Layout { Strided  } // Sparse, // MklDnn

#[rustfmt::skip]
#[derive(Clone, Debug)]
pub enum Dtype { Bool, Float16, Float32, Float64, Int16, Int32, Int64 }

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
            input_op: None,
            storage: Rc::new(RefCell::new(Storage { data, grad: None })),
            device: Device::Cpu,
            layout: Layout::Strided,
            dtype: Dtype::Float32,
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
            input_op: None,
            storage: Rc::new(RefCell::new(Storage { data, grad: None })),
            device: Device::Cpu,
            layout: Layout::Strided,
            dtype: Dtype::Float32,
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
            input_op: None,
            storage: Rc::new(RefCell::new(Storage {
                data: vec![0.0; size],
                grad: None,
            })),
            device: Device::Cpu,
            layout: Layout::Strided,
            dtype: Dtype::Float32,
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
            input_op: None,
            storage: Rc::new(RefCell::new(Storage {
                data: vec![1.0; size],
                grad: None,
            })),
            device: Device::Cpu,
            layout: Layout::Strided,
            dtype: Dtype::Float32,
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
                write!(fmt, "{:.4}", self.storage.borrow().data[offset])?;
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

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

pub mod functional;
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
    pub ndim: usize,
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
            ndim: self.ndim,
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
        Self {
            ndim: 1,
            shape: vec![data.len()],
            stride: Self::stride(&vec![data.len()]),
            input_op: None,
            storage: Rc::new(RefCell::new(Storage { data, grad: None })),
            device: Device::Cpu,
            layout: Layout::Strided,
            dtype: Dtype::Float32,
        }
    }

    pub fn randn(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product::<usize>();
        let data = (0..size)
            .map(|_| rand::rng().sample(StandardUniform))
            .collect::<Vec<f32>>();

        Self {
            ndim: shape.len(),
            shape: shape.to_owned(),
            stride: Self::stride(shape),
            input_op: None,
            storage: Rc::new(RefCell::new(Storage { data, grad: None })),
            device: Device::Cpu,
            layout: Layout::Strided,
            dtype: Dtype::Float32,
        }
    }

    pub fn arange(start: f32, end: f32, step: f32) -> Self {
        todo!()
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product::<usize>();
        Self {
            ndim: shape.len(),
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

    pub fn ones(shape: &[usize]) -> Self {
        let size = shape.iter().product::<usize>();
        Self {
            ndim: shape.len(),
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

    pub fn view(&self, shape: &[i32]) -> Self {
        todo!()
    }

    pub fn permute(&self, shape: &[usize]) -> Self {
        let new_shape = shape.iter().map(|&old_dim| self.shape[old_dim]).collect();
        let new_stride = shape.iter().map(|&old_dim| self.stride[old_dim]).collect();

        Self {
            ndim: self.ndim,
            shape: new_shape,
            stride: new_stride,
            input_op: None,
            storage: self.storage.clone(),
            device: self.device.clone(),
            layout: self.layout.clone(),
            dtype: self.dtype.clone(),
        }
    }

    pub fn reshape(&self, shape: &[usize]) -> Self {
        let new_size = shape.iter().product::<usize>();

        // assert_eq!(
        //     self.numel(),
        //     new_size,
        //     "new shape must have same number of elements as current shape"
        // );

        Self {
            ndim: shape.len(),
            shape: shape.to_owned(),
            stride: Self::stride(shape),
            input_op: None,
            storage: self.storage.clone(), // Rc::clone()
            device: self.device.clone(),
            layout: self.layout.clone(),
            dtype: self.dtype.clone(),
        }
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

use pyo3::prelude::*;
use rand::distr::StandardUniform;
use rand::Rng;
use std::{
    fmt::{self, Display},
    ops::{Index, IndexMut},
};

// pub mod nn;

// TODO
// - strides
// - broadcasting
// - algebraic ops (+, *)
// - transcendetal ops
// - fuzzer

#[rustfmt::skip]
#[derive(Clone, Debug)]
enum Device { Cpu, Cuda, Mps }

#[rustfmt::skip]
#[derive(Clone, Debug)]
enum Layout { Strided  } // Sparse, // MklDnn

#[rustfmt::skip]
#[derive(Clone, Debug)]
enum Dtype { Bool, Float16, Float32, Float64, Int16, Int32, Int64 }

#[derive(Clone, Debug)]
pub struct Tensor {
    // logical
    shape: Vec<usize>,
    stride: bool,
    // offset: bool,
    // grad: Box<Tensor>,

    // physical
    device: Device,
    layout: Layout,
    dtype: Dtype,
    data: Vec<f32>, // picograd fixed on fp32 to bootstrap
}

impl Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.data)
    }
}

fn tensor(input: Vec<usize>) -> Tensor {
    Tensor::new()
}

impl Tensor {
    pub fn new() -> Self {
        todo!()
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
        let size: usize = shape.iter().sum::<usize>();
        let data = (0..size)
            .map(|_| rand::rng().sample(StandardUniform))
            .collect::<Vec<f32>>();

        Tensor {
            shape: shape.to_owned(),
            stride: false,
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
        let size: usize = shape.iter().sum::<usize>();
        Tensor {
            shape: shape.to_owned(),
            stride: todo!(),
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
        let size = shape.iter().sum::<usize>();
        Tensor {
            shape: shape.to_owned(),
            stride: todo!(),
            device: Device::Cpu,
            layout: Layout::Strided,
            dtype: Dtype::Float32,
            data: vec![1.0; size],
        }
    }

    /// returns a new tensor with the same data as the self tensor but of a different shape.
    ///
    /// The returned tensor shares the same data and must have the same number of elements, but may have a different size.
    /// For a tensor to be viewed, the new view size must be compatible with its original size and stride
    /// i.e., each new view dimension must either be a subspace of an original dimension, or only span across original dimensions
    ///
    /// Examples
    /// ```rust
    /// let x = Tensor::randn(&[2, 3]);
    /// assert_eq!(x.shape, vec![2, 3]);
    ///
    /// let y = x.view(16);
    /// assert_eq!(y.shape, vec![16]);
    ///
    /// let z = y.view(&[-1, 8]);
    /// assert_eq!(z.shape, vec![2, 8]);
    /// ```
    pub fn view(&self) -> Self {
        todo!()
    }

    pub fn permute(&self) -> Self {
        todo!()
    }

    pub fn reshape(&self) -> Self {
        todo!()
    }
}

impl Index<(usize, usize)> for Tensor {
    type Output = f32; // fixed to fp32 for now

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (i, j) = index;
        let idx = i * self.shape[1] + j; // Row-major ordering
        &self.data[idx]
    }
}

impl IndexMut<(usize, usize)> for Tensor {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (i, j) = index;
        let idx = i * self.shape[1] + j; // Row-major ordering
        &mut self.data[idx]
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

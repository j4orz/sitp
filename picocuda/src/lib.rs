use std::ops::{Index, IndexMut};

use pyo3::prelude::*;
// pub mod nn;

// TODO
// - strides
// - broadcasting
// - algebraic ops (+, *)
// - transcendetal ops
// - fuzzer

#[derive(Clone, Debug)]
struct Tensor {
    // logical
    shape: Vec<usize>,
    stride: Vec<usize>,
    // offset: bool,
    // grad: Box<Tensor>,

    // physical
    device: Device,
    layout: Layout,
    dtype: Dtype,
    data: Vec<f32>, // picograd fixed on fp32 to bootstrap
}

#[rustfmt::skip]
#[derive(Clone, Debug)]
enum Device { Cpu, Cuda, Mps }

#[rustfmt::skip]
#[derive(Clone, Debug)]
enum Layout { Strided  } // Sparse, // MklDnn

#[rustfmt::skip]
#[derive(Clone, Debug)]
enum Dtype { Bool, Float16, Float32, Float64, Int16, Int32, Int64 }

fn tensor(input: Vec<usize>) -> Tensor {
    Tensor::new()
}

impl Tensor {
    fn new() -> Self {
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
    fn randn(shape: &[usize]) -> Self {
        todo!()
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
    fn zeros(shape: &[usize]) -> Self {
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
    fn ones(shape: &[usize]) -> Self {
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

    fn view(&self) -> Self {
        todo!()
    }

    fn permute(&self) -> Self {
        todo!()
    }

    fn reshape(&self) -> Self {
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

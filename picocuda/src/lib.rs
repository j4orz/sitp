use pyo3::prelude::*;
// pub mod nn;

// TODO
// - strides
// - broadcasting
// - algebraic ops (+, *)
// - transcendetal ops

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
    data: Vec<f32>, // picograd fixed on fp32 to bootstrap
    dtype: Dtype,
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

    /// randn eturns a tensor filled with random numbers from a normal distribution
    /// with mean 0 and variance 1 (also called the standard normal distribution).
    ///
    /// Example
    /// ```rust
    /// let x = Tensor::randn(4)
    /// println!("{:?}", x)
    ///
    /// let W = Tensor::randn(2, 3)
    /// println!("{:?}", W)
    /// ```
    fn randn(size: (i32, i32)) -> Self {
        todo!()
    }

    fn zeros() -> Self {
        todo!()
    }

    fn ones() -> Self {
        todo!()
    }

    fn reshape(&self) -> Self {
        todo!()
    }

    fn view(&self) -> Self {
        todo!()
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

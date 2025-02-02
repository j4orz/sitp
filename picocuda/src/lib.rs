use pyo3::prelude::*;
// pub mod nn;

enum Layout {
    Strided,
}

enum Device {
    Cpu,
    Cuda,
}

struct Tensor {
    // logical
    shape: Vec<usize>,
    stride: Vec<usize>,
    offset: bool,
    grad: Box<Tensor>,

    // physical
    layout: Layout,
    device: Device,
    data: bool,
    dtype: bool,
}

fn tensor(input: Vec<usize>) -> Tensor {
    Tensor::new()
}

impl Tensor {
    fn new() -> Self {
        todo!()
    }

    fn randn() -> Self {
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

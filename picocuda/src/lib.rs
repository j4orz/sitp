pub mod differentiator;
pub mod functional;
// pub mod nn;
// pub mod optim;

use differentiator::Op;
use pyo3::prelude::*;
use rand::distr::StandardUniform;
use rand::Rng;
use std::{
    cell::RefCell,
    fmt::{self, Display},
    io,
    rc::Rc,
};
use thiserror::Error;

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
    pub storage: Rc<RefCell<Storage<DtypeVal>>>,
    pub device: Device,
    pub layout: Layout,
    pub dtype: Dtype, // TODO? not typed with storage
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
pub struct Storage<DtypeVal> {
    pub data: Vec<DtypeVal>, // picograd fixed on fp32 to bootstrap
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
pub enum Dtype { Bool, Float16, Float32, Float64, Int16, Int32, Int64}

#[rustfmt::skip]
#[derive(Clone, Copy, Debug)]
pub enum DtypeVal { Bool(bool), Float32(f32), Float64(f64), Int16(i16), Int32(i32), Int64(i64) } // f16 is unstable

// TODO: remove usize for pythonic bindings?

impl From<DtypeVal> for f32 {
    fn from(value: DtypeVal) -> Self {
        match value {
            DtypeVal::Float32(x) => x,
            DtypeVal::Int32(x) => x as f32,
            DtypeVal::Int64(x) => x as f32,
            _ => todo!(),
        }
    }
}

impl Tensor {
    // *****************************************************************************************************************
    // ********************************************* CONSTRUCTORS (alloc) **********************************************
    // *****************************************************************************************************************

    fn alloc(shape: &[usize], data: Vec<DtypeVal>) -> Self {
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

    // todo: requires_grad: bool
    pub fn new(data: Vec<DtypeVal>) -> Self {
        Self::alloc(&vec![data.len()], data)
    }

    pub fn zeros(shape: &[usize], dtype: Dtype) -> Self {
        let n = shape.iter().product();
        match dtype {
            Dtype::Float32 => Self::alloc(shape, vec![DtypeVal::Float32(0.0); n]),
            _ => todo!(),
        }
    }

    pub fn ones(shape: &[usize]) -> Self {
        let n = shape.iter().product();
        let data = vec![DtypeVal::Float32(1.0); n];
        Self::alloc(shape, data)
    }

    pub fn randn(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product::<usize>();
        let data = (0..n)
            .map(|_| DtypeVal::Float32(rand::rng().sample(StandardUniform)))
            .collect::<Vec<_>>();

        Self::alloc(shape, data)
    }

    pub fn arange(start: f32, end: f32, step: f32) -> Self {
        todo!()
    }

    // *****************************************************************************************************************
    // *********************************************** VIEWS (no alloc) ************************************************
    // *****************************************************************************************************************

    // TODO: permute, reshape, should be Op??

    fn no_alloc(&self, shape: &[usize]) -> Self {
        Self {
            ndim: shape.len(),
            shape: shape.to_vec(),
            stride: Self::stride(shape),
            input_op: self.input_op.clone(), // Box<_>.clone()?
            storage: self.storage.clone(),
            device: self.device.clone(),
            layout: self.layout.clone(),
            dtype: self.dtype.clone(),
        }
    }

    pub fn permute(&self, shape: &[usize]) -> Self {
        let new_shape = shape
            .iter()
            .map(|&old_dim| self.shape[old_dim])
            .collect::<Vec<_>>();

        self.no_alloc(&new_shape)
    }

    pub fn view(&self, shape: &[i32]) -> Self {
        todo!()
    }

    pub fn reshape(&self, shape: &[usize]) -> Result<Self, io::Error> {
        let new_size = shape.iter().product::<usize>();

        if self.numel() != new_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "new shape must have same number of elements as current shape",
            ));
        }

        Ok(self.no_alloc(shape))
    }

    pub fn detach(&self) -> Self {
        // self.no_alloc(&self.shape)
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
                write!(fmt, "{:?}", self.storage.borrow().data[offset])?;
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

    // *****************************************************************************************************************
    // ********************************************* INDEXING/BROADCASTING *********************************************
    // *****************************************************************************************************************
    fn numel(&self) -> usize {
        self.shape.iter().product::<usize>()
    }

    // note: best way to think about strides
    // is to imagine the indices of the nested for loop when unrolling
    // e.g: shape: [3, 4, 6] ==> stride: [24, 6, 1]
    //      - one step for outer is 24 physical indices
    //      - one step for middle is 6 physical indices
    //      - one step for inner is 1 physical index (always)
    fn stride(shape: &[usize]) -> Vec<usize> {
        let stride = shape
            .iter()
            .rev()
            .fold((vec![], 1), |(mut strides, acc), &dim| {
                strides.push(acc);
                (strides, acc * dim) // last acc does not push
            })
            .0
            .into_iter()
            .rev()
            .collect::<Vec<_>>();

        stride
    }

    // note: Vec<usize> is used for internal indexing
    // Vec<DtypeVal=Int32> is used by library crate users

    // encode: phys(usize) -> log(Vec<usize>)
    fn encode(phys: usize, shape: &[usize]) -> Vec<usize> {
        let mut log = vec![0; shape.len()];
        let (stride, mut phys) = (Self::stride(shape), phys);

        // unroll (factorize) the quotient from largest to smallest
        for i in 0..stride.len() {
            log[i] = phys / stride[i]; // how many stride[i] fit into phys
            phys %= stride[i]; // remainder
        }

        log
    }

    // decode: log(Vec<usize>) -> phys(usize)
    fn decode(log: &[usize], stride: &[usize]) -> usize {
        log.iter()
            .zip(stride.iter())
            .fold(0, |acc, (i, s)| acc + i * s)
    }

    fn broadcast_shape(shape_x: &[usize], shape_y: &[usize]) -> Result<Vec<usize>, TensorError> {
        todo!()
    }

    fn broadcast_logidx(log_x: &[usize], log_y: &[usize]) -> Result<Vec<usize>, TensorError> {
        todo!()
    }
}

#[derive(Error, Debug)]
pub enum TensorError {
    #[error("broadcast mismatch")]
    BroadcastMismatch,
    #[error("unknown tensor error")]
    Unknown,
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

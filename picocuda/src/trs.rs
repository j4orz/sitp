use crate::{Device, Dtype, DtypeVal, Layout, ops::Op};
use pyo3::prelude::*;
use std::{
    cell::RefCell,
    cmp::{Ordering, max},
    fmt::{self, Display},
    io, iter,
    ops::Index,
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

#[pyclass(unsendable)] // for now. does pytorch user code multithread tensors?
#[derive(Debug)]
pub struct Tensor {
    // logical
    pub ndim: usize,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    pub input_op: Option<Box<Op>>, // need indirection since Op owns a Tensor
    pub requires_grad: bool,

    // physical
    pub storage: Rc<RefCell<Storage<DtypeVal>>>, // pub storage: Arc<RwLock<Storage<DtypeVal>>>,
    pub device: Device,
    pub layout: Layout,
    pub dtype: Dtype, // not typed with storage
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Self {
            ndim: self.ndim,
            shape: self.shape.clone(),
            stride: self.stride.clone(),
            input_op: self.input_op.clone(),
            requires_grad: self.requires_grad,
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

// *****************************************************************************************************************
// ********************************************* CONSTRUCTORS (alloc) **********************************************
// *****************************************************************************************************************

pub fn alloc(shape: &[usize], data: Vec<DtypeVal>) -> Tensor {
    Tensor {
        ndim: shape.len(),
        shape: shape.to_owned(),
        stride: Tensor::shape_to_stride(shape),
        input_op: None,
        requires_grad: false,
        storage: Rc::new(RefCell::new(Storage { data, grad: None })),
        device: Device::Cpu,
        layout: Layout::Strided,
        dtype: Dtype::Float32,
    }
}

impl Tensor {
    pub fn to(&self, d: &Device) -> Self {
        let foo = match d {
            Device::Cpu => todo!(),
            Device::Gpu => todo!(),
            Device::Cuda => todo!(),
        };

        todo!()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product::<usize>()
    }

    // *****************************************************************************************************************
    // ***************************************** VIEWS (no alloc/datamovement) *****************************************
    // *****************************************************************************************************************

    pub fn continuous_view(&self, shape: &[usize]) -> Self {
        Self {
            ndim: shape.len(),
            shape: shape.to_vec(),
            stride: Self::shape_to_stride(shape),
            input_op: self.input_op.clone(), // Box<_>.clone()?
            requires_grad: self.requires_grad,
            storage: self.storage.clone(),
            device: self.device.clone(),
            layout: self.layout.clone(),
            dtype: self.dtype.clone(),
        }
    }

    pub fn noncontinuous_view(&self, shape: &[usize], stride: &[usize]) -> Self {
        Self {
            ndim: shape.len(),
            shape: shape.to_vec(),
            stride: stride.to_vec(),
            input_op: self.input_op.clone(), // Box<_>.clone()?
            requires_grad: self.requires_grad,
            storage: self.storage.clone(),
            device: self.device.clone(),
            layout: self.layout.clone(),
            dtype: self.dtype.clone(),
        }
    }

    pub fn view(&self) -> Self {
        todo!()
    }

    pub fn _reshape(&self, new_shape: &[i32]) -> Result<Self, ViewOpError> {
        let old_size = self.shape.iter().product::<usize>();
        let negone_count = new_shape.iter().filter(|&&x| x == -1).count();

        let new_shape = match negone_count.cmp(&1) {
            Ordering::Greater => Err(ViewOpError::InvalidReshapeInput),
            Ordering::Less => Ok(new_shape
                .iter()
                .map(|&x| usize::try_from(x).map_err(|_| ViewOpError::InvalidReshapeInput))
                .collect::<Result<Vec<_>, _>>()?),
            Ordering::Equal => {
                let rest_shape = new_shape
                    .iter()
                    .filter(|&&x| x != -1)
                    .map(|&x| usize::try_from(x).map_err(|_| ViewOpError::InvalidReshapeInput))
                    .collect::<Result<Vec<_>, _>>()?;
                let rest_shape_size = rest_shape.iter().product::<usize>();

                if old_size % rest_shape_size != 0 {
                    Err(ViewOpError::ShapeMismatch)
                } else {
                    let inferred_dim = old_size / rest_shape_size;
                    Ok(new_shape
                        .iter()
                        .map(|&x| {
                            if x == -1 {
                                Ok(inferred_dim)
                            } else {
                                usize::try_from(x).map_err(|_| ViewOpError::InvalidReshapeInput)
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()?)
                }
            }
        }?;

        Ok(self.continuous_view(&new_shape))
    }

    pub fn _permute(&self, indices: &[usize]) -> Self {
        let new_shape = indices.iter().map(|&i| self.shape[i]).collect::<Vec<_>>();
        let new_stride = indices
            .iter()
            .map(|&i| self.stride[self.ndim - 1 - i])
            .collect::<Vec<_>>();
        println!("moose {:?}, {:?}", self.shape, self.stride);
        println!("deer {:?} {:?}", new_shape, new_stride);
        self.noncontinuous_view(&new_shape, &new_stride)
    }

    pub fn transpose(&self) -> Self {
        todo!()
    }

    pub fn gather(&self) -> Self {
        todo!()
    }

    pub fn scatter(&self) -> Self {
        todo!()
    }

    pub fn cat(&self) -> Self {
        todo!()
    }

    pub fn stack(&self) -> Self {
        todo!()
    }

    pub fn squeeze(&self) -> Self {
        todo!()
    }

    pub fn unsqueeze(&self) -> Self {
        todo!()
    }

    pub fn flatten(&self) -> Self {
        todo!()
    }

    pub fn unflatten(&self) -> Self {
        todo!()
    }

    pub fn contiguous(&self) -> Self {
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
    // ********************************************* INDEXING *********************************************
    // *****************************************************************************************************************

    // note: best way to think about strides
    // is to imagine the indices of the nested for loop when unrolling
    // e.g: shape: [3, 4, 6] ==> stride: [24, 6, 1]
    //      - one step for outer is 24 physical indices
    //      - one step for middle is 6 physical indices
    //      - one step for inner is 1 physical index (always)

    // 1 is last because of C/C++/Rust row-major ordering over Fortran/IDL column-major ordering
    fn shape_to_stride(shape: &[usize]) -> Vec<usize> {
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

    // encode (lifting): phys(usize) -> log(Vec<usize>)
    pub fn encode(phys: usize, shape: &[usize]) -> Vec<usize> {
        let mut log = vec![0; shape.len()];
        let (stride, mut phys) = (Self::shape_to_stride(shape), phys);

        // expand the product into factorization
        // traversing from largest step to smallest step
        for i in 0..stride.len() {
            log[i] = phys / stride[i]; // how many stride[i] fit into phys
            phys %= stride[i]; // remainder
        }

        log
    }

    // decode (lowering): log(Vec<usize>) -> phys(usize)
    pub fn decode(log: &[usize], stride: &[usize]) -> usize {
        // collapse the factorization into a product
        log.iter()
            .zip(stride.iter())
            // .inspect(|(i, s)| println!("i: {:?}, s: {:?}", i, s))
            .fold(0, |acc, (i, s)| acc + i * s)
    }

    // *****************************************************************************************************************
    // ********************************************* BROADCASTING *********************************************
    // *****************************************************************************************************************
    pub fn broadcast_shape(
        shape_x: &[usize],
        shape_y: &[usize],
    ) -> Result<Vec<usize>, ViewOpError> {
        let max_len = max(shape_x.len(), shape_y.len());
        let shape_x = iter::repeat(1)
            .take(max_len - shape_x.len())
            .chain(shape_x.iter().copied())
            .collect::<Vec<_>>();
        let shape_y = iter::repeat(1)
            .take(max_len - shape_y.len())
            .chain(shape_y.iter().copied())
            .collect::<Vec<_>>();

        let output = shape_x
            .iter()
            .zip(shape_y.iter())
            .map(|(&x, &y)| match (x, y) {
                (x, y) if x == y => Ok(x),
                (1, y) => Ok(y),
                (x, 1) => Ok(x),
                _ => Err(ViewOpError::BroadcastMismatch),
            })
            .collect::<Result<Vec<usize>, ViewOpError>>();

        output
    }

    pub fn clamp_stride(shape_x: &[usize]) -> Vec<usize> {
        // TODO: assumes input_shape successfully broadcasted with other input
        // TODO: ignore padded scenario when input_shape.len() < output_shape.len()
        let output = shape_x
            .iter()
            .rev()
            .fold((vec![], 1), |(mut stride, mut acc), &dim| {
                if dim == 1 {
                    stride.push(0);
                } else {
                    stride.push(acc);
                    acc *= dim;
                }
                (stride, acc)
            })
            .0
            .into_iter()
            .rev()
            .collect::<Vec<_>>();

        output
    }

    pub fn broadcast_logidx(log_x: &[usize], log_y: &[usize]) -> Result<Vec<usize>, ViewOpError> {
        let max_len = max(log_x.len(), log_y.len());
        let padded_x = iter::repeat(0)
            .take(max_len - log_x.len())
            .chain(log_x.iter().copied());
        let padded_y = iter::repeat(0)
            .take(max_len - log_y.len())
            .chain(log_y.iter().copied());

        let output = padded_x
            .zip(padded_y)
            .map(|(x, y)| match (x, y) {
                (x, y) if x == y => Ok(x),
                (0, y) => Ok(y),
                (x, 0) => Ok(x),
                _ => Err(ViewOpError::BroadcastMismatch),
            })
            .collect::<Result<Vec<usize>, ViewOpError>>();

        output
    }
}

#[derive(Error, Debug)]
pub enum ViewOpError {
    #[error("broadcast mismatch")]
    BroadcastMismatch,
    #[error("shape mismatch")]
    ShapeMismatch,
    #[error("invalid reshape input")]
    InvalidReshapeInput,
    #[error("unknown tensor error")]
    Unknown,
}

impl Index<&[usize]> for Tensor {
    type Output = DtypeVal;

    fn index(&self, i: &[usize]) -> &Self::Output {
        todo!()
    }
}

// impl IndexMut<(usize, usize)> for Tensor {
//     fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
//         let (i, j) = index;
//         let idx = i * self.shape[1] + j; // Row-major ordering
//         &mut self.data[idx]
//     }
// }

// TODO: partial indexing return Tensor with one less dim
// fn __getitem__(s: &Tensor, I: Tensor) -> Tensor {
//     let output_shape = I
//         .shape
//         .iter()
//         .chain(s.shape.iter().skip(1)) // collapse the first dim of self via indexing
//         .copied()
//         .collect::<Vec<_>>();
//     let output = zeros(output_shape, Dtype::Float32);

//     {
//         let I_storage = I.storage.borrow();
//         let input_storage = s.storage.borrow();
//         let mut output_storage = output.storage.borrow_mut();

//         for phy_I in 0..I_storage.data.len() {
//             let i = usize::from(I_storage.data[phy_I]);
//             let (l, r) = (s.stride[0] * i, (s.stride[0] * i) + s.stride[0]);
//             let plucked_tensor = &input_storage.data[l..r];
//             // place plucked_tensor (nested ndarray) in output_storage

//             let log_I = Tensor::encode(phy_I, &I.shape); // where we slot the plucked input in the output tensor
//             let log_output = log_I
//                 .iter()
//                 .chain(iter::repeat(&0).take(s.shape.len() - 1)) // input.shape.len()
//                 .copied()
//                 .collect::<Vec<_>>();

//             let phys_output = Tensor::decode(&log_output, &output.shape);
//             output_storage.data[phys_output..phys_output + plucked_tensor.len()]
//                 .copy_from_slice(plucked_tensor);
//         }
//     }
//     output
// }

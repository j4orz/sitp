// *****************************************************************************************************************
// ******************************************** .backward() *********************************************
// *****************************************************************************************************************

use std::collections::HashSet;

use crate::{ops::Op, DtypeVal, Tensor};

// autodifferentiation is algorithmic but uses numerical objects, not symbolic.
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
    pub fn backward(&mut self) -> () {
        self.storage.borrow_mut().grad = Some(crate::ones(self.shape.clone())); // base case dfdx. // clone because of python/rust memory mismatch
        let (mut topo, mut visited) = (Vec::new(), HashSet::new());
        self.topo(&mut topo, &mut visited);

        for tensor in topo.into_iter() {
            if let Some(ref input_op) = tensor.input_op {
                let (inputs, input_grads) = (
                    input_op.inputs(),
                    input_op.backward(&tensor.storage.borrow().grad.as_ref().unwrap()),
                );

                for (x, dfdx_next) in inputs.into_iter().zip(input_grads.iter()) {
                    let mut storage = x.storage.borrow_mut();
                    match storage.grad {
                        Some(ref mut dfdx_prev) => {
                            *dfdx_prev = (&*dfdx_prev + dfdx_next).unwrap(); // todo: dfdx_next.detatch()?
                        }
                        None => {
                            storage.grad = Some(dfdx_next.clone());
                        }
                    }
                }
            }
        }
    }

    fn topo(&self, topo: &mut Vec<Tensor>, visited: &mut HashSet<Op>) {
        match self.input_op {
            Some(ref op) => {
                if visited.contains(&op) {
                    return;
                }
                visited.insert(*op.clone());
                for input in op.inputs() {
                    input.topo(topo, visited);
                }
            }
            None => {}
        }

        topo.push(self.clone());
    }
}

impl Op {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        match self {
            Op::Add(_x, _y) => {
                // d/dx(x + y) = 1, d/dy(x + y) = 1
                let (dx, dy) = (
                    crate::ones(grad.shape.clone()), // clone because of python/rust memory mismatch
                    crate::ones(grad.shape.clone()), // clone because of python/rust memory mismatch
                );
                vec![
                    (&dx * &grad.clone()).unwrap(),
                    (&dy * &grad.clone()).unwrap(),
                ]
            }
            Op::Sub(_x, _y) => {
                // d/dx(x - y) = 1, d/dy(x - y) = -1
                let (dx, dy) = (
                    crate::ones(grad.shape.clone()), // clone because of python/rust memory mismatch
                    crate::new(vec![DtypeVal::Float32(-1.0); grad.numel()]),
                );
                vec![
                    (&dx * &grad.clone()).unwrap(),
                    (&dy * &grad.clone()).unwrap(),
                ]
            }
            Op::Mul(x, y) => {
                // d/dx(x * y) = y, d/dy(x * y) = x
                let (dx, dy) = (y, x);
                vec![(dx * &grad.clone()).unwrap(), (dy * &grad.clone()).unwrap()]
            }
            Op::Div(x, y) => todo!(),
            Op::Neg(x) => todo!(),
            Op::Exp(x) => todo!(),
            Op::Log(x) => todo!(),
            Op::Sinh(x) => todo!(),
            Op::Cosh(x) => todo!(),
            Op::Tanh(x) => todo!(),
            // Op::Mean(x) => todo!(),
            // Op::Var(x) => todo!(),
            Op::Matmul(x, y) => todo!(),
            Op::Sum(tensor, _, _) => todo!(),
        }
    }
}

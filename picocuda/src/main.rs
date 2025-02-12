use picograd::Tensor;

// singularity systems course notes
// ================================

// - assignment 0: https://colab.research.google.com/github/dlsyscourse/hw0/blob/main/hw0.ipynb#scrollTo=KmqZjTlPeI91
// - https://minitorch.github.io/mlprimer/#neural-networks
// - https://minitorch.github.io/module0/module0/

// section 1: FOUNDATIONS
// * we build the wide hip and build up the lego block abstractions for ai researchers
// * before we drill down into the metal

// part 1: core
// tensor: logical(shape,stride,input_op), storage(data, grad) physical(device,layout,dtype)
//   tensor methods: indexing/broadcasting/views
// ops: forward(map,zip,reduce), backward(chain rule on tensor graph)
//                 - assignment 1: https://colab.research.google.com/github/dlsyscourse/hw1/blob/main/hw1.ipynb#scrollTo=bNSqZVv9cE_b
//                 - https://pytorch.org/docs/stable/notes/autograd.html
//                 - mintorch: https://minitorch.github.io/module1/module1/
// taste of speed: SIMD: https://colab.research.google.com/github/dlsyscourse/hw3/blob/main/hw3.ipynb#scrollTo=35e1e9c0
// taste of abstraction (few lines): cross_entropy_loss()/softmax()

// part 2: nn
// assignment 2: https://colab.research.google.com/github/dlsyscourse/hw2/blob/main/hw2.ipynb#scrollTo=Fx4GG0VrcFMQ
// https://minitorch.github.io/module4/module4/
//
//
//
//
//
// GLUE
// - use detach view op in autograd
// - graphviz/ascii print
// - fuzzer
//      - https://pytorch.org/docs/stable/notes/numerical_accuracy.html
// static? jax : https://docs.jax.dev/en/latest/autodidax.html
// - pyo3 bindings

// inference (rng)
// - picograd.randint()
// - picograd.multinomial()
// - picograd.Generator().manual_seed()
// - https://pytorch.org/docs/stable/notes/randomness.html

// SECTION 1 PART 2: NN

fn main() {
    println!(
        "
    ⠀⠀⠀⠀⠀⣼⣧⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⣼⣿⣿⣧⠀⠀⠀⠀
    ⠀⠀⠀⠾⠿⠿⠿⠿⠷⠀⠀⠀
    ⠀⠀⣼⣆⠀⠀⠀⠀⣰⣧⠀⠀
    ⠀⣼⣿⣿⣆⠀⠀⣰⣿⣿⣧⠀
    ⠾⠟⠿⠿⠿⠧⠼⠿⠿⠿⠻⠷
    picograd: torch.tensor->{{PTX, Triton}} compiler
    ⚠︎ this is a one man effort to compile gpt2, llama3, and r1. beware of dragons ⚠︎
    "
    );

    let x = Tensor::new(vec![9.0]);
    let y = Tensor::new(vec![10.0]);
    let mut z = (&x + &y).unwrap();
    z.backward();

    println!("z: {}", z);
    println!("z.grad: {}", z.storage.borrow().grad.as_ref().unwrap());
    println!("x.grad: {}", x.storage.borrow().grad.as_ref().unwrap());
    println!("y.grad: {}", y.storage.borrow().grad.as_ref().unwrap());

    let x = Tensor::new(vec![9.0]);
    let y = Tensor::new(vec![10.0]);
    let mut z = (&x * &y).unwrap();
    z.backward();

    println!("z: {}", z);
    println!("z.grad: {}", z.storage.borrow().grad.as_ref().unwrap());
    println!("x.grad: {}", x.storage.borrow().grad.as_ref().unwrap());
    println!("y.grad: {}", y.storage.borrow().grad.as_ref().unwrap());

    let X = Tensor::randn(&[2, 3]);
    let Y = Tensor::randn(&[3, 4]);
    let Z = X.matmul(&Y).unwrap();
    // z.backward();

    println!("z: {}", X);
    println!("z: {}", Y);
    println!("z: {}", Z);
    // println!("z.grad: {}", z.storage.borrow().grad.as_ref().unwrap());
    // println!("x.grad: {}", x.storage.borrow().grad.as_ref().unwrap());
    // println!("y.grad: {}", y.storage.borrow().grad.as_ref().unwrap());
}

use picograd::Tensor;

// SECTION 1 PART 1: CORE
// TENSOR/OPS: FORWARD/BACKWARD (assignment 3 dlsyscourse)
// - .forward() 1. tiled matmul? 2. crossentropy/softmax -> reduce: .sum() -> view: reshape/permute (todo: Results) START HERE
// - .backward() for everything
// - SIMD > CUDA: assignment 3: https://colab.research.google.com/github/dlsyscourse/hw3/blob/main/hw3.ipynb#scrollTo=35e1e9c0

// engine:
// - assignment 1: https://colab.research.google.com/github/dlsyscourse/hw1/blob/main/hw1.ipynb#scrollTo=bNSqZVv9cE_b
// - https://pytorch.org/docs/stable/notes/autograd.html

// - assignment 0: https://colab.research.google.com/github/dlsyscourse/hw0/blob/main/hw0.ipynb#scrollTo=KmqZjTlPeI91

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
// assignment 2: https://colab.research.google.com/github/dlsyscourse/hw2/blob/main/hw2.ipynb#scrollTo=Fx4GG0VrcFMQ

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
    let mut z = &x + &y;
    z.backward();

    println!("z: {}", z);
    println!("z.grad: {}", z.storage.borrow().grad.as_ref().unwrap());
    println!("x.grad: {}", x.storage.borrow().grad.as_ref().unwrap());
    println!("y.grad: {}", y.storage.borrow().grad.as_ref().unwrap());

    let x = Tensor::new(vec![9.0]);
    let y = Tensor::new(vec![10.0]);
    let mut z = &x * &y;
    z.backward();

    println!("z: {}", z);
    println!("z.grad: {}", z.storage.borrow().grad.as_ref().unwrap());
    println!("x.grad: {}", x.storage.borrow().grad.as_ref().unwrap());
    println!("y.grad: {}", y.storage.borrow().grad.as_ref().unwrap());

    let X = Tensor::randn(&[2, 3]);
    let Y = Tensor::randn(&[3, 4]);
    let Z = X.matmul(&Y);
    // z.backward();

    println!("z: {}", X);
    println!("z: {}", Y);
    println!("z: {}", Z);
    // println!("z.grad: {}", z.storage.borrow().grad.as_ref().unwrap());
    // println!("x.grad: {}", x.storage.borrow().grad.as_ref().unwrap());
    // println!("y.grad: {}", y.storage.borrow().grad.as_ref().unwrap());
}

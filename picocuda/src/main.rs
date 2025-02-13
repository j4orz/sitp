use picograd::Tensor;

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

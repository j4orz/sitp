use picograd::{DtypeVal, Tensor};

// GLUE
// - use detach view op in autograd
// - graphviz/ascii print
// - fuzzer
//      - https://pytorch.org/docs/stable/notes/numerical_accuracy.html
// static? jax : https://docs.jax.dev/en/latest/autodidax.html

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

    let X = picograd::randn(vec![2, 3]);
    let Y = picograd::randn(vec![3, 4]);
    let mut Z = (&X + &Y).unwrap();
    Z.backward();

    println!("X: {}", X);
    println!("Y: {}", Y);
    println!("Z: {}", Z);

    println!("Z.grad: {}", Z.storage.borrow().grad.as_ref().unwrap());
    println!("X.grad: {}", X.storage.borrow().grad.as_ref().unwrap());
    println!("Y.grad: {}", Y.storage.borrow().grad.as_ref().unwrap());
}

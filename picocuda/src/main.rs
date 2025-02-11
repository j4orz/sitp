use picograd::Tensor;

// TODO SECTION 1 PART 1: CORE
// TENSOR
// ---------
// - broadcast, views, reshapes
// - detatch??
// - graphviz/ascii print

// SYSTEMS
// ---------
// fuzzer. SIMD
// static? jax : https://docs.jax.dev/en/latest/autodidax.html

//
//
//
//
// rng (inference)
// - picograd.randint()
// - picograd.multinomial()
// - picograd.Generator().manual_seed()

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

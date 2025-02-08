use picograd::Tensor;

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

    let x = Tensor::randn(&[2, 3]);
    let y = Tensor::randn(&[3, 4]);
    let mut z = x.matmul(&y);
    z.backward();

    println!("z: {}", z);
    println!("z.grad: {}", z.storage.borrow().grad.as_ref().unwrap());
    println!("x.grad: {}", x.storage.borrow().grad.as_ref().unwrap());
    println!("y.grad: {}", y.storage.borrow().grad.as_ref().unwrap());
}

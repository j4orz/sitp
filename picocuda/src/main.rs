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

    let X = picograd::ones(vec![3, 1]);
    let Y = picograd::ones(vec![1, 3]);
    let Z = (&X * &Y).unwrap();
    // Z.backward();

    // println!("X: {}", X);4
    // println!("Z: {}", Z);
    // println!("Z.grad: {}", Z.storage.borrow().grad.as_ref().unwrap());
    // println!("X.grad: {}", X.storage.borrow().grad.as_ref().unwrap());
    // println!("Y.grad: {}", Y.storage.borrow().grad.as_ref().unwrap());
}

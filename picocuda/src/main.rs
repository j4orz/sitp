mod lib;
use std::rc::Rc;

// TODO?
use lib::Tensor;

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

    let x = Tensor::randn(&[4, 4]);
    let y = Tensor::randn(&[4, 4]);
    let mut z = &x + &y; // TODO: autoderef
    println!("{}", x);
    println!("{}", y);
    println!("{}", z);

    println!("{:?}", x.grad);
    println!("{:?}", y.grad);
    z.backward();
    println!("{:?}", x.grad);
    println!("{:?}", y.grad);
}

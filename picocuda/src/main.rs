mod lib; // TODO?
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
    println!("{:?}", x);
    println!("{}", x);

    let y = Tensor::randn(&[4, 4]);
    println!("{:?}", y);
    println!("{}", y);

    let z = x + y;
    println!("{:?}", z);
    println!("{}", z);

    // let y = x.view(&[16]);
    // println!("{}", y);

    // let z = y.view(&[-1, 8]);
    // println!("{}", z);
}

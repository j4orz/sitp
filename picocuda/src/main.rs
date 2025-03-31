use picograd::Device;

fn main() {
    println!(
        "
    ⠀⠀⠀⠀⠀⣼⣧⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⣼⣿⣿⣧⠀⠀⠀⠀
    ⠀⠀⠀⠾⠿⠿⠿⠿⠷⠀⠀⠀
    ⠀⠀⣼⣆⠀⠀⠀⠀⣰⣧⠀⠀
    ⠀⣼⣿⣿⣆⠀⠀⣰⣿⣿⣧⠀
    ⠾⠟⠿⠿⠿⠧⠼⠿⠿⠿⠻⠷
    picograd: torch.tensor->CUDA compiler
    ⚠︎ this is a one man effort to compile llama and r1. beware of dragons ⚠︎
    "
    );

    // TODO: -smell?: map_err: tensor/op error -> python error in lib_py.rs
    let device = Device::Gpu;
    let X = picograd::tpy::ones(vec![3, 1]);
    let Y = picograd::tpy::ones(vec![1, 3]);

    let X = X.to(&device);
    let X = Y.to(&device);
    let Z = (&X * &Y).unwrap();
    // Z.backward();
}

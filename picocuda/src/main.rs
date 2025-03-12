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
    picograd: torch.tensor->{{PTX, Triton}} compiler
    ⚠︎ this is a one man effort to compile gpt2, llama3, and r1. beware of dragons ⚠︎
    "
    );

    // TODO:
    // ===forward:
    // -smell?: map_err: tensor/op error -> python error in lib_py.rs
    // -declarative rust module in lib_py.rs
    // -go through all comments. TODO, todo!(), panic!()...

    // ===backward:
    // speed: CPU perf
    // SSE, AVX, AVX2, AVX512, ARM

    let device = Device::Gpu;
    let X = picograd::tpy::ones(vec![3, 1]);
    let Y = picograd::tpy::ones(vec![1, 3]);

    let X = X.to(&device);
    let X = Y.to(&device);
    let Z = (&X * &Y).unwrap();
    // Z.backward();
}

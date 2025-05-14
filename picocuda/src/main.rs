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
    picograd: tensor->triton
    "
    );

    // TODO: -smell?: map_err: tensor/op error -> python error in lib_py.rs
    // TODO: unwraps(), todos, fixmes. wrap a bow.
    let device = Device::Gpu;
    let X = picograd::tpy::ones(vec![3, 1]);
    let Y = picograd::tpy::ones(vec![1, 3]);

    let X = X.to(&device);
    let X = Y.to(&device);
    let Z = (&X * &Y).unwrap();
    // Z.backward();
}

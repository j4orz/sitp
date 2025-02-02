mod lib; // TODO?
use lib::Tensor;

fn main() {
    let x = Tensor::randn(&[2, 3]);
    println!("{}", x);
}

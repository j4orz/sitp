use picoc::cfg;

fn main() {
    let foo = cfg::compile();        
    println!("bark {:?}", foo)
}
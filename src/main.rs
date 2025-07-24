use elements::graphs::{AdjacencyList, Graph};

fn main() {
    println!("Hello, world!");
    let al = AdjacencyList::new();
    let foo = al.min_spanning_tree().collect::<Vec<_>>();
}

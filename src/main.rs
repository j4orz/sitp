use elements::graphs::{AdjLinkedList, Graph};

fn main() {
    println!("Hello, world!");
    let al = AdjLinkedList::from_edges(&[
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3),
        (2, 3),
    ]);
    let foo = al.bfs(2).map(|v| v+2).collect::<Vec<_>>();
}
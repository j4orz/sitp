use elements::graph::{AdjLL, Graph};

fn main() {
    println!("Hello, world!");
    let g = AdjLL::from_edges(&[
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3),
        (2, 3),
    ]);
    let foo = g
        .into_bfs(NodeIndex::new(2))
        .map(|nid| nid+2).collect::<Vec<_>>();
}
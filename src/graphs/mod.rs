use std::collections::{BinaryHeap, HashMap};

struct UnionFind {}

// =============================================================================
// ALGORITHMS ==================================================================
// =============================================================================

pub trait Graph {
    type NodeRef<'a>;
    type EdgeRef<'a>;

    type NodeRefs<'a>: Iterator<Item=Self::NodeRef<'a>>;
    type EdgeRefs<'a>: Iterator<Item=Self::EdgeRef<'a>>;

    fn min_spanning_tree(&self) -> MST<Self> where Self: Sized {
        MST { g: todo!(), subgraphs: todo!(), edgeheap: todo!(), nodemap: todo!() }
    }

    fn sssp(&self) -> SSSP { todo!() }
    fn apsp(&self) -> APSP { todo!() }
}




struct SSSP {} // ______________________________________________________________
impl Iterator for SSSP {
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
} // ___________________________________________________________________________



struct APSP {} // ______________________________________________________________
impl Iterator for APSP {
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
} // ___________________________________________________________________________



struct MST<'a, G: Graph + ?Sized> { // _________________________________________
    g: &'a G,
    subgraphs: UnionFind,
    edgeheap: BinaryHeap<()>,
    nodemap: HashMap<(),()>
}

impl<'a, G: Graph + Sized> Iterator for MST<'a, G> {
    type Item = i32;
    fn next(&mut self) -> Option<Self::Item> { todo!() }
}
// _____________________________________________________________________________



// =============================================================================
// =============================================================================
// =============================================================================








// TRAITS ______________________________________________________________________

impl<V, E> Graph for AdjacencyList<V, E> {
    type NodeRef<'a>;

    type EdgeRef<'a>;

    type NodeRefs<'a>;

    type EdgeRefs<'a>;
}

// TODO: marker traits?
// _____________________________________________________________________________








// DATA STRUCTURES _____________________________________________________________
pub struct AdjacencyList<V, E> { nodes: Vec<V>, edges: Vec<(usize, usize, E)> }
impl<V, E> AdjacencyList<V, E> {
    pub fn new() -> Self { Self { nodes: Vec::new(), edges: Vec::new() }}
}
pub struct AdjacencyArray {}
pub struct EdgeList {}

// _____________________________________________________________________________
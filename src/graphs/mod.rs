use core::hash::Hash;

// =============================================================================
// ALGORITHMS ==================================================================
// =============================================================================
pub trait Graph {
    type NodeId: Hash;
    fn nodes(&self) -> impl Iterator<Item=Self::NodeId>;

    // NB: .neighbors(v: NodeId) returns nodes u_1,...,un adjacent to v
    //     .edges(v: NodeId)     returns (v,u_1),...,(v,u_n)
    fn neighbors(&self, v: Self::NodeId) -> impl Iterator<Item=Self::NodeId>;
    fn edges(&self, v: Self::NodeId) -> impl Iterator<Item=(Self::NodeId, Self::NodeId)>;

    fn into_bfs(self) -> IntoBfs<Self>
    where Self: Sized {
        IntoBfs { g: self  }
    }

    fn bfs(&self) -> Bfs<'_, Self>
    where Self: Sized {
        Bfs { g: self }
    }

    fn bfs_mut(&mut self) -> BfsMut<'_, Self>
    where Self: Sized {
        BfsMut { g: self }
    }
}

pub struct IntoBfs<G: Graph> { g: G }
pub struct Bfs<'a, G: Graph> { g: &'a G }
pub struct BfsMut<'a, G: Graph> { g: &'a mut G }

impl<G: Graph> Iterator for IntoBfs<G> {
    type Item = G::NodeId;

    fn next(&mut self) -> Option<Self::Item> { todo!() }
}

impl<'a, G: Graph> Iterator for Bfs<'_, G> {
    type Item = G::NodeId;

    fn next(&mut self) -> Option<Self::Item> { todo!() }
}

impl<'a, G: Graph> Iterator for BfsMut<'_, G> {
    type Item = G::NodeId;

    fn next(&mut self) -> Option<Self::Item> { todo!() }
}



// =============================================================================
// =============================================================================
// =============================================================================








// TRAITS ______________________________________________________________________


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
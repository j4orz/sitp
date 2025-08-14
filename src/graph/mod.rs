use std::{collections::{HashSet, VecDeque}, iter, hash::Hash, fmt::Debug};
mod traversals; use traversals::IntoBfs;
mod shortest_paths;

pub trait VisitMap<NId> { // polymorphic over visitation storages (i.e dense/sparse representations)
    fn visit(&mut self, v: &NId) -> bool;
    fn unvisit(&mut self, v: &NId) -> bool;
    fn visited(&self, v: NId) -> bool;
}

impl<NId, T> VisitMap<NId> for HashSet<T> { // TODO: HashSet<NId>?
    fn visit(&mut self, v: &NId) -> bool { todo!() }
    fn unvisit(&mut self, v: &NId) -> bool { todo!() }
    fn visited(&self, v: NId) -> bool { todo!() }
}
// impl<NId, T> VisitMap<NId> for BitSet

pub trait Graph { // polymorphic over graph storage
    type NId: Copy + PartialEq; // ids need to be copyable and identifiable
    type EId: Copy + PartialEq;
    fn node_ids(&self) -> impl Iterator<Item=Self::NId>;
    fn neighbors(&self, v: Self::NId) -> impl Iterator<Item=Self::NId>;
    type Map: VisitMap<Self::NId>;
    fn visit_map(&self) -> Self::Map;
    fn reset_map(&self, map: &mut Self::Map);
    // _________________________________________________________________________

    fn into_bfs(self, from: Self::NId) -> IntoBfs<Self> where Self: Sized {
        let vm = self.visit_map(); // visit_map before self moves to IntoBfs combinator
        IntoBfs { g: self, q: VecDeque::from([from]), visited: vm }
    }
}

pub trait DirectedGraph : Graph {
    fn succs(&self, v: Self::NId) -> impl Iterator<Item=Self::NId>;
    fn preds(&self, v: Self::NId) -> impl Iterator<Item=Self::NId>;
}





// ADJLL _______________________________________________________________________
impl<N, E, I: Index> Graph for AdjLL<N, E, I> { // polymorphic over indices (edge devices)
    type NId = NodeIndex<I>;
    type EId = EdgeIndex<I>;
    fn node_ids(&self) -> impl Iterator<Item=Self::NId> { iter::empty() }
    fn neighbors(&self, v: Self::NId) -> impl Iterator<Item=Self::NId> { iter::empty() }
    type Map = HashSet<Self::NId>;
    fn visit_map(&self) -> Self::Map { todo!() }
    fn reset_map(&self, map: &mut Self::Map) { todo!() }
}

pub trait Index: Copy + Default + Hash + Ord + Debug + 'static {
    fn new(i: usize) -> Self;
    fn index(&self) -> usize;
    fn max() -> Self;
}

#[derive(Clone, Copy, PartialEq)] pub struct NodeIndex<I>(I);
#[derive(Clone, Copy, PartialEq)] pub struct EdgeIndex<I>(I);
impl<I> NodeIndex<I> { pub fn new(i: I) -> Self { NodeIndex(i)}}
impl<I> EdgeIndex<I> { pub fn new(i: I) -> Self { EdgeIndex(i)}}

// pub const INVALID_EDGE_INDEX: EdgeIndex = EdgeIndex(usize::MAX);
// const OUTGOING: usize = 0;
// const INCOMING: usize = 1;

/// adjlist representation of a graph using linkedlists, based off rustc's linkedgraph,
/// with some generic modifications (index and direction) based off the petgraph crate.
/// `adjlinkedlist` contiguously stores the set of nodes and edges in two flat vectors,
/// and each node stores it's adjacent neighbors with the head of a linkedlist of
/// edge indices, which index into the vector of edges.
/// see: https://smallcultfollowing.com/babysteps/blog/2015/04/06/modeling-graphs-in-rust-using-vector-indices/
/// see: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_data_structures/graph/linked_graph/struct.LinkedGraph.html
/// 
/// TODO: logical graph
///       physical graph
pub struct AdjLL<N, E, I> { nodes: Vec<AdjLLNode<N, I>>, edges: Vec<AdjLLEdge<E, I>> }
struct AdjLLNode<N, I> { data: N, head_edges: [EdgeIndex<I>; 2] }
struct AdjLLEdge<E, I> { data: E, next_edges: [EdgeIndex<I>; 2], src: NodeIndex<I>, tgt: NodeIndex<I> }



impl<N, E, I: Index> AdjLL<N, E, I> {
    pub fn new() -> Self { Self { nodes: Vec::new(), edges: Vec::new() }}
    pub fn from_edges<It: IntoIterator>(i: It) -> Self { todo!() }

    pub fn add_node(&mut self, data: N) -> NodeIndex<I> {
        // let i = NodeIndex(self.nodes.len());
        // self.nodes.push(Node { data, head_edges: [INVALID_EDGE_INDEX, INVALID_EDGE_INDEX] });
        // i
        todo!()
    }
    pub fn add_edge(&mut self, data: E, src: NodeIndex<I>, tgt: NodeIndex<I>) -> EdgeIndex<I> {
        // let src_outgoing_head = self.nodes[src.0].head_edges[OUTGOING];
        // let tgt_incoming_head = self.nodes[tgt.0].head_edges[INCOMING];
        // self.edges.push(Edge { data, next_edges: todo!(), src, tgt });
        todo!()
    }

    pub fn node_weights(&self) -> impl Iterator<Item=N> { std::iter::empty() }
    pub fn node_references(&self) -> impl Iterator<Item=(NodeIndex<I>, N)> { std::iter::empty() }
    // pub fn update_node() -> () {}
    // pub fn update_edge() -> () {}

    // pub fn delete_node() -> () {}
    // pub fn delete_edge() -> () {}
}

mod test_adjll {
    #[test]
    fn foo() {
        
    }
}
// _____________________________________________________________________________


impl Graph for AdjMat {
    type NId;
    type EId;

    fn node_ids(&self) -> impl Iterator<Item=Self::NId> { todo!() }
    fn neighbors(&self, v: Self::NId) -> impl Iterator<Item=Self::NId> { todo!() }

    type Map;
    fn visit_map(&self) -> Self::Map { todo!() }
    fn reset_map(&self, map: &mut Self::Map) { todo!() }
}
pub struct AdjMat {}
impl AdjMat {}

pub struct AdjHashMap<N, E, Idx> { nodes: Vec<AdjLLNode<N, Idx>>, edges: Vec<AdjLLEdge<E, Idx>> }
impl<N, E, Idx> AdjHashMap<N, E, Idx> {
    pub fn new() -> Self { Self { nodes: Vec::new(), edges: Vec::new() }}
    pub fn from_edges<I: IntoIterator>(i: I) -> Self { todo!() }
    pub fn foobazz() -> () { todo!() }

    // pub fn add_node(&mut self, data: N) -> NodeIndex<Idx> {
    //     let i = NodeIndex(self.nodes.len());
    //     self.nodes.push(Node { data, head_edges: [INVALID_EDGE_INDEX, INVALID_EDGE_INDEX] });
    //     i
    // }

    // pub fn add_edge(&mut self, data: E, src: NodeIndex<Idx>, tgt: NodeIndex<Idx>) -> EdgeIndex<Idx> {
    //     let src_outgoing_head = self.nodes[src.0].head_edges[OUTGOING];
    //     let tgt_incoming_head = self.nodes[tgt.0].head_edges[INCOMING];
    //     self.edges.push(Edge { data, next_edges: todo!(), src, tgt });
    //     todo!()
    // }

    // pub fn update_node() -> () {}
    // pub fn update_edge() -> () {}

    // pub fn delete_node() -> () {}
    // pub fn delete_edge() -> () {}
}


// TODO:
// - unsafe for bfsmut iterator adapter??
// - parallel bfs
// size, alignment of Option<usize> is ... NPO...
// TODO: Nethercote. measure.

//! the graph module is designed so that from the users/callers/clients perspective,
//! the logical algorithmic operations (located in submodules) are fixed, whereas
//! the physical datastructure representations can be extended by implementing
//! the graph trait for their custom representation.
//! 
//! the abstraction is zero-cost because of the use of static dispatch
//! via traits and higher kinded types (GATs), which works because the algorithms
//! are fixed and the data structures vary. the trait only specifies read-only
//! queries (with respect to the graph), and each graph representation implements
//! their own add, update, and remove methods. the methods reflect the fact the
//! graphs operate on indices. (TODO: some graph representation don't use indices?)
//! 
//! if users require the operations to vary without changing the library's source,
//! this becomes more complex in rust because of the lack of RTTI necessary for
//! dynamic cast operators. a visitor-like approach needs to be incorporated
mod traversals;
pub mod index;

// TODO:
// - unsafe for bfsmut iterator adapter??
// - parallel bfs

use std::{collections::{HashSet}, iter};
use crate::graphs::{index::{EdgeIndex, Index, NodeIndex}};

/// the VisitMap trait is a separate trait from the Graph trait, and is used
/// as an associated type for the latter. this is because the set of visited
/// nodes is transient state used in graph algorithms, and increases the
/// 1. library flexibility in borrow fideltiy (leasing &g *and* &mut vm) without the cost of interior mutability
/// 2. user extensibility in pluggable storages (hashset, bitset, densebitset,...) without the cost of M*N explosion
trait VisitMap<NId> {
    fn visit(&mut self, v: &NId) -> bool;
    fn unvisit(&mut self, v: &NId) -> bool;
    fn visited(&self, v: NId) -> bool;
}

/// the Graph trait is a unification of petgraph's three base traits using GATs:
/// 1. node/edge identifiers types (pointer, reference, handle, or index)
/// 2. adjacenct neighbors function
/// 3. visitation types and functions
pub trait Graph {
    type NodeId: Copy + PartialEq;
    type EdgeId: Copy + PartialEq;
    fn neighbors(&self, v: Self::NodeId) -> impl Iterator<Item=Self::NodeId>;
    type Map: VisitMap<Self::NodeId>;
    fn visit_map(&self) -> Self::Map;
    fn reset_map(&self, map: &mut Self::Map);



    // fn into_bfs(self, from: Self::NodeId) -> IntoBfs<Self> where Self: Sized { IntoBfs { g: self, q: VecDeque::from([from]), visited: HashSet::from([from]) } }
    // fn bfs(&self, from: Self::NodeId) -> Bfs<'_, Self> where Self: Sized { Bfs { g: self, q: VecDeque::from([from]), visited: HashSet::from([from]) } }
    // fn bfs_mut(&mut self, from: Self::NodeId) -> BfsMut<'_, Self> where Self: Sized { BfsMut { g: self, q: VecDeque::from([from]), visited: HashSet::from([from]) } }
    // fn into_dfs(self, from: Self::NodeId) -> IntoDfs<Self> where Self: Sized { IntoDfs { g: self, s: Vec::from([from]), visited: HashSet::from([from]) } }
    // fn dfs(&self, from: Self::NodeId) -> Dfs<'_, Self> where Self: Sized { Dfs { g: self, s: Vec::from([from]), visited: HashSet::from([from]) } }
    // fn dfs_mut(&mut self, from: Self::NodeId) -> DfsMut<'_, Self> where Self: Sized { DfsMut { g: self, s: Vec::from([from]), visited: HashSet::from([from]) } }
}




impl<NId, T> VisitMap<NId> for HashSet<T> { // TODO: HashSet<NId>?
    fn visit(&mut self, v: &NId) -> bool { todo!() }
    fn unvisit(&mut self, v: &NId) -> bool { todo!() }
    fn visited(&self, v: NId) -> bool { todo!() }
}

impl<N, E, Idx: Index> Graph for AdjLinkedList<N, E, Idx> {
    type NodeId = NodeIndex<Idx>;
    type EdgeId = EdgeIndex<Idx>;
    fn neighbors(&self, v: Self::NodeId) -> impl Iterator<Item=Self::NodeId> { iter::empty() }
    type Map = HashSet<Self::NodeId>;
    fn visit_map(&self) -> Self::Map { todo!() }
    fn reset_map(&self, map: &mut Self::Map) { todo!() }
}





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
pub struct AdjLinkedList<N, E, Idx> { nodes: Vec<Node<N, Idx>>, edges: Vec<Edge<E, Idx>> }
struct Node<N, Idx> { data: N, head_edges: [EdgeIndex<Idx>; 2] }
struct Edge<E, Idx> { data: E, next_edges: [EdgeIndex<Idx>; 2], src: NodeIndex<Idx>, tgt: NodeIndex<Idx> }



impl<N, E, Idx> AdjLinkedList<N, E, Idx> {
    pub fn new() -> Self { Self { nodes: Vec::new(), edges: Vec::new() }}
    pub fn from_edges<I: IntoIterator>(i: I) -> Self { todo!() }

    pub fn add_node(&mut self, data: N) -> NodeIndex<Idx> {
        // let i = NodeIndex(self.nodes.len());
        // self.nodes.push(Node { data, head_edges: [INVALID_EDGE_INDEX, INVALID_EDGE_INDEX] });
        // i
        todo!()
    }
    pub fn add_edge(&mut self, data: E, src: NodeIndex<Idx>, tgt: NodeIndex<Idx>) -> EdgeIndex<Idx> {
        // let src_outgoing_head = self.nodes[src.0].head_edges[OUTGOING];
        // let tgt_incoming_head = self.nodes[tgt.0].head_edges[INCOMING];
        // self.edges.push(Edge { data, next_edges: todo!(), src, tgt });
        todo!()
    }

    pub fn node_indices(&self) -> impl Iterator<Item=NodeIndex<Idx>> { std::iter::empty() }
    pub fn node_weights(&self) -> impl Iterator<Item=N> { std::iter::empty() }
    pub fn node_references(&self) -> impl Iterator<Item=(NodeIndex<Idx>, N)> { std::iter::empty() }


    // pub fn update_node() -> () {}
    // pub fn update_edge() -> () {}

    // pub fn delete_node() -> () {}
    // pub fn delete_edge() -> () {}
}











pub struct AdjHashMap<N, E, Idx> { nodes: Vec<Node<N, Idx>>, edges: Vec<Edge<E, Idx>> }
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
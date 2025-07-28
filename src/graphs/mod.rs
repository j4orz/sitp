//! the graph module is designed so that from the users/callers/clients perspective,
//! the logical algorithmic operations (located in submodules) are fixed, whereas
//! the physical datastructure representations can be extended by implementing
//! the graph trait for their custom representation. while this abstraction is
//! zero-cost because of the use of static dispatch with traits and higher kinded types (GATs)
//! (as oposed to dynamic dispatch with vtables/vfunctions), if users require the
//! ability to extend operations without changing the library's source, this
//! becomes more complex in rust because of the lack of RTTI necessary for
//! dynamic cast operators. a visitor-like approach needs to be incorporated
mod traversals;


// TODO:
// - unsafe for bfsmut iterator adapter??
// - parallel bfs

use std::{collections::{HashSet, VecDeque}, iter};
use crate::graphs::traversals::{Bfs, BfsMut, Dfs, DfsMut, IntoBfs, IntoDfs};

/// the VisitMap trait is a separate trait from the Graph trait, and is used
/// as an associated type for the latter. this is because the set of visited
/// nodes is transient state used in graph algorithms, and increases the
/// 1. library flexibility in borrow fideltiy (leasing &g *and* &mut vm) without the cost of interior mutability
/// 2. user extensibility in pluggable storages (hashset, bitset, densebitset,...) without the cost of M*N problem
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



    fn into_bfs(self, from: Self::NodeId) -> IntoBfs<Self> where Self: Sized { IntoBfs { g: self, q: VecDeque::from([from]), visited: HashSet::from([from]) } }
    fn bfs(&self, from: Self::NodeId) -> Bfs<'_, Self> where Self: Sized { Bfs { g: self, q: VecDeque::from([from]), visited: HashSet::from([from]) } }
    fn bfs_mut(&mut self, from: Self::NodeId) -> BfsMut<'_, Self> where Self: Sized { BfsMut { g: self, q: VecDeque::from([from]), visited: HashSet::from([from]) } }
    fn into_dfs(self, from: Self::NodeId) -> IntoDfs<Self> where Self: Sized { IntoDfs { g: self, s: Vec::from([from]), visited: HashSet::from([from]) } }
    fn dfs(&self, from: Self::NodeId) -> Dfs<'_, Self> where Self: Sized { Dfs { g: self, s: Vec::from([from]), visited: HashSet::from([from]) } }
    fn dfs_mut(&mut self, from: Self::NodeId) -> DfsMut<'_, Self> where Self: Sized { DfsMut { g: self, s: Vec::from([from]), visited: HashSet::from([from]) } }
}




impl<NId, T> VisitMap<NId> for HashSet<T> { // TODO: HashSet<NId>?
    fn visit(&mut self, v: &NId) -> bool { todo!() }
    fn unvisit(&mut self, v: &NId) -> bool { todo!() }
    fn visited(&self, v: NId) -> bool { todo!() }
}

impl<N, E> Graph for AdjLinkedList<N, E> {
    type NodeId = NodeIndex;
    type EdgeId = EdgeIndex;
    fn neighbors(&self, v: Self::NodeId) -> impl Iterator<Item=Self::NodeId> {
        iter::empty()
    }
    
    type Map = HashSet<Self::NodeId>;
    fn visit_map(&self) -> Self::Map { todo!() }
    fn reset_map(&self, map: &mut Self::Map) { todo!() }
}







#[derive(Clone, Copy, PartialEq)] pub struct NodeIndex(usize);
#[derive(Clone, Copy, PartialEq)] pub struct EdgeIndex(usize);

struct Node<N> { data: N, first: [EdgeIndex; 2] }
struct Edge<E> { data: E, src: NodeIndex, tgt: NodeIndex, next: [EdgeIndex; 2] }
pub struct AdjLL<N, E> { nodes: Vec<Node<N>>, edges: Vec<Edge<E>> }
//      ┌───────────── Nodes Vec ─────────────┐
// Idx  │ first_out   first_in                │
// ─────┼─────────────────────────────────────┤
//  0   │   0  ───► e0   3  ───► e3           │   (node 0 = “A”)
//  1   │   2  ───► e2   0  ───► e0           │   (node 1 = “B”)
//  2   │   3  ───► e3   1  ───► e1 ─► e2     │   (node 2 = “C”)
//      └─────────────────────────────────────┘

//      ┌────────────────────── Edges Vec ──────────────────────┐
// Idx  │    src → dst     next_out   next_in                   │
// ─────┼───────────────────────────────────────────────────────┤
//  e0  │   0  → 1         1          –                         │   (A → B)
//  e1  │   0  → 2         –          2                         │   (A → C)
//  e2  │   1  → 2         –          –                         │   (B → C)
//  e3  │   2  → 0         –          –                         │   (C → A)
//      └───────────────────────────────────────────────────────┘
// Legend: “–” = `EdgeIndex::end()` / “no next”

impl<N, E> AdjLL<N, E> {
    // NB: create, update, and delete methods are defined in concrete impl
    //     whereas read methods are defined in trait impl for concrete type,
    //     given that algorithms are read-only queries (with respect to the graph).
    //     if destructive algorithms are required, then the graph trait needs to be modified.
    pub fn new() -> Self { Self { nodes: Vec::new(), edges: Vec::new() }}
    pub fn from_edges<I: IntoIterator>(i: I) -> Self {
        for foo in i.into_iter() {

        }

        todo!()
    }

    fn gen_node_index(&self) -> NodeIndex { NodeIndex(self.nodes.len()) }
    pub fn add_node(&mut self, data: N) -> NodeIndex {
        let i = self.gen_node_index();
        self.nodes.push(Node { data });
        i
    }
}

pub struct AdjVec {}
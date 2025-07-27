//! this module contains a single graph trait which logically defines graph algorithms
//! generically, independant from their physical representation in memory.
//! this design is 1. zero cost and 2. non-anemic since the plethora of graph algorithms
//! implemented via provided methods can treat graphs homogenously through required methods.
//! that is, variant-specific logic via virtual dispatch is not required.
//! both required and provided methods hook into rust iterators.
mod traversals;



// TODO:
// - associated types?
// - generic associated types?
// - unsafe for bfsmut iterator adapter??
// - parallel bfs

use std::{collections::{HashSet, VecDeque}, iter};
use crate::graphs::traversals::{Bfs, BfsMut, IntoBfs};
pub trait Graph {
    // from the perspective of the logical algorithm, physical data structures
    // need to provide *some* way of identifying nodes and edges. the choice
    // of whether this identifier is a pointer, a reference, a handle, or an index
    // is up to the data structure, and does not concern the algorithm.
    type NodeId: Copy + PartialEq;
    type EdgeId: Copy + PartialEq;

    fn nodes(&self) -> impl Iterator<Item=Self::NodeId>;
    fn edges(&self) -> impl Iterator<Item=Self::EdgeId>;
    fn neighbors(&self, v: Self::NodeId) -> impl Iterator<Item=Self::NodeId>;


    fn into_bfs(self, v: Self::NodeId) -> IntoBfs<Self>
    where Self: Sized {
        IntoBfs { g: self, frontier: VecDeque::from([v]), visited: HashSet::from([v])  }
    }

    fn bfs(&self, v: Self::NodeId) -> Bfs<'_, Self>
    where Self: Sized {
        Bfs { g: self, frontier: Vec::from([v]), visited: HashSet::from([v])  }
    }

    fn bfs_mut(&mut self, v: Self::NodeId) -> BfsMut<'_, Self>
    where Self: Sized {
        BfsMut { g: self, frontier: Vec::from([v]), visited: HashSet::from([v]) }
    }
}








impl<N, E> Graph for AdjLL<N, E> {
    type NodeId = NodeIndex;
    type EdgeId = EdgeIndex;

    fn nodes(&self) -> impl Iterator<Item=Self::NodeId> {
        iter::empty()
    }

    fn edges(&self) -> impl Iterator<Item=Self::EdgeId> {
        iter::empty()
    }

    fn neighbors(&self, v: Self::NodeId) -> impl Iterator<Item=Self::NodeId> {
        iter::empty()
    }
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
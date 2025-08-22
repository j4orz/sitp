mod cc;
mod sp;
mod mst;

use std::collections::{HashSet, VecDeque};

pub struct Graph { al: Vec<Vec<usize>> }
impl Graph { pub fn new(al: Vec<Vec<usize>>) -> Self { Self { al, } } }

impl Graph {
    pub fn into_bfs(self, s: usize) -> IntoBfs {
        IntoBfs { g: self, q: VecDeque::from([s]), v: HashSet::new(), }
    }
}

pub struct IntoBfs {
    pub g: Graph,
    pub q: VecDeque<usize>,
    pub v: HashSet<usize>
}

impl Iterator for IntoBfs {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(i) = self.q.pop_front() {
            for j in &self.g.al[i] { if !self.v.contains(&j) { self.q.push_back(j.clone()); } }
            return Some(i);
        }
        
        None
    }
}

// //!            space            read                     update            delete
// //! adjlist
// //! adjmat
// //! csr
// use std::{collections::{HashSet, VecDeque}, fmt::Debug, hash::Hash, iter, marker::PhantomData};
// mod traversals; use traversals::IntoBfs;
// mod shortest_paths;

// pub trait Graph { // polymorphic over graph storage
//     type NId: Copy + PartialEq; // ids need to be copyable and identifiable
//     type EId: Copy + PartialEq;
//     fn node_ids(&self) -> impl Iterator<Item=Self::NId>;
//     fn neighbors(&self, v: Self::NId) -> impl Iterator<Item=Self::NId>;
//     type Map: VisitMap<Self::NId>;
//     fn visit_map(&self) -> Self::Map;
//     fn reset_map(&self, map: &mut Self::Map);
//     // _________________________________________________________________________

//     fn into_bfs(self, from: Self::NId) -> IntoBfs<Self> where Self: Sized {
//         let vm = self.visit_map(); // visit_map before self moves to IntoBfs combinator
//         IntoBfs { g: self, q: VecDeque::from([from]), visited: vm }
//     }
// }

// pub trait DirectedGraph : Graph {
//     fn succs(&self, v: Self::NId) -> impl Iterator<Item=Self::NId>;
//     fn preds(&self, v: Self::NId) -> impl Iterator<Item=Self::NId>;
// }

// //         time




// //                                  ^^^^^^^^^^
// // =================================ALGORITHMS==================================






// pub trait VisitMap<NId> { // polymorphic over visitation storages (i.e dense/sparse NIds)
//     fn visit(&mut self, v: &NId) -> bool;
//     fn unvisit(&mut self, v: &NId) -> bool;
//     fn visited(&self, v: NId) -> bool;
// }

// impl<NId, T> VisitMap<NId> for HashSet<T> { // TODO: HashSet<NId>?
//     fn visit(&mut self, v: &NId) -> bool { todo!() }
//     fn unvisit(&mut self, v: &NId) -> bool { todo!() }
//     fn visited(&self, v: NId) -> bool { todo!() }
// }
// // impl<NId, T> VisitMap<NId> for BitSet










// // ===============================DATA STRUCTURES===============================
// //                                vvvvvvvvvvvvvvv

// // ADJLL _______________________________________________________________________
// impl<N, E, I: Index, Ty> Graph for AdjList<N, E, I, Ty> { // polymorphic over indices (edge devices)
//     type NId = NodeIndex<I>;
//     type EId = EdgeIndex<I>;
//     fn node_ids(&self) -> impl Iterator<Item=Self::NId> { iter::empty() }
//     fn neighbors(&self, v: Self::NId) -> impl Iterator<Item=Self::NId> { iter::empty() }
//     type Map = HashSet<Self::NId>;
//     fn visit_map(&self) -> Self::Map { todo!() }
//     fn reset_map(&self, map: &mut Self::Map) { todo!() }
// }

// pub trait Index: Copy + Default + Hash + Ord + Debug + 'static {
//     fn new(i: usize) -> Self;
//     fn index(&self) -> usize;
//     fn max() -> Self;
// }

// #[derive(Clone, Copy, PartialEq)] pub struct NodeIndex<I>(I);
// #[derive(Clone, Copy, PartialEq)] pub struct EdgeIndex<I>(I);
// impl<I> NodeIndex<I> { pub fn new(i: I) -> Self { NodeIndex(i)}}
// impl<I> EdgeIndex<I> { pub fn new(i: I) -> Self { EdgeIndex(i)}}

// // pub const INVALID_EDGE_INDEX: EdgeIndex = EdgeIndex(usize::MAX);
// // const OUTGOING: usize = 0;
// // const INCOMING: usize = 1;
// struct AdjLLNode<N, I> { data: N, head_edges: [EdgeIndex<I>; 2] }
// struct AdjLLEdge<E, I> { data: E, next_edges: [EdgeIndex<I>; 2], src: NodeIndex<I>, tgt: NodeIndex<I> }
// pub struct AdjList<N, E, I, Ty> {
//     nodes: Vec<AdjLLNode<N, I>>,
//     edges: Vec<AdjLLEdge<E, I>>,
//     ty: PhantomData<Ty>
// }
// impl<N, E, I: Index, Ty> AdjList<N, E, I, Ty> {
//     pub fn new() -> Self { Self { nodes: Vec::new(), edges: Vec::new() }}
//     pub fn from_outgoing_edges<It: IntoIterator>(i: It) -> Self { todo!() }
//     pub fn node_weights(&self) -> impl Iterator<Item=N> { std::iter::empty() }
//     pub fn node_references(&self) -> impl Iterator<Item=(NodeIndex<I>, N)> { std::iter::empty() }
// }

// mod test_adjll {
//     #[test]
//     fn foo() {
        
//     }
// }

// // STABLEADJLL _________________________________________________________________




// // ADJVEC ______________________________________________________________________
// struct Edge<E> {
//     node: usize,
//     weight: E
// }


// pub struct AdjVec<N, E, I, Ty> { al: Vec<Vec<Edge<E>>> }
// impl<N, E, I: Index, Ty> AdjVec<N, E, I, Ty> {
//     pub fn new() -> Self { Self { al: vec![vec![]] }}
//     pub fn from_edges<It: IntoIterator>(i: It) -> Self { todo!() }
//     pub fn node_weights(&self) -> impl Iterator<Item=N> { std::iter::empty() }
//     pub fn node_references(&self) -> impl Iterator<Item=(NodeIndex<I>, N)> { std::iter::empty() }
// }








// // ADJMAT ______________________________________________________________________
// pub struct AdjMat<N, E, Ty> {
//     mat: Vec<Vec<Option<T>>>
// }
// impl<N, E, Ty> AdjMat<N, E, Ty> {
//     fn new() -> Self { Self { mat: vec![vec![]] }}
// }


// // TODO:
// // - unsafe for bfsmut iterator adapter??
// // - parallel bfs
// // size, alignment of Option<usize> is ... NPO...
// // TODO: Nethercote. measure.

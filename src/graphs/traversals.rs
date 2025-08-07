use std::collections::{HashSet, VecDeque};

use crate::graphs::{Graph, VisitMap};

pub struct _IntoBfs<G: Graph, VM: VisitMap<G::NodeId>> { pub g: G, pub q: VecDeque<G::NodeId>, pub visited: VM }
pub struct _Bfs<G: Graph, VM: VisitMap<G::NodeId>> { pub g: G, pub q: VecDeque<G::NodeId>, pub visited: VM }
pub struct _BfsMut<G: Graph, VM: VisitMap<G::NodeId>> { pub g: G, pub q: VecDeque<G::NodeId>, pub visited: VM }

impl<G: Graph, VM: VisitMap<G::NodeId>> Iterator for _IntoBfs<G, VM> {
    type Item = G::NodeId;

    fn next(&mut self) -> Option<Self::Item> {
        self.q.pop_front().map(|v| {
            let _ = self.g
                .neighbors(v.clone())
                .filter(|u| self.visited.visit(u))
                .for_each(|u| self.q.push_back(u));
            v
        })
    }
}

// impl<'g, G: Graph> Iterator for Bfs<'_, G> {
//     type Item = G::NodeId;

//     fn next(&mut self) -> Option<Self::Item> { todo!() }
// }

// impl<'g, G: Graph> Iterator for BfsMut<'_, G> {
//     type Item = G::NodeId;

//     fn next(&mut self) -> Option<Self::Item> { todo!() }
// }

pub struct _IntoDfs<G: Graph> { pub g: G, pub s: Vec<G::NodeId>, pub visited: HashSet<G::NodeId> }
pub struct _Dfs<'g, G: Graph> { pub g: &'g G, pub s: Vec<G::NodeId>, pub visited: HashSet<G::NodeId> }
pub struct _DfsMut<'g, G: Graph> { pub g: &'g mut G, pub s: Vec<G::NodeId>, pub visited: HashSet<G::NodeId> }

impl<G: Graph> Iterator for _IntoDfs<G> {
    type Item = G::NodeId;

    fn next(&mut self) -> Option<Self::Item> {
        self.s.pop().map(|v| {
            let _ = self.g
                .neighbors(v.clone())
                .filter(|u| self.visited.visit(u))
                .for_each(|u| self.s.push(u));
            v
        })
    }
}

impl<'a, G: Graph> Iterator for _Dfs<'_, G> {
    type Item = G::NodeId;

    fn next(&mut self) -> Option<Self::Item> { todo!() }
}

impl<'a, G: Graph> Iterator for _DfsMut<'_, G> {
    type Item = G::NodeId;

    fn next(&mut self) -> Option<Self::Item> { todo!() }
}
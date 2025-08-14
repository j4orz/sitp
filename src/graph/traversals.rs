use std::collections::{HashSet, VecDeque};
use crate::graph::{Graph, VisitMap};

pub struct IntoBfs<G: Graph> { pub g: G, pub q: VecDeque<G::NId>, pub visited: G::Map }
impl<G: Graph> Iterator for IntoBfs<G> {
    type Item = G::NId;

    fn next(&mut self) -> Option<Self::Item> {
        self.q.pop_front().map(|v| {
            let _ =
                self.g
                .neighbors(v.clone())
                .filter(|u| self.visited.visit(u))
                .for_each(|u| self.q.push_back(u));
            v
        })
    }
}

pub struct IntoDfs<G: Graph> { pub g: G, pub s: Vec<G::NId>, pub visited: HashSet<G::NId> }
pub struct Dfs<'g, G: Graph> { pub g: &'g G, pub s: Vec<G::NId>, pub visited: HashSet<G::NId> }
pub struct DfsMut<'g, G: Graph> { pub g: &'g mut G, pub s: Vec<G::NId>, pub visited: HashSet<G::NId> }

impl<G: Graph> Iterator for IntoDfs<G> {
    type Item = G::NId;

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

impl<'a, G: Graph> Iterator for Dfs<'_, G> {
    type Item = G::NId;

    fn next(&mut self) -> Option<Self::Item> { todo!() }
}

impl<'a, G: Graph> Iterator for DfsMut<'_, G> {
    type Item = G::NId;

    fn next(&mut self) -> Option<Self::Item> { todo!() }
}
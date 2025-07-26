use std::collections::{HashSet, VecDeque};

use crate::graphs::Graph;

pub struct IntoBfs<G: Graph> { pub g: G, pub frontier: VecDeque<G::NodeId>, pub seen: HashSet<G::NodeId> }
pub struct Bfs<'a, G: Graph> { pub g: &'a G, pub frontier: Vec<G::NodeId>, pub seen: HashSet<G::NodeId> }
pub struct BfsMut<'a, G: Graph> { pub g: &'a mut G, pub frontier: Vec<G::NodeId>, pub seen: HashSet<G::NodeId> }

impl<G: Graph> Iterator for IntoBfs<G> {
    type Item = G::NodeId;

    fn next(&mut self) -> Option<Self::Item> {
        self.frontier.pop_front().map(|v| {
            let _ = self.g
                .neighbors(v.clone())
                .filter(|u| self.seen.insert(u.clone()))
                .for_each(|u| self.frontier.push_back(u));
            v
        })
    }
}

impl<'a, G: Graph> Iterator for Bfs<'_, G> {
    type Item = G::NodeId;

    fn next(&mut self) -> Option<Self::Item> { todo!() }
}

impl<'a, G: Graph> Iterator for BfsMut<'_, G> {
    type Item = G::NodeId;

    fn next(&mut self) -> Option<Self::Item> { todo!() }
}
use std::{collections::{HashSet, VecDeque}, fmt::Debug};
use crate::opto::Fn;

pub type NodeId = usize;
pub struct Fixpoint<D> where D: DataflowAnal { pub in_facts: Vec<D::Fact>, pub out_facts: Vec<D::Fact> }
fn dflow_solver<D: DataflowAnal>(cfg: Fn<bril::Instruction>, dfa: D) -> Fixpoint<D> {
    let n = cfg.size();
    let (mut in_facts, out_facts)  = (Vec::with_capacity(n), Vec::with_capacity(n));
    let mut q = cfg.node_indices().collect::<VecDeque<_>>();
    // todo: optional boundary condition?

    // todo: handle backward analyses
    while let Some(node_id) = q.pop_front() {
        let acc = D::bot();
        for p in &cfg.preds(node_id) { D::meet(&mut acc, &in_facts[p]); }; // 1. meet
        // in_facts[bb] =

        let newo = dfa.xfn(in_facts[node_id]); // 2. xfn
        if newo != out_facts[node_id] { out_facts[node_id] = newo; for s in &cfg.succs(node_id) { q.push_back(s); } }
    }

    Fixpoint { in_facts, out_facts }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Dir { F, B }
pub trait DataflowAnal {
    // lattice
    type Fact: Clone + Debug + PartialEq;
    fn bot() -> Self::Fact;
    fn top() -> Self::Fact;
    fn meet(x: &Self::Fact, y: &Self::Fact) -> Self::Fact; // glb
    fn join(x: &Self::Fact, y: &Self::Fact) -> Self::Fact; // lub

    const DIR: Dir;   
    fn xfn(&self, x: Self::Fact) -> Self::Fact;
}

struct ReachingAnal {}
impl ReachingAnal { fn new() -> Self { todo!( )}}
// impl Dataflow for ReachingAnal {}
// backward, meet = union, transfer = USE ∪ (OUT \ DEF)
struct LiveAnal { uses: Vec<HashSet<usize>>, defs: Vec<HashSet<usize>>, }
impl LiveAnal { fn new() -> Self { Self { uses: todo!(), defs: todo!() }} }

struct AvailAnal {}
impl AvailAnal { fn new() -> Self { todo!( )}}
// impl Dataflow for AvailAnal {}

struct VeryBusyAnal {}
impl VeryBusyAnal { fn new() -> Self { todo!( )}}
// impl Dataflow for VeryBusyAnal {}

struct ConstPropAnal {}
impl ConstPropAnal { fn new() -> Self { todo!( )}}
// impl Dataflow for ConstPropAnal {}
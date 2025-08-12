use std::{collections::{HashSet, VecDeque}, fmt::Debug};
use crate::opto::Fn;

pub type NodeId = usize;
fn dflow_solver<D: Dataflow>(cfg: Fn<bril::Instruction>, dfa: D) -> Fixpoint<D> {
    let mut q = cfg.node_indices().collect::<VecDeque<_>>();
    let (mut i, mut o) = (vec![], vec![]);

    while let Some(bb) = q.pop_front() {
        match D::DIR {
        Dir::F => {
            let acc = dfa.bot();
            for p in &cfg.preds(bb) { dfa.meet(&mut acc, &in_state[s]); }; // 1. meet
            // in_facts[bb] =

            let newo = dfa.xfn(i[bb]); // 2. xfn
            if newo != o[bb] { o[bb] = newo; for s in &cfg.succs(bb) { q.push_back(s); } }
        },
        Dir::B => {}
        }
    }

    Fixpoint { i, o }
}


#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Dir { F, B }
pub trait Lattice: Clone + Debug + PartialEq {
    fn bot() -> Self; // initial value for may analyses pesimistic?
    fn top() -> Self; // initial value must analyses optimistic?
    fn join(&self, other: &Self) -> Self; // lub (for may analyses)
    fn meet(&self, other: &Self) -> Self; // glb (for must analyses)
}
pub trait XFn<L: Lattice> { fn apply(&self, node: NodeId, input: &L) -> L; }

pub trait Dataflow {
    type Domain: Lattice; type XFn: XFn<Self::Domain>; const DIR: Dir;
    
    fn xfn(&self) -> &Self::XFn;
    fn init_val(&self) -> Self::Domain; // inital value for nodes (except entry/exit)
    fn bound_val(&self) -> Self::Domain; // Boundary condition (value at entry for forward, exit for backward)
    fn is_may_analysis(&self) -> bool; // Whether this is a may analysis (use join) or must analysis (use meet)
}
#[derive(Debug, Clone)] pub struct Fixpoint<D: Dataflow> { pub i: Vec<D::Domain>, pub o: Vec<D::Domain> }













struct ReachingAnal {}
// impl Dataflow for ReachingAnal {}



// backward, meet = union, transfer = USE ∪ (OUT \ DEF)
struct LiveAnal { uses: Vec<HashSet<usize>>, defs: Vec<HashSet<usize>>, }
impl LiveAnal { fn new() -> Self { Self { uses: todo!(), defs: todo!() }} }

impl Dataflow for LiveAnal {
    type Domain;
    type XFn;
    const DIR: Dir;

    fn xfn(&self) -> &Self::XFn { todo!() }
    fn init_val(&self) -> Self::Domain { todo!() }
    fn bound_val(&self) -> Self::Domain { todo!() }
    fn is_may_analysis(&self) -> bool { todo!() }
}







struct AvailAnal {}
// impl Dataflow for AvailAnal {}

struct VeryBusyAnal {}
// impl Dataflow for VeryBusyAnal {}

struct ConstPropAnal {}
// impl Dataflow for ConstPropAnal {}
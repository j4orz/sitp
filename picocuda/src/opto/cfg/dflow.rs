use std::collections::VecDeque;
use std::fmt::Debug;
use crate::opto::OptodPrg;

pub type NodeId = usize;
fn dflow_solver<L: Lattice>(g: OptodPrg, d: Dir) -> Fixpoint<L> {
    // initialize

    // boundary conditions

    let mut worklist: VecDeque<NodeId> = self.cfg.nodes.iter().copied().collect();
    let mut in_worklist: HashSet<NodeId> = self.cfg.nodes.iter().copied().collect();

    while let Some(node) = worklist.pop_front() {
        in_worklist.remove(&node);
        
        let changed =
        match d {
        Dir::F => self.process_forward(node, &mut state),
        Dir::B => self.process_backward(node, &mut state),
        };
        
        if changed {
            // Add affected nodes to worklist
            let affected =
            match d {
            Dir::F => self.cfg.successors(node),
            Dir::B => self.cfg.predecessors(node),
            };
            
            for affected_node in affected {
                if !in_worklist.contains(&affected_node) {
                    worklist.push_back(affected_node);
                    in_worklist.insert(affected_node);
                }
            }
        }
    }

    state
}


#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Dir { F, B }
pub trait Lattice: Clone + Debug + PartialEq {
    fn bot() -> Self; // initial value for may analyses pesimistic?
    fn top() -> Self; // initial value must analyses optimistic?
    fn join(&self, other: &Self) -> Self; // lub    
    fn meet(&self, other: &Self) -> Self; // glb
}
pub trait XFn<L: Lattice> { fn apply(&self, node: NodeId, input: &L) -> L; }

pub trait Dataflow: Clone {
    type L: Lattice; type XFn: XFn<Self::L>; const D: Dir;
    
    fn transfer_function(&self) -> &Self::XFn;
    fn initial_value(&self) -> Self::L; // inital value for nodes (except entry/exit)
    fn boundary_value(&self) -> Self::L; // Boundary condition (value at entry for forward, exit for backward)
    fn is_may_analysis(&self) -> bool; // Whether this is a may analysis (use join) or must analysis (use meet)
}
#[derive(Debug, Clone)] pub struct Fixpoint<L: Lattice> { pub in_states: Vec<L>, pub out_states: Vec<L> }
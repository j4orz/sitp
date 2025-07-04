// TODOs (day5)
// - ir dump ascii/graphviz, then finish scope tests
// - remove golden tests for graphs: the expected graph changes too often on peephole additions
// - disable peephole (-O0)
// - ssa
pub mod generator;
pub mod optimizer;
pub mod parser;
pub mod dumper;
pub mod utils;

use std::{cell::RefCell, collections::VecDeque, fmt::{Debug, Display}, ops::Deref, rc::{Rc, Weak}};
use thiserror::Error;
use crate::son::optimizer::Type;

// NB: parallelizing the compiler requires moving the static mutable counter into TLS
//     also skipping 0 since node ids show up in bit vectors, work lists, etc.
static mut NODEID_COUNTER: usize = 0;
pub fn generate_nodeid() -> usize { unsafe { NODEID_COUNTER += 1; NODEID_COUNTER } }

// some code in simple relies on invariant that first edge is control.
// this is removed for now so edge type is not optioned. watch out for this.
// removed: matched on the opcode. if add/sub/etc set self.defs[0] to none}

// TODO: is dynamic matching on opcode too slow vs dynamic dispatch with vtables (trait items) or static __ with generics?
#[derive(Clone, Copy)] pub enum OpCode { Start, Ret, Con, Add, Sub, Mul, Div, Scope }
pub struct Node { id: usize, pub opcode: OpCode, typ: Type, defs: VecDeque<DefEdge>, uses: VecDeque<UseEdge> }
#[derive(Error, Debug)] pub enum NodeError { #[error("use not found")] UseNotFound }
// NB: - all nodes including control have types.
//     - the order of defs has semantic meaning. order of uses does not.
//       q: what's the point of maintaining D->U edges? (aka outputs/uses)

pub struct DefEdge(Rc<RefCell<Node>>); pub struct UseEdge(Weak<RefCell<Node>>);
impl Display for DefEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.borrow().opcode {
            OpCode::Start => write!(f, "Start"),
            OpCode::Ret => write!(f, "Ret"),
            OpCode::Con => write!(f, "Con_{}", self.borrow().typ),
            OpCode::Add => write!(f, "Add"),
            OpCode::Sub => write!(f, "Sub"),
            OpCode::Mul => write!(f, "Mul"),
            OpCode::Div => write!(f, "Div"),
            OpCode::Scope => write!(f, "Scope"),
        }
    }
}

impl Debug for DefEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.borrow().opcode {
            OpCode::Start => write!(f, "Start"),
            OpCode::Ret => write!(f, "Ret"),
            OpCode::Con => write!(f, "{}", self.borrow().typ),
            OpCode::Add => write!(f, "+"),
            OpCode::Sub => write!(f, "-"),
            OpCode::Mul => write!(f, "*"),
            OpCode::Div => write!(f, "/"),
            OpCode::Scope => write!(f, "nv"),
        }
    }
}

impl DefEdge {
    pub fn new_constant(op: OpCode, typ: Type) -> Self { Self(Rc::new(RefCell::new(Node { id: generate_nodeid(), opcode: op, typ, defs: VecDeque::new(), uses: VecDeque::new()}))) }
    pub fn new(op: OpCode) -> Self { Self(Rc::new(RefCell::new(Node { id: generate_nodeid(), opcode: op, typ: Type::Bot, defs: VecDeque::new(), uses: VecDeque::new() }))) }
    fn from_upgraded(strong: Rc<RefCell<Node>>) -> Self { Self(strong) }

    pub fn add_def(&self, def: &Self) -> () {
        self.borrow_mut().defs.push_back(def.clone());
        def.borrow_mut().uses.push_back(UseEdge::new(self));
    }

    fn is_cfg(&self) -> bool { match self.borrow().opcode {
        OpCode::Start | OpCode::Ret => true,
        _ => false
    }}

    pub fn unique_label(&self) -> String { format!("{}{}", self.to_string(), self.borrow().id) }
}

impl UseEdge { fn new(e: &DefEdge) -> Self { Self(Rc::downgrade(&e.0)) }}
impl Deref for DefEdge { type Target = Rc<RefCell<Node>>; fn deref(&self) -> &Self::Target { &self.0 }}
impl Deref for UseEdge { type Target = Weak<RefCell<Node>>; fn deref(&self) -> &Self::Target { &self.0 }}
impl Clone for DefEdge { fn clone(&self) -> Self { Self(self.0.clone()) }}
impl Clone for UseEdge { fn clone(&self) -> Self { Self(self.0.clone()) }}
impl PartialEq for UseEdge { fn eq(&self, other: &Self) -> bool { Weak::ptr_eq(&self.0, &other.0) }}
impl Drop for DefEdge { fn drop(&mut self) {
    if Rc::strong_count(&self.0) == 1 {
        while let Some(mut def) = self.borrow_mut().defs.pop_back() { def.del_use(&UseEdge::new(&self)).unwrap(); }
    }
}}
impl DefEdge {
    pub fn del_use(&mut self, u_target: &UseEdge) -> Result<(), NodeError> {
        let i = &self.borrow().uses.iter().position(|u| u == u_target).ok_or(NodeError::UseNotFound)?;
        let uses = &mut self.borrow_mut().uses;
        let _ = uses.make_contiguous();
        let (head, _tail) = uses.as_mut_slices();
        head.swap(*i, head.len()-1);
        uses.pop_back();
        Ok(())
    }
}
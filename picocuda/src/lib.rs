#![feature(assert_matches)]

// Qs
// - what's the point of maintaining D->U edges? (aka outputs/uses)

// TODOs (day4)
// - garbage collect weak edges from D->U
// - test scope
// - ir dump
// - disable peephole (-O0)

pub mod generator;
pub mod optimizer;
pub mod parser;

use std::{cell::RefCell, collections::VecDeque, ops::Deref, rc::{Rc, Weak}};
use thiserror::Error;
use crate::{optimizer::Type, parser::Parser};

pub struct NodeIdCounter(u128);
impl NodeIdCounter { fn new(start: u128) -> Self { Self(start)} fn gen_nodeid(&mut self) -> u128 { self.0 += 1; self.0 }}
pub fn driver() -> () {
    let mut nodeid_counter = NodeIdCounter::new(0);
    let mut parser = Parser::new(&mut nodeid_counter);
    let _ = parser.parse(&vec![]);
}

// NB. SoN is the third:
//     a. tree: precedence is represented via tree's hierarchy.
//     b. two-tiered nested graph of basic blocks of instructions: edges denote ONLY control flow
//     c. single-tiered flat graph of instructions: edges denote control flow OR data flow

#[derive(Debug, Clone, Copy)] pub enum OpCode { Start, Ret, Con, Add, Sub, Mul, Div, Scope }
#[derive(Debug)] pub struct Node {
    id: u128, opcode: OpCode, typ: Type, // all nodes including control have types
    defs: VecDeque<NodeDef>, uses: VecDeque<NodeUse>, // uses/users
}

// some code in simple relies on invariant that first edge is control.
// this is removed for now so edge type is not optioned. watch out for this.
// removed: matched on the opcode. if add/sub/etc set self.defs[0] to none
impl Node {
    pub fn new(nodeid_coutner: &mut NodeIdCounter, op: OpCode) -> NodeDef { NodeDef::new(Node { id: nodeid_coutner.gen_nodeid(), opcode: op, typ: Type::Bot, defs: VecDeque::new(), uses: VecDeque::new() })}
    pub fn new_constant(nodeid_counter: &mut NodeIdCounter, op: OpCode, typint: Type) -> NodeDef { NodeDef::new(Node { id: nodeid_counter.gen_nodeid(), opcode: op, typ: typint, defs: VecDeque::new(), uses: VecDeque::new()} )}
}

#[derive(Error, Debug)] pub enum NodeError { #[error("use not found")] UseNotFound }
#[derive(Debug)] pub struct NodeDef(Rc<RefCell<Node>>);
#[derive(Debug)] pub struct NodeUse(Weak<RefCell<Node>>);
impl NodeDef {
    fn new(n: Node) -> Self { Self(Rc::new(RefCell::new(n))) }
    pub fn add_def(&self, def: &Self) -> () {
        self.borrow_mut().defs.push_back(def.clone());
        def.borrow_mut().uses.push_back(NodeUse::new(self));
    }
}

impl NodeUse { fn new(e: &NodeDef) -> Self { Self(Rc::downgrade(&e.0)) }}
impl Deref for NodeDef { type Target = Rc<RefCell<Node>>; fn deref(&self) -> &Self::Target { &self.0 }}
impl Deref for NodeUse { type Target = Weak<RefCell<Node>>; fn deref(&self) -> &Self::Target { &self.0 }}
impl Clone for NodeDef { fn clone(&self) -> Self { Self(self.0.clone()) }}
impl Clone for NodeUse { fn clone(&self) -> Self { Self(self.0.clone()) }}
impl PartialEq for NodeUse { fn eq(&self, other: &Self) -> bool { Weak::ptr_eq(&self.0, &other.0) }}
impl Drop for NodeDef { fn drop(&mut self) {
    if Rc::strong_count(&self.0) == 1 {
        while let Some(mut def) = self.borrow_mut().defs.pop_back() { def.del_use(&NodeUse::new(&self)).unwrap(); }
    }
}}
impl NodeDef {
    pub fn del_use(&mut self, u_target: &NodeUse) -> Result<(), NodeError> {
        let i = &self.borrow().uses.iter().position(|u| u == u_target).ok_or(NodeError::UseNotFound)?;
        let uses = &mut self.borrow_mut().uses;
        let _ = uses.make_contiguous();
        let (head, _tail) = uses.as_mut_slices();
        head.swap(*i, head.len()-1);
        uses.pop_back();
        Ok(())
    }
}
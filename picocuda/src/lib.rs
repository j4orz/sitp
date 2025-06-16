#![feature(assert_matches)]

pub mod generator;
pub mod optimizer;
pub mod parser;

use std::{cell::RefCell, collections::VecDeque, ops::Deref, rc::{Rc, Weak}};
use crate::optimizer::Type;

// NB. generally speaking there are three variants for intermediate representation
//     a. tree ("AST")
//        -> precedence is represented via tree's hierarchy.
//     b. two-tiered (nested) graph of basic blocks of instructions. ("CFG+BB")
//        -> edges denote ONLY control flow
//     c. single-tiered (flat) graph of instructions ("SoN")
//        -> edges denote control flow OR data flow

#[derive(Debug)] pub enum OpCode { Start, Ret, Con, Add, Sub, Mul, Div }
#[derive(Debug)]
pub struct Node {
    opcode: OpCode, typ: Type, // all nodes including control have types
    defs: VecDeque<NodeDef>, uses: VecDeque<NodeUse>, // uses/users
}

impl Node {
    // some code in simple relies on invariant that first edge is control.
    // this is removed for now so edge type is not optioned. watch out for this.
    pub fn new(op: OpCode) -> NodeDef { NodeDef::new(Node { opcode: op, typ: Type::Bot, defs: VecDeque::new(), uses: VecDeque::new() }) }
    pub fn new_constant(op: OpCode, typint: Type) -> NodeDef { NodeDef::new(Node { opcode: op, typ: typint, defs: VecDeque::new(), uses: VecDeque::new()} )}

    pub fn add_input(slf: &NodeDef, d: NodeDef) -> () {
        slf.borrow_mut().defs.push_back(d.clone());
        d.borrow_mut().uses.push_back(NodeUse::new(slf));
    }
}

#[derive(Debug)] pub struct NodeDef(Rc<RefCell<Node>>);
#[derive(Debug)] pub struct NodeUse(Weak<RefCell<Node>>);
impl NodeDef { fn new(n: Node) -> Self { Self(Rc::new(RefCell::new(n))) } }
impl NodeUse { fn new(e: &NodeDef) -> Self { Self(Rc::downgrade(&e.0)) } }
impl Deref for NodeDef { type Target = Rc<RefCell<Node>>; fn deref(&self) -> &Self::Target { &self.0 } }
impl Deref for NodeUse { type Target = Weak<RefCell<Node>>; fn deref(&self) -> &Self::Target { &self.0 }}
impl Clone for NodeDef { fn clone(&self) -> Self { Self(self.0.clone()) } }
impl Clone for NodeUse { fn clone(&self) -> Self { Self(self.0.clone()) } }
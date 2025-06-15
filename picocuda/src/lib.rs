#![feature(assert_matches)]
pub mod generator;
pub mod optimizer;
pub mod parser;

#[derive(Debug)]
pub enum InstrNode {
    Start, Ret(Box<InstrNode>, Box<InstrNode>),
    Lit(i32), Add(Box<InstrNode>, Box<InstrNode>), Sub(Box<InstrNode>, Box<InstrNode>), Mul(Box<InstrNode>, Box<InstrNode>), Div(Box<InstrNode>, Box<InstrNode>)
}
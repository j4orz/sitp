use elements::graphs::AdjLinkedList;
use crate::sema::Ast;

pub mod cfg;
pub mod cfgssa;
pub mod son;

pub struct OptoConfig { pub ir: OptoIR, pub level: OptoLevel } impl OptoConfig { pub fn new(ir: OptoIR, level: OptoLevel) -> Self { Self { ir, level } } }
pub enum OptoIR { Ast, Cfg, CfgSsa, Son } pub enum OptoLevel { O0, O1 }
pub enum OptodPrg { Ast(Ast), Cfg(Cfg), CfgSsa(Cfg), Son }



pub type Cfg = Vec<AdjLinkedList<BB, (), usize>>;
#[derive(Debug)] pub struct BB(Vec<bril::Code>);
impl BB { fn new(instrs: Vec<bril::Code>) -> Self { Self(instrs) } }
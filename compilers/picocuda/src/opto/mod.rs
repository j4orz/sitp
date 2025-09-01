use elements::graphs::AdjLinkedList;
use crate::sema::Ast;

pub mod cfg;
pub mod cfgssa;
pub mod son;

pub struct OptoConfig { pub ir: OptoIR, pub level: OptoLevel } impl OptoConfig { pub fn new(ir: OptoIR, level: OptoLevel) -> Self { Self { ir, level } } }
pub enum OptoIR { Ast, Cfg, CfgSsa, Son } pub enum OptoLevel { O0, O1 }
pub enum OptodPrg { Ast(Ast), Cfg(Prg<bril::Code>), CfgSsa(Prg<bril::Code>), Son }

pub type Prg<I> = Vec<Fn<I>>;
pub type Fn<I> = AdjLinkedList<BB<I>, (), usize>;
#[derive(Debug)] pub struct BB<I>(pub Vec<I>);
impl<I> BB<I> { fn new(instrs: Vec<I>) -> Self { Self(instrs) } }
pub mod parser;
pub mod typer;
pub mod selector;
pub mod allocator;
pub mod encoder;
pub mod exporter;

use std::path::Path;
use thiserror::Error;
use crate::ast::{exporter::Format, typer::TypeError};

///////////////////////////// SOURCE (C89 subset) //////////////////////////////
pub type AbsPrg = Vec<Stmt>; pub enum Stmt { Ret(Expr) }
pub enum Expr {
    Con(i128),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>)
}
////////////////////////////////////////////////////////////////////////////////



////////////////////////// COMPILER: SOURCE -> TARGET //////////////////////////
#[derive(Error, Debug)] pub enum CompileError {
    #[error("type error")] TypeError(#[from] TypeError)
}
pub fn compile(src: &Path) -> Result<(), CompileError> {
    let ast = parser::parse(src);
    let _ = typer::typ()?;
    let aasmtree = selector::select(ast, CPU::R5, CallingConvention::SystemV);
    let asmtree = allocator::allocate(aasmtree);
    let machcode = encoder::encode(aasmtree);
    let elf = exporter::export(machcode, Format::Executable);
    // TODO: write elf to disk
    Ok(())
}
////////////////////////////////////////////////////////////////////////////////



///////////////////////// TARGET: {R5,ARM,x86} subset //////////////////////////
pub enum CPU { R5, ARM, X86 } pub enum CallingConvention { SystemV }
pub enum MachPrg { R5(Vec<R5MachInstr>), ARM(Vec<ARMInstr>), X86(Vec<X86Instr>) }
// NB: parallelizing the compiler requires moving the static mutable counter into TLS
//     also skipping 0 since node ids show up in bit vectors, work lists, etc.
static mut VREG_COUNTER: u32 = 0;
pub fn generate_vreg() -> u32 { unsafe { VREG_COUNTER += 1; VREG_COUNTER } }

pub enum R5OpCode { // TARGET R5
    Int, Int8, Add, AddI, Sub, Lui, Auipc, // arithmetic 
    Ret
}
struct R5MachInstr {
    // NB: machine instruction maintains retains semantic facts discovered/generated
    //     so 1. use->def facts (operands) and 2. registers (vreg/phyreg)
    opcode: R5OpCode, operands: Box<[R5MachInstr]>, // Box<[]> keeps children operands fixed arity
    vreg: u32, phyreg: Option<u32>,
}
impl R5MachInstr {
    fn new(opcode: R5OpCode, operands: Box<[R5MachInstr]>) -> Self {
        Self { opcode, operands, vreg: generate_vreg(), phyreg: None, }
    }
}

pub enum ARMInstr {} // TARGET ARM

pub enum X86Instr {} // TARGET x86
////////////////////////////////////////////////////////////////////////////////
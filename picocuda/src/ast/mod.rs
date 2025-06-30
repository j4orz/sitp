pub mod parser;
pub mod typer;
pub mod selector;
pub mod encoder;
pub mod exporter;

use std::path::Path;
use thiserror::Error;
use crate::ast::{exporter::Format, selector::{CallingConvention, CPU}, typer::TypeError};

type AbsPrg = Vec<Stmt>; pub enum Stmt { Ret(Expr) }
pub enum Expr { Con, Add, Sub, Mul, Div }

pub enum MachPrg { R5(Vec<R5Instr>), ARM(Vec<ARMInstr>), X86(Vec<X86Instr>) }
pub enum R5Instr { Add, AddI, Sub } pub enum ARMInstr {} pub enum X86Instr {}

#[derive(Error, Debug)] pub enum CompileError {
    #[error("type error")] TypeError(#[from] TypeError)
}

pub fn driver(src: &Path) -> Result<(), CompileError> {
    let forest = parser::parse(src);
    let _ = typer::typ()?;
    let instrs = selector::select(forest, CPU::R5, CallingConvention::SystemV);
    let encodings = encoder::encode(instrs);
    let _ = exporter::export(encodings, Format::Executable);
    Ok(())
}
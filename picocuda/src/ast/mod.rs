pub mod parser;
pub mod typer;
pub mod selector;
pub mod exporter;

use std::path::Path;
use thiserror::Error;
use crate::ast::{exporter::Format, selector::{CallingConvention, CPU}, typer::TypeError};

#[derive(Error, Debug)] pub enum CompileError {
    #[error("type error")] TypeError(#[from] TypeError)
}

pub fn compile(src: &Path) -> Result<(), CompileError> {
    let forest = parser::parse(src);
    let _ = typer::typ()?;
    let instrs = selector::select(forest, CPU::R5, CallingConvention::SystemV);
    let _ = exporter::export(instrs, Format::Executable); // TODO: io::Writer?
    Ok(())
}
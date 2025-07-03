use std::io;
use thiserror::Error;
use crate::ast::selector::MachPrg;

pub enum Format { Object, Executable, JIT }
#[derive(Error, Debug)] pub enum ExportError { #[error("IO Error")] IOError(#[from] io::Error), }
pub fn export(instrs: MachPrg, _f: Format) -> Result<(), ExportError> {
    todo!()
}
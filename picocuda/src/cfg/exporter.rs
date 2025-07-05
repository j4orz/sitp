use std::io::{self, Write};
use thiserror::Error;

pub enum Format { Object, Executable, JIT }
#[derive(Error, Debug)] pub enum ExportError { #[error("IO Error")] IOError(#[from] io::Error), }
pub fn export<W: Write>(machcode: Vec<u8>, _f: Format, dst: W) -> Result<(), ExportError> {
    todo!()
}
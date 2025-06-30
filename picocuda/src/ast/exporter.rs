use std::io;

use thiserror::Error;

pub enum Format { Object, Executable, JIT }
#[derive(Error, Debug)] pub enum ExportError { #[error("IO Error")] IOError(#[from] io::Error), }
pub fn export(_encodings: Vec<u8>, _f: Format) -> Result<(), ExportError> {
    todo!()
}
//!
//!                  __________________________________________________________________
//!                  |      sem                     opt                  gen          |
//!                  |   ____________             ________         ________________   |
//!                  |   |      ast |             | cfg--|-->cfg-->|\ isel/isched |   |             ____
//!     o            |   |type  / \ |             | /    |         | \ ra         |   |            ||""||
//!  _ /<. -->c0 u8--|-->|parse/   \|-->bril u8-->|/     |         |  \ enc--exp--|---|-->r5 elf-->||__||
//! (*)>(*)          |   -----------              --------         ----------------   |            [ -=.]`)
//!                  |   OLD front(1)            NEW mid(2)      UPDATED back(3)      |            ====== 0
//!                  -----------------------------------------------------------------|
//!
//!                                            PICOC
use std::path::PathBuf;
use std::{fs::File, io};
use thiserror::Error;

use crate::ast::{self, typer, CPU, MachPrg, R5OpCode, R5MachInstr, CallingConvention};
use crate::cfg;
pub mod parser;
pub mod selector;
pub mod allocator;
pub mod encoder;
pub mod exporter;

struct BB { entry: Instr, instrs: Vec<Instr>, exit: Instr } struct Instr {}

pub fn compile() -> Result<(), CompileError> {
    let (concrete_c0, elf_r5) = (File::open("hello.c")?, File::create("foo.txt")?);

    // (1) lift
    let ast = ast::parser::parse(concrete_c0);
    let _ = typer::typ()?;
    
    // (2) optimize
    let linear_bril = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vendor/bril/benchmarks/core/fact.bril");
    let cfg = cfg::parser::parse(&linear_bril)?;
    // local, regional, global, program opts...
    
    // (3) lower
    // let aasmtree = selector::select(threeac, CPU::R5, CallingConvention::SystemV);
    // let asmtree = allocator::allocate(aasmtree);
    // let machcode = encoder::encode(asmtree);
    // let elf = exporter::export(machcode, Format::Executable, dst_r5);

    // TODO: write elf to disk
    Ok(())
}

#[derive(Error, Debug)] pub enum CompileError {
    #[error("i/o error")] IOError(#[from] io::Error),
    #[error("type error")] TypeError(#[from] ast::typer::TypeError),
    #[error("parse error")] ParseError(#[from] cfg::parser::ParseError)
}
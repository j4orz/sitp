use std::path::PathBuf;
use std::{fs::File, io};
use thiserror::Error;
use crate::cfg::exporter::Format;

//          data(types) | algo(impls)
//          ------------|-----------
// OLD    =    same     |   same
// MOD    =    same     |   mod
// NEW    =    same     |   new

// /////////////////////////////////////////////////////////////////////////////
// 1. HIGH LEVEL BLOCK DIAGRAM
//                  __________________________________________________________________
//                  |      sem                     opt                  gen          |
//                  |   ____________             ________         ________________   |
//                  |   |      ast |             | cfg--|-->cfg-->|\ isel/isched |   |             ____
//     o            |   |type  / \ |             | /    |         | \ ra         |   |            ||""||
//  _ /<. -->c0 u8--|-->|parse/   \|-->bril u8-->|/     |         |  \ enc--exp--|---|-->r5 elf-->||__||
// (*)>(*)          |   -----------              --------         ----------------   |            [ -=.]`)
//                  |   OLD front(1)            NEW mid(2)      UPDATED back(3)      |            ====== 0
//                  -----------------------------------------------------------------|
//
//                                            PICOC
// /////////////////////////////////////////////////////////////////////////////


// /////////////////////////////////////////////////////////////////////////////
// 2. ALGORITHMS + DATA STRUCTURES
// ALGORITHMS
use crate::ast::{self, typer}; // OLD front(1)
pub mod parser; use crate::cfg::{self};                           // NEW mid(2)
pub mod selector; pub mod allocator; pub mod encoder; pub mod exporter; // MOD back(3)
// DATA STRUCTURES
use crate::ast::Ast; //  OLD SOURCE (C0)

struct BB { entry: Instr, instrs: Vec<Instr>, exit: Instr, } // NEW IR: CFG(BB)+SSA
struct Instr {}

use crate::ast::{CPU, MachPrg, R5OpCode, R5MachInstr, CallingConvention}; // OLD TARGET: {R5,ARM,x86}
// /////////////////////////////////////////////////////////////////////////////


// /////////////////////////////////////////////////////////////////////////////
// 3. ALGORITHMS + DATA STRUCTURES = PROGRAMS B)
pub fn compile() -> Result<(), CompileError> {
    let (concrete_c0, elf_r5) = (File::open("hello.c")?, File::create("foo.txt")?);

    // (1)
    let ast = ast::parser::parse(concrete_c0);
    let _ = typer::typ()?;
    
    // (2) 
    let linear_bril = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vendor/bril/benchmarks/core/fact.bril");
    let cfg = cfg::parser::parse(&linear_bril)?;
    // -local opts
    // -regional opts
    // -intraproc opts
    // -interproc opts

    
    // (3)
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
// /////////////////////////////////////////////////////////////////////////////
use std::path::PathBuf;
use std::{fs::File, io, path::Path};
use picoc::sema::{sema_parser, typer};
use picoc::opto::{self, opto_parser};
use picoc::cgen::{allocator, encoder, exporter::{self, Format}, selector, CC, CPU};

fn main() { println!("picocuda") }

pub fn compile_cfg(src: &Path) -> Result<(), CompileError> {
    let (src_c0, dst_r5) = (File::open("hello.c")?, File::create("foo.txt")?);
    let ast = sema_parser::parse_c02ast(src_c0);
    let _ = typer::typ()?;

    let opto = true;

    if opto {
        let linear = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vendor/bril/benchmarks/core/fact.bril");
        let cfg = opto_parser::parse_bril2cfg(&linear)?;
        // local, regional, global, program opts...
    }

    let aasmtree = selector::select(ast, CPU::R5, CC::SystemV);
    let asmtree = allocator::allocate(aasmtree);
    let machcode = encoder::encode(asmtree);
    let elf = exporter::export(machcode, Format::Executable, dst_r5);
    // TODO: write elf to disk
    Ok(())
}

fn compile_son() -> () {}

#[derive(thiserror::Error, Debug)] pub enum CompileError {
    #[error("i/o error")] IOError(#[from] io::Error),
    #[error("type error")] TypeError(#[from] typer::TypeError),
    #[error("parse error")] ParseError(#[from] opto::opto_parser::ParseError)
}
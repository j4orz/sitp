use std::path::PathBuf;
use std::{fs::File, io, path::Path};
use elements::graphs::AdjLinkedList;
use picoc::sema::{sema_parser, typer, Ast};
use picoc::opto::{self, cfg::opto_parser};
use picoc::cgen::{allocator, encoder, exporter::{self, Format}, selector, CC, CPU};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let foo = println!("picocuda");
    let bar = compile_cfg()?;
    Ok(())
}

enum OptoConfig { Ast, Cfg, CfgSsa, Son }
enum OptodPrg { Ast(Ast), Cfg(Cfg), CfgSsa(Cfg), Son }

pub fn compile_cfg() -> Result<(), CompileError> {
    // let (src_c0, dst_r5) = (File::open("hello.c")?, File::create("foo.txt")?);
    let ast = sema_parser::parse_c02ast(src_c0);
    let _ = typer::typ()?;

    let opto_config = OptoConfig::Ast;
    let optod_prg = match opto_config {
        Opto::None => OptodPrg::Ast(ast),
        Opto::Cfg => {
            let src_bril = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vendor/bril/test/interp/core/jmp.bril");
            let f_cfgs = opto_parser::parse_bril2cfg(&src_bril)?;
            // local opts
            // globla opts
        },
        Opto::CfgSsa => {
            let src_bril = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vendor/bril/test/interp/core/jmp.bril");
            let f_cfgs = opto_parser::parse_bril2cfg(&src_bril)?;
            // local opts
            // global opts
        },
        Opto::Son => todo!(),
    };

    let lowered_prg = match optod_prg {
        OptodPrg::Ast(stmts) => {
            // let aasmtree = selector::select(ast, CPU::R5, CC::SystemV);
            // let asmtree = allocator::allocate(aasmtree);
            // let machcode = encoder::encode(asmtree);
            // let elf = exporter::export(machcode, Format::Executable, dst_r5);
            // TODO: write elf to disk
        },
        OptodPrg::Cfg(_) => todo!(),
        OptodPrg::CfgSsa(_) => todo!(),
        OptodPrg::Son => todo!(),
    };
    Ok(())
}

fn compile_son() -> () {}

#[derive(thiserror::Error, Debug)] pub enum CompileError {
    #[error("i/o error")] IOError(#[from] io::Error),
    #[error("type error")] TypeError(#[from] typer::TypeError),
    #[error("parse error")] ParseError(#[from] opto::cfg::opto_parser::ParseError)
}
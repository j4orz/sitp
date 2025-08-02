use std::path::PathBuf;
use std::{fs::File, io, path::Path};
use elements::graphs::AdjLinkedList;
use picoc::opto::Cfg;
use picoc::sema::{sema_parser, typer, Ast};
use picoc::opto::{self, cfg::opto_parser};
use picoc::cgen::{allocator, encoder, exporter::{self, Format}, selector, CC, CPU};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let foo = println!("picocuda");
    let bar = compile()?;
    Ok(())
}

struct OptoConfig { ir: OptoIR, level: OptoLevel } enum OptoIR { Ast, Cfg, CfgSsa, Son } enum OptoLevel { O0, O1 }
impl OptoConfig { fn new(ir: OptoIR, level: OptoLevel) -> Self { Self { ir, level } } }

enum OptodPrg { Ast(Ast), Cfg(Cfg), CfgSsa(Cfg), Son }

pub fn compile() -> Result<(), CompileError> {
    // let (src_c0, dst_r5) = (File::open("hello.c")?, File::create("foo.txt")?);
    // let ast = sema_parser::parse_c02ast(src_c0);
    let _ = typer::typ()?;

    let opto_config = OptoConfig::new(OptoIR::Cfg, OptoLevel::O0);
    let optod_prg = match opto_config.ir {
        OptoIR::Ast => todo!(),
        OptoIR::Cfg => {
            let src_bril = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vendor/bril/test/interp/core/jmp.bril");
            let f_cfgs = opto_parser::parse_bril2cfg(&src_bril)?;
            match opto_config.level {
            OptoLevel::O0 => OptodPrg::Cfg(f_cfgs),
            OptoLevel::O1 => {
                // lvn
                // dataflow
                // dominators
                todo!()
            }}
        },
        OptoIR::CfgSsa => todo!(),
        OptoIR::Son => todo!(),
    };

    let lowered_prg = match optod_prg {
        OptodPrg::Ast(stmts) => {
            // let aasmtree = selector::select(ast, CPU::R5, CC::SystemV);
            // let asmtree = allocator::allocate(aasmtree);
            // let machcode = encoder::encode(asmtree);
            // let elf = exporter::export(machcode, Format::Executable, dst_r5);
            // TODO: write elf to disk
        },
        OptodPrg::Cfg(f_cfgs) => {
            todo!()
            // select
            // allocate
            // encode
            // export
        },
        OptodPrg::CfgSsa(_) => todo!(),
        OptodPrg::Son => todo!(),
    };
    Ok(())
}

#[derive(thiserror::Error, Debug)] pub enum CompileError {
    #[error("i/o error")] IOError(#[from] io::Error),
    #[error("type error")] TypeError(#[from] typer::TypeError),
    #[error("parse error")] ParseError(#[from] opto::cfg::opto_parser::ParseError)
}
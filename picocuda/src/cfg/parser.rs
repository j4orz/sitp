use std::{collections::HashMap, fs::File, io, path::Path, process::{Command, Stdio}};
use bril::{load_program_from_read, Code, EffectOps, Instruction};

// NB1: bril's rust tooling uses git as their package registry (no crates.io).
//      there are 2 rust implementations of bril2json (u8->json) upstream
//          1. https://github.com/sampsyo/bril/tree/main/bril2json-rs
//          2. https://github.com/sampsyo/bril/tree/main/bril-rs

//      if in the future access needs to be modified from binary crate to library crate,
//          (i.e accessing/modifying bril's semantic analysis: lexer/parser/typer/etc),
//          cargo git dependencies cannot be used because they both namespace under `bril2json`.
//      either
//          - fork and rename one of the crates to continue using git dependencies,
//          - or vendor with git (sparse) submodules using path dependencies.

//      unfortunately, as of Cargo 1.88 (2025-06-26),
//      "git and path cannot be used at the same time"
//      see: https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html#specifying-dependencies-from-git-repositories


// NB2: parser parses concrete .bril to in memory .json (u8 -> json) with
//      system installed `bril2json` (see: https://github.com/sampsyo/bril/tree/main/bril2json-rs).
//      the u8 -> json mapping is computed on an online as-needed basis rather than
//      an offline batch job to avoid out of sync issues with upstream .bril and
//      downstream .json artifacts see: https://github.com/sampsyo/bril/tree/main/benchmarks
pub fn parse(path: &Path) -> Result<bril::Program, ParseError> {
    let src_bril = File::open(path)?;
    let u82json = Command::new("bril2json").stdout(Stdio::piped()).stdin(src_bril).spawn()?;
    let linear = load_program_from_read(u82json.stdout.unwrap());
    let cfg = linear2cfg(linear);
    todo!()
    // Ok(cfg)
}

fn linear2cfg<'a>(threeac: bril::Program) -> impl Iterator<Item = Vec<Vec<Instruction>>> {
    threeac.functions.into_iter().map(|f| {
        let (mut bbs, bbnv, bb) =

        f.instrs.into_iter().fold((Vec::new(), HashMap::new(), Vec::new()),
     |(mut bbs, mut bbnv, mut bb), code| {
            match code {
            // NB1: on terminators bb is pushed by move (no flush) and reinitialized.
            // NB2: on labels an alias->usize mapping is created in the bbnv TODO: no Map, store facts in BB?
            Code::Label { label, .. } => { bbs.push(bb); bb = Vec::new(); bbnv.insert(label, bbs.len()); }
            Code::Instruction(instr) => { match instr {
                Instruction::Constant { .. } | Instruction::Value { .. } => bb.push(instr),
                Instruction::Effect { op, .. } => { match op {
                    EffectOps::Jump | EffectOps::Branch | EffectOps::Return => { bb.push(instr); bbs.push(bb); bb = Vec::new(); },
                    _ => bb.push(instr),
                }
            }}}}
            (bbs, bbnv, bb)
        });

        if !bb.is_empty() { bbs.push(bb) }
        bbs
    })
}

use thiserror::Error;
#[derive(Error, Debug)] pub enum ParseError {
    #[error("foobarbaz")] FooBarBaz,
    #[error("i/o error")] IOError(#[from] io::Error),
}
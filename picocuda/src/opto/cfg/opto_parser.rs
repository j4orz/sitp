use std::{fs::File, io, path::Path, process::{Command, Stdio}};
use itertools::{Itertools, put_back};
use elements::graphs as elements;
use bril::{Code, EffectOps, Instruction};

use crate::opto::cfg::BB;

// NB2: parser parses concrete .bril to in memory .json (u8 -> json) with
//      system installed `bril2json` (see: https://github.com/sampsyo/bril/tree/main/bril2json-rs).
//      the u8 -> json mapping is computed with an online as-needed basis rather than
//      an offline batch job to avoid out of sync issues with upstream .bril and
//      downstream .json artifacts see: https://github.com/sampsyo/bril/tree/main/benchmarks
pub fn parse_bril2cfg(path: &Path) -> Result<elements::AdjLinkedList<i32, i32, usize>, ParseError> {
    let src_bril = File::open(path)?;
    let u82json = Command::new("bril2json").stdout(Stdio::piped()).stdin(src_bril).spawn()?;
    let linear_prg = bril::load_program_from_read(u82json.stdout.unwrap());

    let foo =
    linear_prg.functions
    .into_iter()
    .map(|f| linf_to_bbs(f.instrs.into_iter()))
    .map(|f_bbs| {

    });
}

// linear2blocks chunks a linear stream of instrs into a linear stream of bbs.
// the chunk boundaries are non-trivial so .chunks(n) doesn't work. two options
// are .chunk_by()/.group_by() or .batching(). the former is used, but the
// combinator's api is .chunk_by(self, key: F) where F: FnMut(&Self::Item) -> K so
//      1. mutable state to track keys must be used and
//      2. allocation is necessary to avoid returning an iterator that references
//         a stack value since .chunk_by() and .group_by() only iterate by reference.
fn linf_to_bbs(function: impl Iterator<Item=bril::Code>) -> impl Iterator<Item=BB> {
    let (mut id, mut bb_ended) = (0usize, true);

    let bbs = function
    .chunk_by(move |code| {
        let bb_is_starting = matches!(code, Code::Label { .. }); // 1. label
        if bb_ended || bb_is_starting { id += 1; bb_ended = false; }; // 2. bump key
        bb_ended = if matches!(code, Code::Instruction(Instruction::Effect { op: EffectOps::Jump | EffectOps::Branch | EffectOps::Return, .. })) { true } else { false }; // 3. terminator
        id
    })
    .into_iter() // ChunkBy implements IntoIterator (it is *not* an iterator itself)
    .map(|(_, group)| BB::new(group.collect::<Vec<_>>()))
    .collect::<Vec<_>>();
    
    bbs.into_iter()
}

fn bbs2cfg(f_bbs: impl Iterator<Item=BB>) -> elements::AdjLinkedList<i32, i32, usize> {
    todo!()
}

use thiserror::Error;
#[derive(Error, Debug)] pub enum ParseError {
    #[error("foobarbaz")] FooBarBaz,
    #[error("i/o error")] IOError(#[from] io::Error),
}
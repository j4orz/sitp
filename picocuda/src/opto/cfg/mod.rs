use elements::graphs::{self as e, index::NodeIndex, AdjLinkedList, Graph};

use crate::opto::{Cfg, BB};

pub mod opto_parser;

//  CPU, MachPrg, R5OpCode, R5MachInstr, CallingConvention
// struct BB { entry: Instr, instrs: Vec<Instr>, exit: Instr } struct Instr {}
// impl BB {
//     fn new() -> Self { Self { entry: todo!(), instrs: todo!(), exit: todo!() } }
// }

// struct Cfg {
//     g:
// }

pub fn local_passes(input_prg: Cfg) -> Cfg {
    // let output_prg = input_prg
    // .into_iter()
    // .map(|f| {
    //     todo!()
    // })
    // .collect::<Vec<_>>();
    
    // output_prg
}

fn lvn(input: BB) -> BB {
    todo!()
}
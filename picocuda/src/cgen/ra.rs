use std::collections::HashSet;

use elements::graphs::AdjLinkedList;
use crate::{cgen::{r5, MachinePrg}, opto::{Prg, Fn}};

pub fn alloc(aabsasm: MachinePrg) -> MachinePrg {
    match aabsasm {
    MachinePrg::Ast(stmts) => todo!(),
    MachinePrg::Cfg(prg) => MachinePrg::Cfg(graph_color(prg)),
    MachinePrg::CfgSsa(prg) => todo!(),
    MachinePrg::Son => todo!(),
    }
}

pub fn graph_color(input: Prg<r5::Instr>) -> Prg<r5::Instr> {
    let foo = &input[0];
    let live_ranges = liveness_anal(foo);
    let interf_graph = construct_interf_graph(&live_ranges);
    // color the interference graph
    // check if every live range was colored
    todo!();
}

fn liveness_anal(f: &Fn<r5::Instr>) -> Vec<HashSet<usize>> {
    let live_ranges =
    f
    .node_references()
    .map(|(_, bb)| bb)
    .map(|bb| { // there should only be 1 bb for now (arithmetic, no control flow)
        bb.0
        .iter()
        .rev()
        .fold(HashSet::new(), |acc: HashSet<usize>, instr| {
            todo!()
            // LL_b(k) = [LL_a(k) - W(k)] ∪ R(k)
        })
    })
    .collect::<Vec<_>>();

    live_ranges
}

fn construct_interf_graph(input: &Vec<HashSet<usize>>) -> AdjLinkedList<i32, (), usize> {
    // vertices
    // edges
    todo!()
}
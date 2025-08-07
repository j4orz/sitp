
use crate::opto::OptodPrg;
use crate::sema::{Ast, Expr, Stmt};
use crate::cgen::{CC, HostMachine, MachinePrg, r5};

pub fn sel(prg: OptodPrg, cpu: HostMachine, _cc: CC) -> MachinePrg {
    match cpu {
    HostMachine::R5 => {
        match prg {
        OptodPrg::Ast(stmts) => todo!(),
        OptodPrg::Cfg(adj_linked_lists) => todo!(),
        OptodPrg::CfgSsa(adj_linked_lists) => todo!(),
        OptodPrg::Son => todo!(),
        }
    },
    HostMachine::ARM => unimplemented!(),
    HostMachine::X86 => unimplemented!(),
    }
    // match cpu {
    // CPU::R5 => MachPrg::R5(select_r5stmt(prg)),
    // CPU::ARM => unimplemented!(),
    // CPU::X86 => unimplemented!()
    // }

    todo!()
}

pub fn sel_r5stmt(prg: Ast) -> Vec<r5::Instr> {
    let mut aasm = vec![];

    for s in prg { match s {
        Stmt::Ret(e) => {
            let expr = sel_r5expr(e);
            let operands = Box::new([expr]);
            let ret = r5::Instr::new(r5::Op::Ret, operands);
            aasm.push(ret)
        },
    }}
    aasm
}

fn sel_r5expr(e: Expr) -> r5::Instr { match e {
    Expr::Con(c) => r5con(c),
    Expr::Add(_, _) => r5add(),
    Expr::Sub(_, _) => r5sub(),
    Expr::Mul(_, _) => r5mul(),
    Expr::Div(_, _) => r5div(),
}}

fn r5con(c: i128) -> r5::Instr {
    let operands = Box::new([]);
    if imm12(c) { r5::Instr::new(r5::Op::Int, operands) }
    else if imm20exact(c) { r5::Instr::new(r5::Op::Lui, operands) }
    else if imm32(c) { r5::Instr::new(r5::Op::AddI, operands) }
    else { r5::Instr::new(r5::Op::Int8, operands) }
}

// private Node addf(AddFNode addf) {
//     return new AddFRISC(addf);
// }

// private Node add(AddNode add) {
//     if( add.in(2) instanceof ConstantNode off2 && off2._con instanceof TypeInteger ti && imm12(ti) )
//         return new AddIRISC(add, (int)ti.value(),true);
//     return new AddRISC(add);
// }
fn r5add() -> r5::Instr { todo!() }
fn imm12(c: i128) -> bool { todo!() }
fn imm20exact(c: i128) -> bool { todo!() }
fn imm32(c: i128) -> bool { todo!() }

// private Node sub(SubNode sub) {
//     return sub.in(2) instanceof ConstantNode con && con._con instanceof TypeInteger ti && imm12(ti)
//         ? new AddIRISC(sub, (int)(-ti.value()),true)
//         : new SubRISC(sub);
// }

fn r5sub() -> r5::Instr { todo!() }

// case MulFNode    mulf -> new MulFRISC(mulf);
// case MulNode      mul -> new MulRISC(mul);
fn r5mul() -> r5::Instr { todo!() }

// case DivFNode    divf -> new DivFRISC(divf);
// case DivNode      div -> new DivRISC(div);
fn r5div() -> r5::Instr { todo!() }
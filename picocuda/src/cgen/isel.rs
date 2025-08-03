
use crate::opto::OptodPrg;
use crate::sema::{Ast, Expr, Stmt};
use crate::cgen::{MachPrg, R5MachInstr, R5OpCode, CC, HostMachine};

pub fn sel(prg: OptodPrg, cpu: HostMachine, _cc: CC) -> MachPrg {
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

pub fn sel_r5stmt(prg: Ast) -> Vec<R5MachInstr> {
    let mut aasm = vec![];

    for s in prg { match s {
        Stmt::Ret(e) => {
            let expr = sel_r5expr(e);
            let operands = Box::new([expr]);
            let ret = R5MachInstr::new(R5OpCode::Ret, operands);
            aasm.push(ret)
        },
    }}
    aasm
}

fn sel_r5expr(e: Expr) -> R5MachInstr { match e {
    Expr::Con(c) => r5con(c),
    Expr::Add(_, _) => r5add(),
    Expr::Sub(_, _) => r5sub(),
    Expr::Mul(_, _) => r5mul(),
    Expr::Div(_, _) => r5div(),
}}

fn r5con(c: i128) -> R5MachInstr {
    let operands = Box::new([]);
    if imm12(c) { R5MachInstr::new(R5OpCode::Int, operands) }
    else if imm20exact(c) { R5MachInstr::new(R5OpCode::Lui, operands) }
    else if imm32(c) { R5MachInstr::new(R5OpCode::AddI, operands) }
    else { R5MachInstr::new(R5OpCode::Int8, operands) }
}

// private Node addf(AddFNode addf) {
//     return new AddFRISC(addf);
// }

// private Node add(AddNode add) {
//     if( add.in(2) instanceof ConstantNode off2 && off2._con instanceof TypeInteger ti && imm12(ti) )
//         return new AddIRISC(add, (int)ti.value(),true);
//     return new AddRISC(add);
// }
fn r5add() -> R5MachInstr { todo!() }
fn imm12(c: i128) -> bool { todo!() }
fn imm20exact(c: i128) -> bool { todo!() }
fn imm32(c: i128) -> bool { todo!() }

// private Node sub(SubNode sub) {
//     return sub.in(2) instanceof ConstantNode con && con._con instanceof TypeInteger ti && imm12(ti)
//         ? new AddIRISC(sub, (int)(-ti.value()),true)
//         : new SubRISC(sub);
// }

fn r5sub() -> R5MachInstr { todo!() }

// case MulFNode    mulf -> new MulFRISC(mulf);
// case MulNode      mul -> new MulRISC(mul);
fn r5mul() -> R5MachInstr { todo!() }

// case DivFNode    divf -> new DivFRISC(divf);
// case DivNode      div -> new DivRISC(div);
fn r5div() -> R5MachInstr { todo!() }
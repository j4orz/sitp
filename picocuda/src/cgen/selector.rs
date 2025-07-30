
use crate::sema::{Ast, Expr, Stmt};
use crate::cgen::{MachPrg, R5MachInstr, R5OpCode, CC, CPU};


pub fn select(prg: Ast, cpu: CPU, _cc: CC) -> MachPrg { match cpu {
    CPU::R5 => MachPrg::R5(select_r5stmt(prg)),
    CPU::ARM => unimplemented!(),
    CPU::X86 => unimplemented!()
}}

pub fn select_r5stmt(prg: Ast) -> Vec<R5MachInstr> {
    let mut aasm = vec![];

    for s in prg { match s {
        Stmt::Ret(e) => {
            let expr = select_r5expr(e);
            let operands = Box::new([expr]);
            let ret = R5MachInstr::new(R5OpCode::Ret, operands);
            aasm.push(ret)
        },
    }}
    aasm
}

fn select_r5expr(e: Expr) -> R5MachInstr { match e {
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

fn r5add() -> R5MachInstr { todo!() }
fn r5sub() -> R5MachInstr { todo!() }
fn r5mul() -> R5MachInstr { todo!() }
fn r5div() -> R5MachInstr { todo!() }

fn imm12(c: i128) -> bool { todo!() }
fn imm20exact(c: i128) -> bool { todo!() }
fn imm32(c: i128) -> bool { todo!() }
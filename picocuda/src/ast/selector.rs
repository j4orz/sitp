use crate::ast::parser::{AbsPrg, Expr, Stmt};

pub enum CPU { R5, ARM, X86 }
pub enum CallingConvention { SystemV }

pub enum MachPrg { R5(Vec<R5Instr>), ARM(Vec<ARMInstr>), X86(Vec<X86Instr>) }
pub enum R5Instr {
    Int, Int8,
    Add, AddI, Sub, Lui, Auipc, // arithmetic
    Ret
}
impl R5Instr {
    fn encode(&self) -> Vec<u8> { todo!() }
}
pub enum ARMInstr {} pub enum X86Instr {}

pub fn select(prg: AbsPrg, cpu: CPU, _cc: CallingConvention) -> MachPrg { match cpu {
    CPU::R5 => MachPrg::R5(select_r5(prg)),
    CPU::ARM => MachPrg::ARM(select_arm(prg)),
    CPU::X86 => MachPrg::X86(select_x86(prg)),
}}

pub fn select_r5(prg: AbsPrg) -> Vec<R5Instr> {
    let mut instrs = vec![];
    for s in prg { match s {
        Stmt::Ret(e) => {
            instrs.push(R5Instr::Ret);
            instrs.push(select_expr(e))
        },
    }}
    instrs
}

fn select_expr(e: Expr) -> R5Instr { match e {
    Expr::Con(c) => con(c),
    Expr::Add => add(),
    Expr::Sub => sub(),
    Expr::Mul => mul(),
    Expr::Div => div(),
}}

fn con(c: i128) -> R5Instr {
    if imm12(c) { R5Instr::Int }
    else if imm20exact(c) { R5Instr::Lui }
    else if imm32(c) { R5Instr::AddI }
    else { R5Instr::Int8 }
}

fn add() -> R5Instr { todo!() }
fn sub() -> R5Instr { todo!() }
fn mul() -> R5Instr { todo!() }
fn div() -> R5Instr { todo!() }

fn imm12(c: i128) -> bool { todo!() }
fn imm20exact(c: i128) -> bool { todo!() }
fn imm32(c: i128) -> bool { todo!() }

pub fn select_arm(prg: AbsPrg) -> Vec<ARMInstr> { todo!() }
pub fn select_x86(prg: AbsPrg) -> Vec<X86Instr> { todo!() }
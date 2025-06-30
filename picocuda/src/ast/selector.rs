use crate::ast::{AbsPrg, MachPrg};

pub enum CPU { R5, ARM, X86 }
pub enum CallingConvention { SystemV }

pub fn select(_prg: AbsPrg, cpu: CPU, _cc: CallingConvention) -> MachPrg {
    match cpu {
        CPU::R5 => MachPrg::R5(vec![]),
        CPU::ARM => MachPrg::ARM(vec![]),
        CPU::X86 => MachPrg::X86(vec![]),
    }
}

// fn _select_stmt() -> () {

// }

// fn _select_expr(e: Expr) -> () {
//     match e {
//         Expr::Con => todo!(),
//         Expr::Add => todo!(),
//         Expr::Sub => todo!(),
//         Expr::Mul => todo!(),
//         Expr::Div => todo!(),
//     }
// }
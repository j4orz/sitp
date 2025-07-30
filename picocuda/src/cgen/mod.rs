pub mod allocator;
pub mod selector;
pub mod encoder;
pub mod exporter;

pub enum CPU { R5, ARM, X86 } pub enum CallingConvention { SystemV }
pub enum MachPrg { R5(Vec<R5MachInstr>), ARM(Vec<ARMInstr>), X86(Vec<X86Instr>) }
// NB: parallelizing the compiler requires moving the static mutable counter into TLS
//     also skipping 0 since node ids show up in bit vectors, work lists, etc.
static mut VREG_COUNTER: u32 = 0;
pub fn generate_vreg() -> u32 { unsafe { VREG_COUNTER += 1; VREG_COUNTER } }

pub enum R5OpCode { // TARGET R5
    Int, Int8, Add, AddI, Sub, Lui, Auipc, // arithmetic 
    Ret
}
pub struct R5MachInstr {
    // NB: machine instruction maintains retains semantic facts discovered/generated
    //     so 1. use->def facts (operands) and 2. registers (vreg/phyreg)
    opcode: R5OpCode, operands: Box<[R5MachInstr]>, // Box<[]> keeps children operands fixed arity
    vreg: u32, phyreg: Option<u32>,
}
impl R5MachInstr {
    pub fn new(opcode: R5OpCode, operands: Box<[R5MachInstr]>) -> Self {
        Self { opcode, operands, vreg: generate_vreg(), phyreg: None, }
    }
}

pub enum ARMInstr {} // TARGET ARM

pub enum X86Instr {} // TARGET x86
pub mod opto_parser;

//  CPU, MachPrg, R5OpCode, R5MachInstr, CallingConvention
struct BB { entry: Instr, instrs: Vec<Instr>, exit: Instr } struct Instr {}
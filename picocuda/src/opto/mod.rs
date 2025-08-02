use elements::graphs::AdjLinkedList;

pub mod cfg;
pub mod cfgssa;
pub mod son;

pub type Cfg = Vec<AdjLinkedList<BB, (), usize>>;
#[derive(Debug)] pub struct BB(Vec<bril::Code>);
impl BB { fn new(instrs: Vec<bril::Code>) -> Self { Self(instrs) } }
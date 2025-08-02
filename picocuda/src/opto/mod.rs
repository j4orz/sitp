use elements::graphs::AdjLinkedList;

pub mod cfg;
pub mod cfgssa;
pub mod son;

type Cfg = AdjLinkedList<BB, (), usize>;
#[derive(Debug)] pub struct BB(Vec<bril::Code>);
impl BB { fn new(instrs: Vec<bril::Code>) -> Self { Self(instrs) } }
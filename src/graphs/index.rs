use std::{hash::Hash, fmt::Debug};

pub trait Index: Copy + Default + Hash + Ord + Debug + 'static {
    fn new(i: usize) -> Self;
    fn index(&self) -> usize;
    fn max() -> Self;
}

// impl Index for u8 {
//     fn new(i: usize) -> Self { todo!() }
//     fn index(&self) -> usize { todo!() }
//     fn max() -> Self { todo!() }
// }

// impl Index for usize {
//     fn new(i: usize) -> Self { todo!() }
//     fn index(&self) -> usize { todo!() }
//     fn max() -> Self { usize::MAX }
// }

// impl Index for u16 {
//     fn new(i: usize) -> Self { todo!() }
//     fn index(&self) -> usize { todo!() }
//     fn max() -> Self { u16::MAX }
// }

// impl Index for u32 {
//     fn new(i: usize) -> Self { todo!() }
//     fn index(&self) -> usize { todo!() }
//     fn max() -> Self { u32::MAX }
// }

// impl Index for u64 {
//     fn new(i: usize) -> Self { todo!() }
//     fn index(&self) -> usize { todo!() }
//     fn max() -> Self { u64::MAX }
// }

// impl Index for u128 {
//     fn new(i: usize) -> Self { todo!() }
//     fn index(&self) -> usize { todo!() }
//     fn max() -> Self { u128::MAX }
// }

#[derive(Clone, Copy, PartialEq)] pub struct NodeIndex<Idx>(Idx);
#[derive(Clone, Copy, PartialEq)] pub struct EdgeIndex<Idx>(Idx);
// size, alignment of Option<usize> is ... NPO...
// TODO: Nethercote. measure.
// pub const INVALID_EDGE_INDEX: EdgeIndex = EdgeIndex(usize::MAX);
// const OUTGOING: usize = 0;
// const INCOMING: usize = 1;
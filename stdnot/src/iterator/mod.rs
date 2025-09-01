mod adapters;
mod sources;

use std::{mem};
use crate::iterator::adapters::Map;

// list<number> ---------------->sort/search

// - atomic data/function ----------------> u32, i32, fp32,
//      - recursive data/function --------> Nat
//      - recursive data/function --------> List(Nat)
//      - egyptian multiplication
//      - egyptian division



// - compound data/function: (structs/enums) (Rat)
    // struct ----> (Rat) List(Rat)
    // enum -------> Number Nat or Rat List(Number)
// - generic data/function: (GP, HOF) (Iterator of T)
    // HOF --------> map, filter, fold. lz_map, lz_filter, lz_fold, par_map, par_filter, par_fold
    // GP ---------->Iterator<T>
    // data structures -------> (Stacks/queues) & algos (sort, search): List -> Vec -> VecDeque -> BinaryHeap

pub enum List<T> {
    None,
    Some(Box<Node<T>>)
}

pub struct Node<T> {
    first: T, rest: List<T>,
}

impl<T> List<T> {
    pub fn new() -> Self { List::None }
    pub fn push(&mut self, e: T) {
        // let n = Box::new(Node { first: e, rest: self.tail.take() });
        // self.tail = Link::Some(n);

        let new_node = Box::new(Node {
            first: e,
            rest: mem::replace(self, List::None),
        });

        *self = List::Some(new_node);
    }

    pub fn map<F, U>(l: &List<T>, mut f: F) -> List<U>
    where
        F: FnMut(&T) -> U
    {
        match l {
        List::None => List::None,
        List::Some(first) => {
            List::Some(Box::new(Node { first: f(&first.first), rest: Self::map(&first.rest, f) }))
        }}
    }

    pub fn lz_map() -> () {}

    // NB: return type of .pop() is values of Option<T> rather than references of Box<Node<T>>,
    pub fn pop(&mut self) -> Option<T> {
        match mem::replace(self, List::None) {
        List::None => None,
        List::Some(n) => {
            *self = n.rest;
            Some(n.first)
        }}

                // match &self {
        //     List::Some(boxed_n) => {
        //         *self = boxed_n.rest;
        //         List::Some(boxed_n.e)
        //     },
        //     List::None => todo!(),
        // }

        // .take().map() is sugar for indiana jonesing with match mem::replace(&mut dst, None) { .. }
        // which upgrades permissions of self from R -> O
        // self.take().map(|boxed_n| {
        //     self = boxed_n.next; // self.head = node.next (partial move)
        //     boxed_n.e
        // })

    }
}



// pub fn into_iter(self) -> IntoIter<T> { IntoIter(self) }
// pub fn iter(&self) -> Iter<T> { Iter { next: self.tail.as_deref().map(|boxed_n| &*boxed_n) }}

// impl<T> Drop for LinkedList<T> {
//     fn drop(&mut self) {
//         let mut cur_link = mem::replace(&mut self.tail, List::None);
//         while let List::Some(mut boxed_n) = cur_link {
//             cur_link = mem::replace(&mut boxed_n.rest, List::None)
//         }
//     }
// }















// mod test {
//     use super::*;

//     #[test]
//     fn basics() {
//         let mut list = LinkedList::new();

//         assert_eq!(list.pop(), None); // check empty
//         list.push(1); list.push(2); list.push(3); // populate
//         assert_eq!(list.pop(), Some(3)); assert_eq!(list.pop(), Some(2)); // removal
//         list.push(4); list.push(5); // populate some more
//         assert_eq!(list.pop(), Some(5)); assert_eq!(list.pop(), Some(4)); // remove some more
//         assert_eq!(list.pop(), Some(1)); assert_eq!(list.pop(), None); // check exhaustion
//     }

//     #[test]
//     fn into_iter() {
//         let mut list = LinkedList::new();
//         list.push(1); list.push(2); list.push(3);

//         let mut iter = list.into_iter();
//         assert_eq!(iter.next(), Some(3)); assert_eq!(iter.next(), Some(2)); assert_eq!(iter.next(), Some(1));
//         assert_eq!(iter.next(), None);
//     }

//     #[test]
//     fn iter() {
//         let mut list = LinkedList::new();
//         list.push(1); list.push(2); list.push(3);

//         let mut iter = list.iter();
//         assert_eq!(iter.next(), Some(&3));
//         assert_eq!(iter.next(), Some(&2));
//         assert_eq!(iter.next(), Some(&1));
//     }
// }

pub struct Vec {}
pub struct BinaryHeap {}
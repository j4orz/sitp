//! .......................
mod adapters;
mod sources;

use std::{mem};
use crate::seqs::adapters::Map;


trait Iterator {
    type Item;

    fn next(&mut self) -> Option<Self::Item>;

    // adapters
    fn array_chunks(self) -> ()
    where Self: Sized
    {}

    fn by_ref_sized() -> ()
    where Self: Sized
    {}

    fn chain() -> ()
    where Self: Sized
    {}

    fn cloned() -> ()
    where Self: Sized
    {}

    fn copied() -> ()
    where Self: Sized
    {}

    fn cycle() -> ()
    where Self: Sized
    {}

    fn enumerate() -> ()
    where Self: Sized
    {}

    fn filter_map() -> ()
    where Self: Sized
    {}

    fn flatten() -> ()
    where Self: Sized
    {}

    fn fuse() -> ()
    where Self: Sized
    {}

    fn inspect() -> ()
    where Self: Sized
    {}

    fn intersperse() -> ()
    where Self: Sized
    {}

    fn map<F, B>(self, f: F) -> Map<Self, F>
    where
        Self: Sized,
        F: FnMut(Self::Item) -> B
    { Map { i: self, f } }
    
    fn map_while() -> ()
    where Self: Sized
    {}

    fn map_windows() -> ()
    where Self: Sized
    {}

    fn peekable() -> ()
    where Self: Sized
    {}

    fn rev() -> ()
    where Self: Sized
    {}

    fn scan() -> ()
    where Self: Sized
    {}

    fn skip() -> ()
    where Self: Sized
    {}

    fn skip_while() -> ()
    where Self: Sized
    {}

    fn step_by() -> ()
    where Self: Sized
    {}

    fn take() -> ()
    where Self: Sized
    {}

    fn take_while() -> ()
    where Self: Sized
    {}

    fn zip() -> ()
    where Self: Sized
    {}

    // _________________________________________________________________________

    // adapters
    fn interleave() -> ()
    where Self: Sized
    {}
    fn cartesian_product() -> ()
    where Self: Sized
    {}

    fn multi_cartesian_product() -> ()
    where Self: Sized
    {}

    fn coalesce() -> ()
    where Self: Sized
    {}

    fn permutations() -> ()
    where Self: Sized
    {}

    fn combinations() -> ()
    where Self: Sized
    {}

    fn powerset() -> ()
    where Self: Sized
    {}

    fn dedup() -> ()
    where Self: Sized
    {}

    fn tee() -> ()
    where Self: Sized
    {}

    fn tuples() -> ()
    where Self: Sized
    {}

    fn tuple_windows() -> ()
    where Self: Sized
    {}

    fn circular_tuple_windows() -> ()
    where Self: Sized
    {}

    fn merge() -> ()
    where Self: Sized
    {}

    fn kmerge() -> ()
    where Self: Sized
    {}

    fn unique() -> ()
    where Self: Sized
    {}

}

pub struct IntoIter<T>(LinkedList<T>);
impl<T> Iterator for IntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> { self.0.pop() }
}

pub struct Iter<'a, T> {
    next: Option<&'a Node<T>>
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.next.map(|n| {
            self.next = n.next.as_deref().map(|boxed_n| &*boxed_n);
            &n.e
        })
    }
}







pub struct LinkedList<T> { tail: Link<T> }
type Link<T> = Option<Box<Node<T>>>;
struct Node<T> { e: T, next: Link<T> }

impl<T> LinkedList<T> {
    pub fn new() -> Self { Self { tail: Link::None } }
    pub fn push(&mut self, val: T) {
        let n = Box::new(Node { e: val, next: self.tail.take() });
        self.tail = Link::Some(n);
    }

    // NB: return type of .pop() is values of Option<T> rather than references of Box<Node<T>>,
    pub fn pop(&mut self) -> Option<T> {
        // match &self.tail {
        //     Some(boxed_n) => {
        //         self.tail = boxed_n.next;
        //         Some(boxed_n.e)
        //     },
        //     None => todo!(),
        // }

        // match mem::replace(&mut self.tail, None) {
        //     Some(boxed_n) => {
        //         self.tail = boxed_n.next;
        //         Some(boxed_n.e)
        //     },
        //     None => todo!(),
        // }

        // .take().map() is sugar for indiana jonesing with match mem::replace(&mut dst, None) { .. }
        // which upgrades permissions of self.tail from R -> O
        self.tail.take().map(|boxed_n| {
            self.tail = boxed_n.next; // self.head = node.next (partial move)
            boxed_n.e
        })
    }

    pub fn into_iter(self) -> IntoIter<T> { IntoIter(self) }
    pub fn iter(&self) -> Iter<T> { Iter { next: self.tail.as_deref().map(|boxed_n| &*boxed_n) }}
}

impl<T> Drop for LinkedList<T> {
    fn drop(&mut self) {
        let mut cur_link = mem::replace(&mut self.tail, Link::None);
        while let Link::Some(mut boxed_n) = cur_link {
            cur_link = mem::replace(&mut boxed_n.next, Link::None)
        }
    }
}















mod test {
    use super::*;

    #[test]
    fn basics() {
        let mut list = LinkedList::new();

        assert_eq!(list.pop(), None); // check empty
        list.push(1); list.push(2); list.push(3); // populate
        assert_eq!(list.pop(), Some(3)); assert_eq!(list.pop(), Some(2)); // removal
        list.push(4); list.push(5); // populate some more
        assert_eq!(list.pop(), Some(5)); assert_eq!(list.pop(), Some(4)); // remove some more
        assert_eq!(list.pop(), Some(1)); assert_eq!(list.pop(), None); // check exhaustion
    }

    #[test]
    fn into_iter() {
        let mut list = LinkedList::new();
        list.push(1); list.push(2); list.push(3);

        let mut iter = list.into_iter();
        assert_eq!(iter.next(), Some(3)); assert_eq!(iter.next(), Some(2)); assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn iter() {
        let mut list = LinkedList::new();
        list.push(1); list.push(2); list.push(3);

        let mut iter = list.iter();
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&1));
    }
}

pub struct Vec {}
pub struct BinaryHeap {}
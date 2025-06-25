use std::{fs::read, path::Path};

pub fn read_chars(path: &Path) -> Vec<char> {
    read(path).expect("file dne").iter().map(|b| *b as char).collect::<Vec<_>>()
}
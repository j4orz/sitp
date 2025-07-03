use std::path::Path;

pub type AbsPrg = Vec<Stmt>; pub enum Stmt { Ret(Expr) }
pub enum Expr { Con(i128), Add, Sub, Mul, Div }

pub fn parse(src: &Path) -> AbsPrg {
    todo!()
}
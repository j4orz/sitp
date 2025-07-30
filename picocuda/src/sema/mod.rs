pub mod parser;
pub mod typer;
pub mod selector;

pub type Ast = Vec<Stmt>; pub enum Stmt { Ret(Expr) }
pub enum Expr {
    Con(i128),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>)
}
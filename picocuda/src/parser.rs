
use std::io;
use std::iter;

use crate::{
    InstrNode, //rep::{ctl::{Branch, Return, Start}, data::{Add, Div, Int, Mul, Sub}, scope::{Scope, ScopeError, ARG, CTRL}, MultiInstr, Proj, TypeAndVal}
};
use thiserror::Error;

// NB1. each function in the parser will parse in two ways
//        a. conditionally (SUM/OR): match tokens(first, rest) first.typ { TT::Foo => {}, TT::Bar => {}, TT::Baz => {} }
//        b. assertively (PROD/AND): require(tokens, TT:Foo), eat(tokens, TT:Bar), eat(tokens, TT:Baz)

// NB2. the parser is composed of pure functions less the start instruction and
//      scope/nv instruction so the token stream (r) is threaded throughout

// NB3. generally speaking there are three variants for intermediate representation
//        a. tree ("AST")
//           -> precedence is represented via tree's hierarchy.
//        b. two-tiered (nested) graph of basic blocks of instructions. ("CFG+BB")
//           -> edges denote ONLY control flow
//        c. single-tiered (flat) graph of instructions ("SoN")
//           -> edges denote control flow OR data flow

//      picoc parses the concrete syntax into (c) because TODO. see optimizer
//      for more details. this means that the total ordering of straightline
//      code (vec<list>) is relaxed to a partial order of a graph

#[derive(Error, Debug)]
pub enum ParseError {
    // #[error("scope")]
    // Scope(#[from] ScopeError),
    #[error("(expected {expected:?}, found {actual:?})")]
    Mismatch { expected: String, actual: String },
}

pub struct Parser {
    // pub start: InstrNode,
} // scope: Rc<Scope> }
impl Parser {
    pub fn new() -> Self {
        Self {
            // start: InstrNode::Start,
        }
    } // scope: Rc::new(Scope::new()) }}

    pub fn parse_prg(&mut self, tokens: &[Token]) -> Result<InstrNode, ParseError> {
        let r = tokens;
        let (_, r) = require(r, TT::KeywordInt)?;
        let (_, r) = require(r, TT::Alias)?;
        let (_, r) = require(r, TT::PuncLeftParen)?;
        let (_, r) = require(r, TT::PuncRightParen)?;

        let (_, r) = require(r, TT::PuncLeftBrace)?;
        // self.scope.push_nv();
        // self.scope.write(CTRL.to_owned(), Proj::new(self.start.clone(), 0));
        // self.scope.write(ARG.to_owned(), Proj::new(self.start.clone(), 1));
        let (block, r) = self.parse_block(r)?;
        // self.scope.pop_nv();

        let (_, r) = require(r, TT::PuncRightBrace)?;
        // try_convert block's Rc<dyn Instr> -> Rc<Return>

        if r.is_empty() {
            Ok(block)
        } else {
            Err(ParseError::Mismatch {
                expected: "empty token stream".to_string(),
                actual: format!("{:?}", r),
            })
        }
    }

    // NB: lexical scope ==> nv's are only pushed/popped in parse_block
    fn parse_block<'a>(
        &mut self,
        tokens: &'a [Token],
    ) -> Result<(InstrNode, &'a [Token]), ParseError> {
        // self.scope.push_nv();
        let (mut output, mut r) = (None, tokens);
        while let Ok((stmt, _r)) = self.parse_stmt(r) {
            output = Some(stmt);
            r = _r;
        }
        // self.scope.push_nv();
        Ok((output.unwrap(), r))
    }

    fn parse_stmt<'a>(
        &mut self,
        tokens: &'a [Token],
    ) -> Result<(InstrNode, &'a [Token]), ParseError> {
        match tokens {
            [] => Err(ParseError::Mismatch {
                expected: "expected: {:?} got an empty token stream".to_string(),
                actual: "".to_string(),
            }),
            [f, r @ ..] => match f.typ {
                TT::KeywordInt => {
                    let (_alias, r) = require(r, TT::Alias)?;
                    let (_, r) = require(r, TT::Equals)?;
                    let (expr, r) = self.parse_expr(r)?;
                    let (_, r) = require(r, TT::PuncSemiColon)?;

                    // let _ = self.scope.write(alias.lexeme.to_owned(), expr.clone())?;
                    Ok((expr, r))
                }
                // TT::KeywordIf => {
                //     let (pred, r) = self.parse_expr(r)?;

                //     let branch = Branch::new(self.scope.read_ctrl(), pred);
                //     let left = Proj::new(branch.clone(), 0).peephole(self.start.clone());
                //     let right = Proj::new(branch, 1).peephole(self.start.clone());
                //     let scope_og = Rc::new((*self.scope).clone()); // TODO: need ascii debugger here to verify

                //     // NB: because condtionals are statements and not expressions
                //     //     in C, the return of parse_stmts are not bound and ignored

                //     self.scope.write_ctrl(left); // 1. set ctrl
                //     let (_, r) = self.parse_stmt(r)?; // 2. parse
                //     let scope_left = Rc::new((*self.scope).clone()); // 3. alias scope

                //     self.scope = scope_og; // reset

                //     self.scope.write_ctrl(right); // 1. set ctrl
                //     if r.len() > 1 && r[0].typ == TT::KeywordEls { let (_, r) = self.parse_stmt(r)?; }; // 2. parse
                //     let scope_right = Rc::new((*self.scope).clone()); // 3. alias scope

                //     let region = scope_left.merge(&scope_right);
                //     self.scope.write_ctrl(region.clone());
                //     Ok((region, r))
                // },
                TT::KeywordRet => {
                    let (expr, r) = self.parse_expr(r)?;
                    let (_, r) = require(r, TT::PuncSemiColon)?;
                    // let retinstr = Return::new(self.start.clone(), expr);
                    let retinstr = InstrNode::Ret(Box::new(InstrNode::Start), Box::new(expr));
                    Ok((retinstr, r))
                }
                t => Err(ParseError::Mismatch {
                    expected: format!("expected: {:?} got: {:?}", TT::KeywordRet, t),
                    actual: f.lexeme.to_owned(),
                }),
            },
        }
    }

    fn parse_expr<'a>(&self, tokens: &'a [Token]) -> Result<(InstrNode, &'a [Token]), ParseError> {
        self.parse_term(tokens)
    }

    fn parse_term<'a>(&self, tokens: &'a [Token]) -> Result<(InstrNode, &'a [Token]), ParseError> {
        let (x, r) = self.parse_factor(tokens)?;

        match r {
            [] => panic!(),
            [f, _r @ ..] => match f.typ {
                TT::Plus => {
                    let (y, r) = self.parse_factor(_r)?;
                    // Ok((Add::new(x, y).peephole(self.start.clone()), r))
                    Ok((InstrNode::Add(Box::new(x), Box::new(y)), r))
                }
                TT::Minus => {
                    let (y, r) = self.parse_factor(_r)?;
                    // Ok((Sub::new(x, y), r))
                    Ok((InstrNode::Sub(Box::new(x), Box::new(y)), r))
                }
                _ => Ok((x, r)),
            },
        }
    }

    fn parse_factor<'a>(
        &self,
        tokens: &'a [Token],
    ) -> Result<(InstrNode, &'a [Token]), ParseError> {
        let (x, r) = self.parse_atom(tokens)?;

        match r {
            [] => panic!(),
            [f, _r @ ..] => match f.typ {
                TT::Star => {
                    let (y, r) = self.parse_atom(_r)?;
                    // Ok((Mul::new(x, y), r))
                    Ok((InstrNode::Mul(Box::new(x), Box::new(y)), r))
                }
                TT::Slash => {
                    let (y, r) = self.parse_atom(_r)?;
                    // Ok((Div::new(x, y), r))
                    Ok((InstrNode::Div(Box::new(x), Box::new(y)), r))
                }
                _ => Ok((x, r)),
            },
        }
    }

    fn parse_atom<'a>(&self, tokens: &'a [Token]) -> Result<(InstrNode, &'a [Token]), ParseError> {
        match tokens {
            [] => Err(ParseError::Mismatch {
                expected: "expected: {:?} got an empty token stream".to_string(),
                actual: "".to_string(),
            }),
            [f, r @ ..] => match f.typ {
                TT::LiteralInt => {
                    // let constantinstr = Int::new(
                    //     self.start.clone(),
                    //     TypeAndVal::Int(f.lexeme.parse().unwrap()),
                    // );

                    // Ok((constantinstr, r))
                    Ok((InstrNode::Lit(f.lexeme.parse().unwrap()), r))
                }
                // TT::Alias => {
                //     let expr = self.scope.read(f.lexeme.to_owned())?;
                //     Ok((expr,r))
                // },
                t => Err(ParseError::Mismatch {
                    expected: format!("expected: {:?} got: {:?}", TT::LiteralInt, t),
                    actual: f.lexeme.to_owned(),
                }),
            },
        }
    }
}

fn require(tokens: &[Token], tt: TT) -> Result<(&Token, &[Token]), ParseError> {
    match tokens {
        [] => Err(ParseError::Mismatch {
            expected: format!("expected: {:?} got: {:?}", tt, tokens),
            actual: "".to_string(),
        }),
        [f, r @ ..] => {
            if f.typ == tt {
                Ok((f, r))
            } else {
                Err(ParseError::Mismatch {
                    expected: format!("expected: {:?} got: {:?}", tt, f),
                    actual: f.lexeme.to_owned(),
                })
            }
        }
    }
}

#[cfg(test)]
mod parse_arith {
    use crate::{parser::{lex, Parser}, InstrNode};
    use std::{assert_matches::assert_matches, fs};
    
    const TEST_DIR: &str = "tests/arith";

    #[test]
    fn lit() {
        let chars = fs::read(format!("{TEST_DIR}/lit.c"))
            .expect("file dne")
            .iter()
            .map(|b| *b as char)
            .collect::<Vec<_>>();
    
        let tokens = lex(&chars).unwrap();
        // let mut parser = super::Parser::new(Start::new(vec![Box::new(TypeAndVal::Bot)]));
        let mut parser = Parser::new();
        let graph = parser.parse_prg(&tokens).unwrap();
        assert_matches!(graph, InstrNode::Ret(ref boxed, _) if matches!(**boxed, InstrNode::Start));
        insta::assert_debug_snapshot!(graph, @r###"
        Ret(
            Start,
            Lit(
                8,
            ),
        )
        "###);
    }
}

// todo. remove allocations.
// iterate on slices and iterators.
// todo: change to iterative
#[rustfmt::skip]
#[derive(Clone, PartialEq, Debug)]
pub struct Token { pub lexeme: String, pub typ: TT }

#[rustfmt::skip]
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum TT {
    LiteralInt, Alias, // introductions (values) RE: [0-9]+ and [a-zA-Z][a-zA-Z0-9]*
    KeywordInt, KeywordChar, KeywordVoid, KeywordRet, KeywordIf, KeywordEls, KeywordFor, KeywordWhile, KeywordTrue, KeywordFalse, // keywords ⊂ identifiers
    Plus, Minus, Star, Slash, LeftAngleBracket, RightAngleBracket, Equals, Bang, Amp, Bar, // eliminations (ops)
    PuncLeftParen, PuncRightParen, PuncLeftBrace, PuncRightBrace, PuncSemiColon, PuncComma,// punctuation
}

//  1. variations are explicitly typed. Collapsing categories like keywords
//     into one variant will lose information since lexeme : String, which
//     will produce redundant work for the parser during syntactic analysis
//  2. non-tokens: comments, preprocessor directives, macros, whitespace

pub fn lex(input: &[char]) -> Result<Vec<Token>, io::Error> {
    let cs = skip_ws(input);

    // literals and identifiers have arbitrary length
    // operations and punctuations are single ASCII characters
    match cs {
        [] => Ok(vec![]),
        [f, r @ ..] => match f {
            '0'..='9' => scan_int(cs),
            'a'..='z' | 'A'..='Z' => scan_id(cs),
            '+' => {
                #[rustfmt::skip]
                let t = Token { lexeme: String::from("+"), typ: TT::Plus };
                Ok(iter::once(t).chain(lex(r)?).collect())
            }
            '-' => {
                #[rustfmt::skip]
                let t = Token { lexeme: String::from("-"), typ: TT::Minus };
                Ok(iter::once(t).chain(lex(r)?).collect())
            }
            '*' => {
                #[rustfmt::skip]
                let t = Token { lexeme: String::from("*"), typ: TT::Star };
                Ok(iter::once(t).chain(lex(r)?).collect())
            }
            '/' => {
                #[rustfmt::skip]
                let t = Token { lexeme: String::from("/"), typ: TT::Slash };
                Ok(iter::once(t).chain(lex(r)?).collect())
            }
            '<' => {
                #[rustfmt::skip]
                let t = Token { lexeme: String::from("<"), typ: TT::LeftAngleBracket };
                Ok(iter::once(t).chain(lex(r)?).collect())
            }
            '>' => {
                #[rustfmt::skip]
                let t = Token { lexeme: String::from(">"), typ: TT::RightAngleBracket };
                Ok(iter::once(t).chain(lex(r)?).collect())
            }
            '=' => {
                #[rustfmt::skip]
                let t = Token { lexeme: String::from("="), typ: TT::Equals };
                Ok(iter::once(t).chain(lex(r)?).collect())
            }
            '!' => {
                #[rustfmt::skip]
                let t = Token { lexeme: String::from("!"), typ: TT::Bang };
                Ok(iter::once(t).chain(lex(r)?).collect())
            }
            '&' => {
                #[rustfmt::skip]
                let t = Token { lexeme: String::from("&"), typ: TT::Amp };
                Ok(iter::once(t).chain(lex(r)?).collect())
            }
            '|' => {
                #[rustfmt::skip]
                let t = Token { lexeme: String::from("|"), typ: TT::Bar };
                Ok(iter::once(t).chain(lex(r)?).collect())
            }
            '(' => {
                #[rustfmt::skip]
                let t = Token { lexeme: String::from("("), typ: TT::PuncLeftParen };
                Ok(iter::once(t).chain(lex(r)?).collect())
            }
            ')' => {
                #[rustfmt::skip]
                let t = Token { lexeme: String::from(")"), typ: TT::PuncRightParen };
                Ok(iter::once(t).chain(lex(r)?).collect())
            }
            '{' => {
                #[rustfmt::skip]
                let t = Token { lexeme: String::from("{"), typ: TT::PuncLeftBrace };
                Ok(iter::once(t).chain(lex(r)?).collect())
            }
            '}' => {
                #[rustfmt::skip]
                let t = Token { lexeme: String::from("}"), typ: TT::PuncRightBrace };
                Ok(iter::once(t).chain(lex(r)?).collect())
            }
            ';' => {
                #[rustfmt::skip]
                let t = Token { lexeme: String::from(";"), typ: TT::PuncSemiColon };
                Ok(iter::once(t).chain(lex(r)?).collect())
            }
            ',' => {
                #[rustfmt::skip]
                let t = Token { lexeme: String::from(","), typ: TT::PuncComma };
                Ok(iter::once(t).chain(lex(r)?).collect())
            }
            _ => Err(io::Error::new(
                io::ErrorKind::Other,
                format!("unexpected token: {:?}", f),
            )),
        },
    }
}

fn scan_int(input: &[char]) -> Result<Vec<Token>, io::Error> {
    // scan_int calls skip_whitespace too to remain idempotent
    let cs = skip_ws(input);

    match cs {
        [] => Ok(vec![]),
        [f, _r @ ..] => match f {
            '0'..='9' => {
                #[rustfmt::skip]
                let i = _r
                    .iter()
                    .take_while(|&&c| c.is_numeric())
                    .count();

                let f = cs[..=i].iter().collect::<String>();
                let r = &cs[i + 1..];

                let t = Token {
                    lexeme: f,
                    typ: TT::LiteralInt,
                };

                Ok(iter::once(t).chain(lex(r)?).collect())
            }
            _ => Err(io::Error::new(
                io::ErrorKind::Other,
                format!("unexpected token: {:?}", f),
            )),
        },
    }
}

// TODO: support identifiers with alpha*numeric* characters after first alphabetic
fn scan_id(input: &[char]) -> Result<Vec<Token>, io::Error> {
    // scan_id calls skip_whitespace too to remain idempotent
    let cs = skip_ws(input);

    match cs {
        [] => Ok(vec![]),
        [f, r @ ..] => match f {
            'a'..='z' => {
                // Find the index where the alphabetic characters end
                #[rustfmt::skip]
                let i = r
                    .iter()
                    .take_while(|&&c| c.is_alphabetic())
                    .count();

                let f = (cs[..=i].iter()).collect::<String>();
                let new_r = &cs[i + 1..];

                let keyword = match f.as_str() {
                    "int" => Some(Token {
                        lexeme: f.to_string(),
                        typ: TT::KeywordInt,
                    }),
                    "if" => Some(Token {
                        lexeme: f.to_string(),
                        typ: TT::KeywordIf,
                    }),
                    "else" => Some(Token {
                        lexeme: f.to_string(),
                        typ: TT::KeywordEls,
                    }),
                    "for" => Some(Token {
                        lexeme: f.to_string(),
                        typ: TT::KeywordFor,
                    }),
                    "while" => Some(Token {
                        lexeme: f.to_string(),
                        typ: TT::KeywordWhile,
                    }),
                    "return" => Some(Token {
                        lexeme: f.to_string(),
                        typ: TT::KeywordRet,
                    }),
                    "true" => Some(Token {
                        lexeme: f.to_string(),
                        typ: TT::KeywordTrue,
                    }),
                    "false" => Some(Token {
                        lexeme: f.to_string(),
                        typ: TT::KeywordFalse,
                    }),
                    _ => None,
                };

                let t = match keyword {
                    Some(k) => k,
                    None => Token {
                        lexeme: f,
                        typ: TT::Alias,
                    },
                };

                Ok(iter::once(t).chain(lex(new_r)?).collect())
            }
            _ => Err(io::Error::new(
                io::ErrorKind::Other,
                format!("unexpected token: {:?}", f),
            )),
        },
    }
}

fn skip_ws(input: &[char]) -> &[char] {
    match input {
        [] => input,
        [f, r @ ..] => {
            if f.is_whitespace() {
                skip_ws(r)
            } else {
                input
            }
        }
    }
}

#[cfg(test)]
mod lex_arith {
    use std::fs;
    const TEST_DIR: &str = "tests/arith";

    #[test]
    fn lit() {
        #[rustfmt::skip]
        let input = fs::read(format!("{TEST_DIR}/lit.c"))
            .expect("file dne")
            .iter()
            .map(|b| *b as char)
            .collect::<Vec<_>>();

        let output = super::lex(input.as_slice()).unwrap();
        insta::assert_debug_snapshot!(output, @r###"
        [
            Token {
                lexeme: "int",
                typ: KeywordInt,
            },
            Token {
                lexeme: "main",
                typ: Alias,
            },
            Token {
                lexeme: "(",
                typ: PuncLeftParen,
            },
            Token {
                lexeme: ")",
                typ: PuncRightParen,
            },
            Token {
                lexeme: "{",
                typ: PuncLeftBrace,
            },
            Token {
                lexeme: "return",
                typ: KeywordRet,
            },
            Token {
                lexeme: "8",
                typ: LiteralInt,
            },
            Token {
                lexeme: ";",
                typ: PuncSemiColon,
            },
            Token {
                lexeme: "}",
                typ: PuncRightBrace,
            },
        ]
        "###);
    }

    #[test]
    fn add() {
        #[rustfmt::skip]
        let input = fs::read(format!("{TEST_DIR}/add.c"))
            .expect("file dne")
            .iter()
            .map(|b| *b as char)
            .collect::<Vec<_>>();

        let output = super::lex(input.as_slice()).unwrap();
        insta::assert_debug_snapshot!(output, @r###"
        [
            Token {
                lexeme: "int",
                typ: KeywordInt,
            },
            Token {
                lexeme: "main",
                typ: Alias,
            },
            Token {
                lexeme: "(",
                typ: PuncLeftParen,
            },
            Token {
                lexeme: ")",
                typ: PuncRightParen,
            },
            Token {
                lexeme: "{",
                typ: PuncLeftBrace,
            },
            Token {
                lexeme: "return",
                typ: KeywordRet,
            },
            Token {
                lexeme: "9",
                typ: LiteralInt,
            },
            Token {
                lexeme: "+",
                typ: Plus,
            },
            Token {
                lexeme: "10",
                typ: LiteralInt,
            },
            Token {
                lexeme: ";",
                typ: PuncSemiColon,
            },
            Token {
                lexeme: "}",
                typ: PuncRightBrace,
            },
        ]
        "###);
    }

    #[test]
    fn add_multi() {
        #[rustfmt::skip]
        let input = fs::read(format!("{TEST_DIR}/add_multi.c"))
            .expect("file dne")
            .iter()
            .map(|b| *b as char)
            .collect::<Vec<_>>();

        let output = super::lex(input.as_slice()).unwrap();
        insta::assert_debug_snapshot!(output, @r###"
        [
            Token {
                lexeme: "int",
                typ: KeywordInt,
            },
            Token {
                lexeme: "main",
                typ: Alias,
            },
            Token {
                lexeme: "(",
                typ: PuncLeftParen,
            },
            Token {
                lexeme: ")",
                typ: PuncRightParen,
            },
            Token {
                lexeme: "{",
                typ: PuncLeftBrace,
            },
            Token {
                lexeme: "return",
                typ: KeywordRet,
            },
            Token {
                lexeme: "9",
                typ: LiteralInt,
            },
            Token {
                lexeme: "+",
                typ: Plus,
            },
            Token {
                lexeme: "10",
                typ: LiteralInt,
            },
            Token {
                lexeme: "+",
                typ: Plus,
            },
            Token {
                lexeme: "11",
                typ: LiteralInt,
            },
            Token {
                lexeme: ";",
                typ: PuncSemiColon,
            },
            Token {
                lexeme: "}",
                typ: PuncRightBrace,
            },
        ]
        "###);
    }

    #[test]
    fn sub() {
        #[rustfmt::skip]
        let input = fs::read(format!("{}/sub.c", TEST_DIR))
            .expect("file dne")
            .iter()
            .map(|b| *b as char)
            .collect::<Vec<_>>();

        let output = super::lex(input.as_slice()).unwrap();
        insta::assert_debug_snapshot!(output, @r###"
        [
            Token {
                lexeme: "int",
                typ: KeywordInt,
            },
            Token {
                lexeme: "main",
                typ: Alias,
            },
            Token {
                lexeme: "(",
                typ: PuncLeftParen,
            },
            Token {
                lexeme: ")",
                typ: PuncRightParen,
            },
            Token {
                lexeme: "{",
                typ: PuncLeftBrace,
            },
            Token {
                lexeme: "return",
                typ: KeywordRet,
            },
            Token {
                lexeme: "88",
                typ: LiteralInt,
            },
            Token {
                lexeme: "-",
                typ: Minus,
            },
            Token {
                lexeme: "32",
                typ: LiteralInt,
            },
            Token {
                lexeme: ";",
                typ: PuncSemiColon,
            },
            Token {
                lexeme: "}",
                typ: PuncRightBrace,
            },
        ]
        "###);
    }

    #[test]
    fn mult() {
        #[rustfmt::skip]
        let input = fs::read(format!("{TEST_DIR}/mult.c"))
            .expect("file dne")
            .iter()
            .map(|b| *b as char)
            .collect::<Vec<_>>();

        let output = super::lex(input.as_slice()).unwrap();
        insta::assert_debug_snapshot!(output, @r###"
        [
            Token {
                lexeme: "int",
                typ: KeywordInt,
            },
            Token {
                lexeme: "main",
                typ: Alias,
            },
            Token {
                lexeme: "(",
                typ: PuncLeftParen,
            },
            Token {
                lexeme: ")",
                typ: PuncRightParen,
            },
            Token {
                lexeme: "{",
                typ: PuncLeftBrace,
            },
            Token {
                lexeme: "return",
                typ: KeywordRet,
            },
            Token {
                lexeme: "9",
                typ: LiteralInt,
            },
            Token {
                lexeme: "*",
                typ: Star,
            },
            Token {
                lexeme: "10",
                typ: LiteralInt,
            },
            Token {
                lexeme: ";",
                typ: PuncSemiColon,
            },
            Token {
                lexeme: "}",
                typ: PuncRightBrace,
            },
        ]
        "###);
    }

    #[test]
    fn div() {
        #[rustfmt::skip]
        let input = fs::read(format!("{TEST_DIR}/div.c"))
            .expect("file dne")
            .iter()
            .map(|b| *b as char)
            .collect::<Vec<_>>();

        let output = super::lex(input.as_slice()).unwrap();
        insta::assert_debug_snapshot!(output, @r###"
        [
            Token {
                lexeme: "int",
                typ: KeywordInt,
            },
            Token {
                lexeme: "main",
                typ: Alias,
            },
            Token {
                lexeme: "(",
                typ: PuncLeftParen,
            },
            Token {
                lexeme: ")",
                typ: PuncRightParen,
            },
            Token {
                lexeme: "{",
                typ: PuncLeftBrace,
            },
            Token {
                lexeme: "return",
                typ: KeywordRet,
            },
            Token {
                lexeme: "100",
                typ: LiteralInt,
            },
            Token {
                lexeme: "/",
                typ: Slash,
            },
            Token {
                lexeme: "9",
                typ: LiteralInt,
            },
            Token {
                lexeme: ";",
                typ: PuncSemiColon,
            },
            Token {
                lexeme: "}",
                typ: PuncRightBrace,
            },
        ]
        "###);
    }
}

#[cfg(test)]
mod lex_bindings {
    use std::fs;
    const TEST_DIR: &str = "tests/bindings";

    #[test]
    fn asnmt() {
        #[rustfmt::skip]
        let input = fs::read(format!("{TEST_DIR}/asnmt.c"))
            .expect("file dne")
            .iter()
            .map(|b| *b as char)
            .collect::<Vec<_>>();

        let output = super::lex(input.as_slice()).unwrap();
        insta::assert_debug_snapshot!(output, @r###"
        [
            Token {
                lexeme: "int",
                typ: KeywordInt,
            },
            Token {
                lexeme: "main",
                typ: Alias,
            },
            Token {
                lexeme: "(",
                typ: PuncLeftParen,
            },
            Token {
                lexeme: ")",
                typ: PuncRightParen,
            },
            Token {
                lexeme: "{",
                typ: PuncLeftBrace,
            },
            Token {
                lexeme: "int",
                typ: KeywordInt,
            },
            Token {
                lexeme: "x",
                typ: Alias,
            },
            Token {
                lexeme: "=",
                typ: Equals,
            },
            Token {
                lexeme: "8",
                typ: LiteralInt,
            },
            Token {
                lexeme: ";",
                typ: PuncSemiColon,
            },
            Token {
                lexeme: "return",
                typ: KeywordRet,
            },
            Token {
                lexeme: "x",
                typ: Alias,
            },
            Token {
                lexeme: ";",
                typ: PuncSemiColon,
            },
            Token {
                lexeme: "}",
                typ: PuncRightBrace,
            },
        ]
        "###);
    }

    #[test]
    fn composition() {
        #[rustfmt::skip]
        let input = fs::read(format!("{TEST_DIR}/composition.c"))
            .expect("file dne")
            .iter()
            .map(|b| *b as char)
            .collect::<Vec<_>>();

        let output = super::lex(input.as_slice()).unwrap();
        insta::assert_debug_snapshot!(output, @r###"
        [
            Token {
                lexeme: "int",
                typ: KeywordInt,
            },
            Token {
                lexeme: "h",
                typ: Alias,
            },
            Token {
                lexeme: "(",
                typ: PuncLeftParen,
            },
            Token {
                lexeme: ")",
                typ: PuncRightParen,
            },
            Token {
                lexeme: "{",
                typ: PuncLeftBrace,
            },
            Token {
                lexeme: "return",
                typ: KeywordRet,
            },
            Token {
                lexeme: "11",
                typ: LiteralInt,
            },
            Token {
                lexeme: ";",
                typ: PuncSemiColon,
            },
            Token {
                lexeme: "}",
                typ: PuncRightBrace,
            },
            Token {
                lexeme: "int",
                typ: KeywordInt,
            },
            Token {
                lexeme: "g",
                typ: Alias,
            },
            Token {
                lexeme: "(",
                typ: PuncLeftParen,
            },
            Token {
                lexeme: ")",
                typ: PuncRightParen,
            },
            Token {
                lexeme: "{",
                typ: PuncLeftBrace,
            },
            Token {
                lexeme: "return",
                typ: KeywordRet,
            },
            Token {
                lexeme: "10",
                typ: LiteralInt,
            },
            Token {
                lexeme: "+",
                typ: Plus,
            },
            Token {
                lexeme: "h",
                typ: Alias,
            },
            Token {
                lexeme: "(",
                typ: PuncLeftParen,
            },
            Token {
                lexeme: ")",
                typ: PuncRightParen,
            },
            Token {
                lexeme: ";",
                typ: PuncSemiColon,
            },
            Token {
                lexeme: "}",
                typ: PuncRightBrace,
            },
            Token {
                lexeme: "int",
                typ: KeywordInt,
            },
            Token {
                lexeme: "f",
                typ: Alias,
            },
            Token {
                lexeme: "(",
                typ: PuncLeftParen,
            },
            Token {
                lexeme: ")",
                typ: PuncRightParen,
            },
            Token {
                lexeme: "{",
                typ: PuncLeftBrace,
            },
            Token {
                lexeme: "return",
                typ: KeywordRet,
            },
            Token {
                lexeme: "9",
                typ: LiteralInt,
            },
            Token {
                lexeme: "+",
                typ: Plus,
            },
            Token {
                lexeme: "g",
                typ: Alias,
            },
            Token {
                lexeme: "(",
                typ: PuncLeftParen,
            },
            Token {
                lexeme: ")",
                typ: PuncRightParen,
            },
            Token {
                lexeme: ";",
                typ: PuncSemiColon,
            },
            Token {
                lexeme: "}",
                typ: PuncRightBrace,
            },
            Token {
                lexeme: "int",
                typ: KeywordInt,
            },
            Token {
                lexeme: "main",
                typ: Alias,
            },
            Token {
                lexeme: "(",
                typ: PuncLeftParen,
            },
            Token {
                lexeme: ")",
                typ: PuncRightParen,
            },
            Token {
                lexeme: "{",
                typ: PuncLeftBrace,
            },
            Token {
                lexeme: "return",
                typ: KeywordRet,
            },
            Token {
                lexeme: "f",
                typ: Alias,
            },
            Token {
                lexeme: "(",
                typ: PuncLeftParen,
            },
            Token {
                lexeme: ")",
                typ: PuncRightParen,
            },
            Token {
                lexeme: ";",
                typ: PuncSemiColon,
            },
            Token {
                lexeme: "}",
                typ: PuncRightBrace,
            },
        ]
        "###);
    }
}

#[cfg(test)]
mod lex_control {
    use std::fs;
    const TEST_DIR: &str = "tests/control";

    #[test]
    fn branch() {
        #[rustfmt::skip]
        let input = fs::read(format!("{TEST_DIR}/ifels_then.c"))
            .expect("file dne")
            .iter()
            .map(|b| *b as char)
            .collect::<Vec<_>>();

        let output = super::lex(input.as_slice()).unwrap();
        insta::assert_debug_snapshot!(output, @r###"
        [
            Token {
                lexeme: "int",
                typ: KeywordInt,
            },
            Token {
                lexeme: "main",
                typ: Alias,
            },
            Token {
                lexeme: "(",
                typ: PuncLeftParen,
            },
            Token {
                lexeme: ")",
                typ: PuncRightParen,
            },
            Token {
                lexeme: "{",
                typ: PuncLeftBrace,
            },
            Token {
                lexeme: "if",
                typ: KeywordIf,
            },
            Token {
                lexeme: "(",
                typ: PuncLeftParen,
            },
            Token {
                lexeme: "1",
                typ: LiteralInt,
            },
            Token {
                lexeme: ")",
                typ: PuncRightParen,
            },
            Token {
                lexeme: "{",
                typ: PuncLeftBrace,
            },
            Token {
                lexeme: "return",
                typ: KeywordRet,
            },
            Token {
                lexeme: "9",
                typ: LiteralInt,
            },
            Token {
                lexeme: ";",
                typ: PuncSemiColon,
            },
            Token {
                lexeme: "}",
                typ: PuncRightBrace,
            },
            Token {
                lexeme: "else",
                typ: KeywordEls,
            },
            Token {
                lexeme: "{",
                typ: PuncLeftBrace,
            },
            Token {
                lexeme: "return",
                typ: KeywordRet,
            },
            Token {
                lexeme: "10",
                typ: LiteralInt,
            },
            Token {
                lexeme: ";",
                typ: PuncSemiColon,
            },
            Token {
                lexeme: "}",
                typ: PuncRightBrace,
            },
            Token {
                lexeme: "}",
                typ: PuncRightBrace,
            },
        ]
        "###);
    }
}

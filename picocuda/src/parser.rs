use std::{io, iter};
use crate::{optimizer::Type, NodeDef, Node, OpCode};
use thiserror::Error;

thread_local! {
    pub static START: NodeDef = Node::new(OpCode::Start)
}

#[derive(Error, Debug)]
pub enum ParseError {
    // #[error("scope")]
    // Scope(#[from] ScopeError),
    #[error("(expected {expected:?}, found {actual:?})")]
    Mismatch { expected: String, actual: String },
}

// NB. each function in the parser will parse in two ways
//     a. conditionally (SUM/OR): match tokens(first, rest) first.typ { TT::Foo => {}, TT::Bar => {}, TT::Baz => {} }
//     b. assertively (PROD/AND): require(tokens, TT:Foo), eat(tokens, TT:Bar), eat(tokens, TT:Baz)
pub fn parse(tokens: &[Token]) -> Result<NodeDef, ParseError> {
    let r = tokens;
    let (_, r) = require(r, TT::KeywordInt)?;
    let (_, r) = require(r, TT::Alias)?;
    let (_, r) = require(r, TT::PuncLeftParen)?;
    let (_, r) = require(r, TT::PuncRightParen)?;

    let (_, r) = require(r, TT::PuncLeftBrace)?;
    // scope.push_nv();
    // scope.write(CTRL.to_owned(), Proj::new(*START.clone(), 0));
    // scope.write(ARG.to_owned(), Proj::new(*START.clone(), 1));
    let (block, r) = parse_block(r)?;
    // scope.pop_nv();

    let (_, r) = require(r, TT::PuncRightBrace)?;
    // try_convert block's Rc<dyn Instr> -> Rc<Return>

    if r.is_empty() { Ok(block) } else { Err(ParseError::Mismatch { expected: "empty token stream".to_string(), actual: format!("{:?}", r) }) }
}

// NB: lexical scope ==> nv's are only pushed/popped in parse_block
fn parse_block<'a>(tokens: &'a [Token]) -> Result<(NodeDef, &'a [Token]), ParseError> {
    // scope.push_nv();
    let (mut output, mut r) = (None, tokens);
    while let Ok((stmt, _r)) = parse_stmt(r) {
        output = Some(stmt);
        r = _r;
    }
    // scope.push_nv();
    Ok((output.unwrap(), r))
}

fn parse_stmt<'a>(tokens: &'a [Token]) -> Result<(NodeDef, &'a [Token]), ParseError> {
    match tokens {
        [] => Err(ParseError::Mismatch {
            expected: "expected: {:?} got an empty token stream".to_string(),
            actual: "".to_string(),
        }),
        [f, r @ ..] => match f.typ {
            TT::KeywordInt => {
                let (_alias, r) = require(r, TT::Alias)?;
                let (_, r) = require(r, TT::Equals)?;
                let (expr, r) = parse_expr(r)?;
                let (_, r) = require(r, TT::PuncSemiColon)?;

                // let _ = scope.write(alias.lexeme.to_owned(), expr.clone())?;
                Ok((expr, r))
            }
            // TT::KeywordIf => {
            //     let (pred, r) = parse_expr(r)?;

            //     let branch = Branch::new(scope.read_ctrl(), pred);
            //     let left = Proj::new(branch.clone(), 0).peephole(*START.clone());
            //     let right = Proj::new(branch, 1).peephole(*START.clone());
            //     let scope_og = Rc::new((*scope).clone()); // TODO: need ascii debugger here to verify

            //     // NB: because condtionals are statements and not expressions
            //     //     in C, the return of parse_stmts are not bound and ignored

            //     scope.write_ctrl(left); // 1. set ctrl
            //     let (_, r) = parse_stmt(r)?; // 2. parse
            //     let scope_left = Rc::new((*scope).clone()); // 3. alias scope

            //     scope = scope_og; // reset

            //     scope.write_ctrl(right); // 1. set ctrl
            //     if r.len() > 1 && r[0].typ == TT::KeywordEls { let (_, r) = parse_stmt(r)?; }; // 2. parse
            //     let scope_right = Rc::new((*scope).clone()); // 3. alias scope

            //     let region = scope_left.merge(&scope_right);
            //     scope.write_ctrl(region.clone());
            //     Ok((region, r))
            // },
            TT::KeywordRet => {
                let (expr, r) = parse_expr(r)?;
                let (_, r) = require(r, TT::PuncSemiColon)?;
                let start = START.with(|s| s.clone());
                let ret = Node::new(OpCode::Ret);
                let _ = Node::add_input(&ret, start);
                let _ = Node::add_input(&ret, expr);

                Ok((ret, r))
            }
            t => Err(ParseError::Mismatch {
                expected: format!("expected: {:?} got: {:?}", TT::KeywordRet, t),
                actual: f.lexeme.to_owned(),
            }),
        },
    }
}

fn parse_expr<'a>(tokens: &'a [Token]) -> Result<(NodeDef, &'a [Token]), ParseError> {
    parse_term(tokens)
}

fn parse_term<'a>(tokens: &'a [Token]) -> Result<(NodeDef, &'a [Token]), ParseError> {
    let (x, r) = parse_factor(tokens)?;

    match r {
        [] => panic!(),
        [f, _r @ ..] => match f.typ {
            TT::Plus => {
                let (y, r) = parse_factor(_r)?;
                let add = Node::new(OpCode::Add);
                let _ = Node::add_input(&add, x);
                let _ = Node::add_input(&add, y);
                
                Ok((add.peephole(), r))
            }
            TT::Minus => {
                let (y, r) = parse_factor(_r)?;
                let sub = Node::new(OpCode::Sub);
                let _ = Node::add_input(&sub, x);
                let _ = Node::add_input(&sub, y);
                
                Ok((sub.peephole(), r))
            }
            _ => Ok((x, r)),
        },
    }
}

fn parse_factor<'a>(tokens: &'a [Token]) -> Result<(NodeDef, &'a [Token]), ParseError> {
    let (x, r) = parse_atom(tokens)?;

    match r {
        [] => panic!(),
        [f, _r @ ..] => match f.typ {
            TT::Star => {
                let (y, r) = parse_atom(_r)?;
                let mul = Node::new(OpCode::Mul);
                let _ = Node::add_input(&mul, x);
                let _ = Node::add_input(&mul, y);
                
                Ok((mul.peephole(), r))
            }
            TT::Slash => {
                let (y, r) = parse_atom(_r)?;
                let div = Node::new(OpCode::Div);
                let _ = Node::add_input(&div, x);
                let _ = Node::add_input(&div, y);
                
                Ok((div.peephole(), r))
            }
            _ => Ok((x, r)),
        },
    }
}

fn parse_atom<'a>(tokens: &'a [Token]) -> Result<(NodeDef, &'a [Token]), ParseError> {
    match tokens {
        [] => Err(ParseError::Mismatch {
            expected: "expected: {:?} got an empty token stream".to_string(),
            actual: "".to_string(),
        }),
        [f, r @ ..] => match f.typ {
            TT::LiteralInt => {
                let start = START.with(|s| s.clone());
                let lit = Node::new_constant(OpCode::Con, Type::Int(f.lexeme.parse().unwrap()));
                let _ = Node::add_input(&lit, start);
                
                Ok((lit.peephole(), r))
            }
            // TT::Alias => {
            //     let expr = scope.read(f.lexeme.to_owned())?;
            //     Ok((expr,r))
            // },
            t => Err(ParseError::Mismatch {
                expected: format!("expected: {:?} got: {:?}", TT::LiteralInt, t),
                actual: f.lexeme.to_owned(),
            }),
        },
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
    use crate::{parser::{lex, parse}, OpCode};
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
        let graph = parse(&tokens).unwrap();

        assert_matches!(graph.borrow().opcode, OpCode::Ret);
        assert_matches!(graph.borrow().defs[0].borrow().opcode, OpCode::Start);
        insta::assert_debug_snapshot!(graph, @r###"
        NodeDef(
            RefCell {
                value: Node {
                    opcode: Ret,
                    typ: Bot,
                    defs: [
                        NodeDef(
                            RefCell {
                                value: Node {
                                    opcode: Start,
                                    typ: Bot,
                                    defs: [],
                                    uses: [
                                        NodeUse(
                                            (Weak),
                                        ),
                                        NodeUse(
                                            (Weak),
                                        ),
                                    ],
                                },
                            },
                        ),
                        NodeDef(
                            RefCell {
                                value: Node {
                                    opcode: Con,
                                    typ: Int(
                                        8,
                                    ),
                                    defs: [
                                        NodeDef(
                                            RefCell {
                                                value: Node {
                                                    opcode: Start,
                                                    typ: Bot,
                                                    defs: [],
                                                    uses: [
                                                        NodeUse(
                                                            (Weak),
                                                        ),
                                                        NodeUse(
                                                            (Weak),
                                                        ),
                                                    ],
                                                },
                                            },
                                        ),
                                    ],
                                    uses: [
                                        NodeUse(
                                            (Weak),
                                        ),
                                    ],
                                },
                            },
                        ),
                    ],
                    uses: [],
                },
            },
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
    fn add_compound() {
        #[rustfmt::skip]
        let input = fs::read(format!("{TEST_DIR}/add_compound.c"))
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
    fn mul() {
        #[rustfmt::skip]
        let input = fs::read(format!("{TEST_DIR}/mul.c"))
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

use std::{collections::HashMap, iter};
use crate::opto::son::{optimizer::Type, DefEdge, OpCode};
use thiserror::Error;

#[derive(Error, Debug)] pub enum ParseError {
    #[error("lex error")] LexError(#[from] LexError),
    #[error("parse error (expected {expected:?}, found {actual:?})")] Mismatch { expected: String, actual: String },
    #[error("scope error")] ScopeError(#[from] ScopeError),
}

pub struct ParseResult { pub start: DefEdge, pub stop: DefEdge, pub scope: Scope }
pub fn parse(chars: &[char]) -> Result<ParseResult, ParseError> {
    let tokens = lex(chars)?;
    let (start, scope) = (DefEdge::new(OpCode::Start), Scope::new());
    let mut parser = Parser::new(start, scope);
    let graph = parser.parse(&tokens, false)?;
    Ok(ParseResult { start: parser.start, stop: graph, scope: parser.scope })
}

// NB: temporary state created for recursive descent's access to &start (peepholes) and scope (varapp &/vardef &mut)
struct Parser { start: DefEdge, scope: Scope }
impl Parser {
    fn new(start: DefEdge, scope: Scope) -> Self { Self { start, scope } }

    // NB. each function in the parser will parse either:
    //     a. match: match tokens(first, rest) first.typ { TT::Foo => {}, TT::Bar => {}, TT::Baz => {} }
    //     b. assert: Self::require(tokens, TT:Foo), Self::require(tokens, TT:Bar), Self::require(tokens, TT:Baz)
    fn parse(&mut self, tokens: &[Token], _dump: bool) -> Result<DefEdge, ParseError> {
        let r = tokens;
        let (_, r) = Self::require(r, TT::KeywordInt)?;
        let (_, r) = Self::require(r, TT::Alias)?;
        let (_, r) = Self::require(r, TT::PuncLeftParen)?;
        let (_, r) = Self::require(r, TT::PuncRightParen)?;

        let (_, r) = Self::require(r, TT::PuncLeftBrace)?;
        self.scope.push_nv(); // global scope
        // scope.write(CTRL.to_owned(), Proj::new(*START.clone(), 0));
        // scope.write(ARG.to_owned(), Proj::new(*START.clone(), 1));
        let (block, r) = self.parse_block(r)?;
        self.scope.pop_nv();

        let (_, r) = Self::require(r, TT::PuncRightBrace)?;
        // try_convert block's Rc<dyn Instr> -> Rc<Return>

        // if dump {}
        if r.is_empty() { Ok(block) } else { Err(ParseError::Mismatch { expected: "empty token stream".to_string(), actual: format!("{:?}", r) }) }
    }

    // NB: lexical scope ==> nv's are only pushed/popped in parse_block
    fn parse_block<'a>(&mut self, tokens: &'a [Token]) -> Result<(DefEdge, &'a [Token]), ParseError> {
        self.scope.push_nv();
        let (mut output, mut r) = (None, tokens);
        while let Ok((stmt, _r)) = self.parse_stmt(r) {
            output = Some(stmt);
            r = _r;
        }
        self.scope.pop_nv();
        Ok((output.unwrap(), r))
    }

    fn parse_stmt<'a>(&mut self, tokens: &'a [Token]) -> Result<(DefEdge, &'a [Token]), ParseError> {
        match tokens {
            [] => Err(ParseError::Mismatch { expected: "".to_string(), actual: "".to_string() }),
            [f, r @ ..] => match f.typ {
                TT::KeywordInt => {
                    let (alias, r) = Self::require(r, TT::Alias)?;
                    let (_, r) = Self::require(r, TT::Equals)?;
                    let (expr, r) = self.parse_expr(r)?;
                    let (_, r) = Self::require(r, TT::PuncSemiColon)?;

                    let _ = self.scope.vardef(&alias.lexeme, expr.clone())?;
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
                    let (expr, r) = self.parse_expr(r)?;
                    let (_, r) = Self::require(r, TT::PuncSemiColon)?;
                    let ret = DefEdge::new(OpCode::Ret);
                    let _ = DefEdge::add_def(&ret, &self.start);
                    let _ = DefEdge::add_def(&ret, &expr);

                    Ok((ret, r))
                }
                t => Err(ParseError::Mismatch {
                    expected: format!("expected: {:?} got: {:?}", TT::KeywordRet, t),
                    actual: f.lexeme.to_owned(),
                }),
            },
        }
    }

    fn parse_expr<'a>(&self, tokens: &'a [Token]) -> Result<(DefEdge, &'a [Token]), ParseError> {
        self.parse_term(tokens)
    }

    fn parse_term<'a>(&self, tokens: &'a [Token]) -> Result<(DefEdge, &'a [Token]), ParseError> {
        let (x, r) = self.parse_factor(tokens)?;

        match r {
            [] => panic!(),
            [f, _r @ ..] => match f.typ {
                TT::Plus => {
                    let (y, r) = self.parse_term(_r)?;
                    let add = DefEdge::new(OpCode::Add);
                    let (_, _) = (add.add_def(&x), add.add_def(&y));
                    Ok((add.peephole(&self.start), r))
                }
                TT::Minus => {
                    let (y, r) = self.parse_term(_r)?;
                    let sub = DefEdge::new(OpCode::Sub);
                    let (_, _) = (sub.add_def(&x), sub.add_def(&y));
                    Ok((sub.peephole(&self.start), r))
                }
                _ => Ok((x, r)),
            },
        }
    }

    fn parse_factor<'a>(&self, tokens: &'a [Token]) -> Result<(DefEdge, &'a [Token]), ParseError> {
        let (x, r) = self.parse_atom(tokens)?;

        match r {
            [] => panic!(),
            [f, _r @ ..] => match f.typ {
                TT::Star => {
                    let (y, r) = self.parse_factor(_r)?;
                    let mul = DefEdge::new(OpCode::Mul);
                    let (_, _) = (mul.add_def(&x), mul.add_def(&y));
                    Ok((mul.peephole(&self.start), r))
                }
                TT::Slash => {
                    let (y, r) = self.parse_factor(_r)?;
                    let div = DefEdge::new(OpCode::Div);
                    let (_, _) = (div.add_def(&x), div.add_def(&y));
                    Ok((div.peephole(&self.start), r))
                }
                _ => Ok((x, r)),
            },
        }
    }

    fn parse_atom<'a>(&self, tokens: &'a [Token]) -> Result<(DefEdge, &'a [Token]), ParseError> {
        match tokens {
            [] => Err(ParseError::Mismatch { expected: "".to_string(), actual: "".to_string() }),
            [f, r @ ..] => match f.typ {
                TT::LiteralInt => {
                    let lit = DefEdge::new_constant(OpCode::Con, Type::Int(f.lexeme.parse().unwrap()));
                    let _ = lit.add_def(&self.start);
                    
                    Ok((lit.peephole(&self.start), r))
                }
                TT::Alias => {
                    let expr = self.scope.varapp(&f.lexeme)?;
                    Ok((expr,r))
                },
                t => Err(ParseError::Mismatch {
                    expected: format!("expected: {:?} got: {:?}", TT::LiteralInt, t),
                    actual: f.lexeme.to_owned(),
                }),
            },
        }
    }

    fn require<'a> (tokens: &'a [Token], tt: TT) -> Result<(&'a Token, &'a [Token]), ParseError> {
        match tokens {
            [] => Err(ParseError::Mismatch { expected: format!("expected: {:?} got: {:?}", tt, tokens), actual: "".to_string(),
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
}

#[derive(Clone, PartialEq, Debug)]
pub struct Token { pub lexeme: String, pub typ: TT }

//  1. variations are explicitly typed. Collapsing categories like keywords
//     into one variant will lose information since lexeme : String, which
//     will produce redundant work for the parser during syntactic analysis
//  2. non-tokens: comments, preprocessor directives, macros, whitespace
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum TT {
    LiteralInt, Alias, // introductions (values) RE: [0-9]+ and [a-zA-Z][a-zA-Z0-9]*
    KeywordInt, KeywordChar, KeywordVoid, KeywordRet, KeywordIf, KeywordEls, KeywordFor, KeywordWhile, KeywordTrue, KeywordFalse, // keywords ⊂ identifiers
    Plus, Minus, Star, Slash, LeftAngleBracket, RightAngleBracket, Equals, Bang, Amp, Bar, // eliminations (ops)
    PuncLeftParen, PuncRightParen, PuncLeftBrace, PuncRightBrace, PuncSemiColon, PuncComma,// punctuation
}

#[derive(Error, Debug)]
pub enum LexError { #[error("(unknown token {unknown:?}")] UnknownToken { unknown: String } }

fn lex(input: &[char]) -> Result<Vec<Token>, LexError> {
    let cs = skip_ws(input);
    // literals and identifiers have arbitrary length, operations and punctuations are single ASCII characters
    match cs {
        [] => Ok(vec![]),
        [f, r @ ..] => match f {
            '0'..='9' => scan_int(cs),
            'a'..='z' | 'A'..='Z' => scan_id(cs),
            '+' => { let t = Token { lexeme: String::from("+"), typ: TT::Plus }; Ok(iter::once(t).chain(lex(r)?).collect()) }
            '-' => { let t = Token { lexeme: String::from("-"), typ: TT::Minus }; Ok(iter::once(t).chain(lex(r)?).collect()) }
            '*' => { let t = Token { lexeme: String::from("*"), typ: TT::Star }; Ok(iter::once(t).chain(lex(r)?).collect()) }
            '/' => { let t = Token { lexeme: String::from("/"), typ: TT::Slash }; Ok(iter::once(t).chain(lex(r)?).collect()) }
            '<' => { let t = Token { lexeme: String::from("<"), typ: TT::LeftAngleBracket }; Ok(iter::once(t).chain(lex(r)?).collect()) }
            '>' => { let t = Token { lexeme: String::from(">"), typ: TT::RightAngleBracket }; Ok(iter::once(t).chain(lex(r)?).collect()) }
            '=' => { let t = Token { lexeme: String::from("="), typ: TT::Equals }; Ok(iter::once(t).chain(lex(r)?).collect()) }
            '!' => { let t = Token { lexeme: String::from("!"), typ: TT::Bang }; Ok(iter::once(t).chain(lex(r)?).collect()) }
            '&' => { let t = Token { lexeme: String::from("&"), typ: TT::Amp }; Ok(iter::once(t).chain(lex(r)?).collect()) }
            '|' => { let t = Token { lexeme: String::from("|"), typ: TT::Bar }; Ok(iter::once(t).chain(lex(r)?).collect()) }
            '(' => { let t = Token { lexeme: String::from("("), typ: TT::PuncLeftParen }; Ok(iter::once(t).chain(lex(r)?).collect()) }
            ')' => { let t = Token { lexeme: String::from(")"), typ: TT::PuncRightParen }; Ok(iter::once(t).chain(lex(r)?).collect()) }
            '{' => { let t = Token { lexeme: String::from("{"), typ: TT::PuncLeftBrace }; Ok(iter::once(t).chain(lex(r)?).collect()) }
            '}' => { let t = Token { lexeme: String::from("}"), typ: TT::PuncRightBrace }; Ok(iter::once(t).chain(lex(r)?).collect()) }
            ';' => { let t = Token { lexeme: String::from(";"), typ: TT::PuncSemiColon }; Ok(iter::once(t).chain(lex(r)?).collect()) }
            ',' => { let t = Token { lexeme: String::from(","), typ: TT::PuncComma }; Ok(iter::once(t).chain(lex(r)?).collect()) }
            _ => Err(LexError::UnknownToken { unknown: f.to_string() }),
        },
    }
}

fn scan_int(input: &[char]) -> Result<Vec<Token>, LexError> {
    // scan_int calls skip_whitespace too to remain idempotent
    let cs = skip_ws(input);

    match cs {
        [] => Ok(vec![]),
        [f, _r @ ..] => match f {
            '0'..='9' => {
                let i = _r.iter().take_while(|&&c| c.is_numeric()).count();
                let f = cs[..=i].iter().collect::<String>();
                let r = &cs[i + 1..];
                let t = Token { lexeme: f, typ: TT::LiteralInt };
                Ok(iter::once(t).chain(lex(r)?).collect())
            }
            _ => Err(LexError::UnknownToken { unknown: f.to_string() }),
        },
    }
}

// TODO: support identifiers with alpha*numeric* characters after first alphabetic
fn scan_id(input: &[char]) -> Result<Vec<Token>, LexError> {
    // scan_id calls skip_whitespace too to remain idempotent
    let cs = skip_ws(input);

    match cs {
        [] => Ok(vec![]),
        [f, r @ ..] => match f {
            'a'..='z' => {
                // Find the index where the alphabetic characters end
                let i = r.iter().take_while(|&&c| c.is_alphabetic()).count();

                let f = (cs[..=i].iter()).collect::<String>();
                let new_r = &cs[i + 1..];

                let keyword = match f.as_str() {
                    "int" => Some(Token { lexeme: f.to_string(), typ: TT::KeywordInt }),
                    "if" => Some(Token { lexeme: f.to_string(), typ: TT::KeywordIf }),
                    "else" => Some(Token { lexeme: f.to_string(), typ: TT::KeywordEls }),
                    "for" => Some(Token { lexeme: f.to_string(), typ: TT::KeywordFor }),
                    "while" => Some(Token { lexeme: f.to_string(), typ: TT::KeywordWhile }),
                    "return" => Some(Token { lexeme: f.to_string(), typ: TT::KeywordRet }),
                    "true" => Some(Token { lexeme: f.to_string(), typ: TT::KeywordTrue }),
                    "false" => Some(Token { lexeme: f.to_string(), typ: TT::KeywordFalse }),
                    _ => None,
                };

                let t = match keyword {
                    Some(k) => k,
                    None => Token { lexeme: f, typ: TT::Alias },
                };
                Ok(iter::once(t).chain(lex(new_r)?).collect())
            }
            _ => Err(LexError::UnknownToken { unknown: f.to_string() }),
        },
    }
}

fn skip_ws(input: &[char]) -> &[char] { match input { [] => input, [f, r @ ..] => if f.is_whitespace() { skip_ws(r) } else { input } } }

#[cfg(test)]
mod test_parser {
    use crate::opto::son::{dumper, parser, utils::read_chars, OpCode};
    use std::{assert_matches::assert_matches, fs, path::Path};
    
    const TEST_DIR: &str = "tests/arith";

    #[test]
    fn lit() {
        let chars = read_chars(Path::new(&format!("{TEST_DIR}/lit.c")));
        let graph = parser::parse(&chars).unwrap();
        // let dot = dumper::dump_dot(&chars, &graph).unwrap();
        // println!("{dot}");

        // assert_matches!(graph.borrow().opcode, OpCode::Ret);
        // assert_matches!(graph.borrow().defs[0].borrow().opcode, OpCode::Start);
        // insta::assert_debug_snapshot!(graph);
    }

    #[test] fn add_compound() {
        let chars = read_chars(Path::new(&format!("{TEST_DIR}/add_compound.c")));
        let graph = parser::parse(&chars).unwrap();
        // let dot = dumper::dump_dot(&chars, &graph).unwrap();
        // println!("{dot}");

        // assert_matches!(graph.borrow().opcode, OpCode::Ret);
        // assert_matches!(graph.borrow().defs[0].borrow().opcode, OpCode::Start);
        // insta::assert_debug_snapshot!(graph);
    }
}

#[derive(Error, Debug)] pub enum ScopeError { #[error("double define")] DoubleDefine, #[error("not found")] NotFound, #[error("no environment exists")] NoNvExists }
// NB: scope *is neither* a control node nor data node. it *uses* a SoN node's
//     def/use edges to keep track of liveliness. more specifically, the only
//     edges used with the node inside the scope struct are def edges: ones
//     that point to nodes that are expressions (in the case of C, just data nodes)
//     that is, the scope's node has no uses.
pub struct Scope { pub lookup: DefEdge, pub nvs: Vec<HashMap<String, usize>> }
impl Scope {
    fn new() -> Self { Self { lookup: DefEdge::new(OpCode::Scope), nvs: Vec::new() } }
    fn push_nv(&mut self) -> () { self.nvs.push(HashMap::new()) }
    fn pop_nv(&mut self) -> () { let _ = self.nvs.pop().unwrap(); }
    fn varapp(&self, alias: &str) -> Result<DefEdge, ScopeError> {self.read_update(alias, ScopeOp::Read, self.nvs.len()-1)}
    fn _varupd(&self, alias: &str, expr: DefEdge) -> Result<DefEdge, ScopeError> {self.read_update(alias, ScopeOp::Update(expr), self.nvs.len()-1)}
    fn vardef(&mut self, alias: &str, expr: DefEdge) -> Result<(), ScopeError> {
        let cur_nv = self.nvs.last_mut().ok_or(ScopeError::NoNvExists)?;
        match cur_nv.contains_key(alias) {
            true => Err(ScopeError::DoubleDefine),
            false => {
                self.lookup.add_def(&expr);
                let i_def = self.lookup.borrow().defs.len()-1;
                cur_nv.insert(alias.to_owned(), i_def);
                Ok(())
            },
        }
    }

    // shared read/update makes lazi phi creation easier ch8
    fn read_update(&self, alias: &str, op: ScopeOp, level: usize ) -> Result<DefEdge, ScopeError> {
        let cur_nv = self.nvs.get(level).unwrap();
        match cur_nv.get(alias) {
            None => if level == 0 { Err(ScopeError::NotFound) } else { self.read_update(alias, op, level-1) },
            Some(i_def) => {
                let expr = self.lookup.borrow().defs[*i_def].clone();
                Ok(match op { ScopeOp::Read => expr, ScopeOp::Update(n) => {
                    self.lookup.borrow_mut().defs[*i_def] = n; // updating std::vec calls drop on rc
                    self.lookup.borrow().defs[*i_def].clone()
                },})
            },
        }
    }
}
enum ScopeOp { Read, Update(DefEdge) }

// #[cfg(test)]
// mod test_scope {
//     use crate::son::{parser::Parser};
//     use std::fs;
    
//     const TEST_DIR: &str = "tests/bindings";

//     #[test]
//     fn asnmt() {
//         let chars = fs::read(format!("{TEST_DIR}/asnmt.c"))
//             .expect("file dne")
//             .iter()
//             .map(|b| *b as char)
//             .collect::<Vec<_>>();
    
//         let mut parser = Parser::new(chars);
//         let tokens = parser.lex().unwrap();
//         let graph = parser.parse(&tokens, false).unwrap();

//         insta::assert_debug_snapshot!(graph);
//     }

//     #[test]
//     fn asnmt_expr() {
//         let chars = fs::read(format!("{TEST_DIR}/asnmt_expr.c"))
//             .expect("file dne")
//             .iter()
//             .map(|b| *b as char)
//             .collect::<Vec<_>>();
    
//         let mut parser = Parser::new(chars);
//         let tokens = parser.lex().unwrap();
//         let graph = parser.parse(&tokens, false).unwrap();

//         insta::assert_debug_snapshot!(graph);
//     }
// }

#[cfg(test)]
mod test_lexer {
    use std::path::Path;

    use crate::opto::son::{parser, utils::read_chars};

    // arithmetic
    #[test] fn lit() {
        let chars = read_chars(Path::new("tests/arith/lit.c"));
        let tokens = parser::lex(&chars).unwrap();
        insta::assert_debug_snapshot!(tokens);
    }
    #[test] fn add() {
        let chars = read_chars(Path::new("tests/arith/add.c"));
        let tokens = parser::lex(&chars).unwrap();
        insta::assert_debug_snapshot!(tokens);
    }
    #[test] fn add_compound() {
        let chars = read_chars(Path::new("tests/arith/add_compound.c"));
        let tokens = parser::lex(&chars).unwrap();
        insta::assert_debug_snapshot!(tokens);
    }
    #[test] fn sub() {
        let chars = read_chars(Path::new("tests/arith/sub.c"));
        let tokens = parser::lex(&chars).unwrap();
        insta::assert_debug_snapshot!(tokens);
    }
    #[test] fn mul() {
        let chars = read_chars(Path::new("tests/arith/mul.c"));
        let tokens = parser::lex(&chars).unwrap();
        insta::assert_debug_snapshot!(tokens);
    }
    #[test] fn div() {
        let chars = read_chars(Path::new("tests/arith/div.c"));
        let tokens = parser::lex(&chars).unwrap();
        insta::assert_debug_snapshot!(tokens);
    }

    // bindings
    #[test] fn asnmt() {
        let chars = read_chars(Path::new("tests/bindings/asnmt.c"));
        let tokens = parser::lex(&chars).unwrap();
        insta::assert_debug_snapshot!(tokens);
    }
    #[test] fn composition() {
        let chars = read_chars(Path::new("tests/bindings/asnmt_composition.c"));
        let tokens = parser::lex(&chars).unwrap();
        insta::assert_debug_snapshot!(tokens);
    }

    // control
    #[test] fn branch() {
        let chars = read_chars(Path::new("tests/control/branch.c"));
        let tokens = parser::lex(&chars).unwrap();
        insta::assert_debug_snapshot!(tokens);
    }
}
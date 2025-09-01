#[derive(Debug, Clone)]
enum Nat {
    Z,
    S(Box<Self>)
}

fn succ(x: Nat) -> Nat {
    Nat::S(Box::new(x))
}

fn pred(x: Nat) -> Option<Nat> {
    match x {
    Nat::Z => None,
    Nat::S(n) => Some(*n),
    }
}

fn plus(x: Nat, y: Nat) -> Nat {
    match x {
    Nat::Z => y,
    Nat::S(n) => Nat::S(Box::new(plus(*n, y))),
    }
}

fn mul(x: Nat, y: Nat) -> Nat {
    match x {
    Nat::Z => Nat::Z,
    Nat::S(n) => plus(y.clone(), mul(*n, y)),
    }
}

fn exp(x: Nat, y: Nat) -> Nat {
    match y {
    Nat::Z => Nat::S(Box::new(Nat::Z)),
    Nat::S(n) => mul(x.clone(), exp(x, *n)),
    }
}

#[cfg(test)]
mod test_nat {
    use super::*;
    use proptest::prelude::*;
    
    #[test]
    fn test_succ() -> () {
        let two = Nat::S(Box::new(Nat::S(Box::new(Nat::Z))));
        let output = succ(two);
        println!("{:?}", output);
    }

    proptest! {
        #[test]
        fn test_succ_doesnotcrash(s in "\\PC*") {
            let two = Nat::S(Box::new(Nat::S(Box::new(Nat::Z))));
            let _output = succ(two);
        }
    }

    #[test]
    fn test_pred() -> () {
        let two = Nat::S(Box::new(Nat::S(Box::new(Nat::Z))));
        let output = pred(two);
        println!("{:?}", output);
    }

    #[test]
    fn test_plus() -> () {
        let two = Nat::S(Box::new(Nat::S(Box::new(Nat::Z))));
        let one = Nat::S(Box::new(Nat::Z));
        let output = plus(one, two);
        println!("{:?}", output);
    }

    #[test]
    fn test_mul() -> () {
        let two = Nat::S(Box::new(Nat::S(Box::new(Nat::Z))));
        let three = Nat::S(Box::new(Nat::S(Box::new(Nat::S(Box::new(Nat::Z))))));
        let output = mul(two, three);
        println!("{:?}", output);
    }
}
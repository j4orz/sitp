#[derive(Debug)]
enum Nat {
    Z,
    S(Box<Self>)
}
struct Int {}
struct Rat {}

impl Nat {
    fn succ(slf: Self) -> Self {
        Nat::S(Box::new(slf))
    }

    fn pred(slf: Self) -> Option<Self> {
        match slf {
        Nat::Z => None,
        Nat::S(n) => Some(*n),
        }
    }

    fn plus(slf: Self, other: Self) -> Self {
        match slf {
        Nat::Z => other,
        Nat::S(n) => Nat::S(Box::new(Self::plus(*n, other))),
        }
    }
}

#[cfg(test)]
mod test_nat {
    use super::*;
    
    #[test]
    fn test_succ() -> () {
        let two = Nat::S(Box::new(Nat::S(Box::new(Nat::Z))));
        let output = Nat::succ(two);
        println!("{:?}", output);
    }

    #[test]
    fn test_pred() -> () {
        let two = Nat::S(Box::new(Nat::S(Box::new(Nat::Z))));
        let output = Nat::pred(two);
        println!("{:?}", output);
    }

    #[test]
    fn test_plus() -> () {
        let two = Nat::S(Box::new(Nat::S(Box::new(Nat::Z))));
        let one = Nat::S(Box::new(Nat::Z));
        let output = Nat::plus(one, two);
        println!("{:?}", output);
    }
}
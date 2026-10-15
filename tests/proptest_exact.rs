//! Property-based tests for the exact-rational analytic core.
//!
//! These exercise the algebraic laws (`Rational` / `ExactComplex`) over many
//! random inputs — the class of subtle bug (overflow, normalization,
//! commutativity) that fixed example tests miss. Numerators/denominators are
//! bounded so intermediate products stay well inside `i128` (overflow is a
//! legitimate `Err`, not a property violation, and is tested separately).

use proptest::prelude::*;
use quantumforge::analytic::{ExactComplex, Pauli, PauliSum, PauliWord, Rational};

/// A bounded non-zero denominator.
fn denom() -> impl Strategy<Value = i128> {
    (1i128..500).prop_map(|d| d)
}

/// A bounded numerator.
fn numer() -> impl Strategy<Value = i128> {
    (-500i128..500).prop_map(|n| n)
}

/// A valid `Rational` with a non-zero denominator (normalized on construction).
fn rational() -> impl Strategy<Value = Rational> {
    (numer(), denom()).prop_map(|(n, d)| Rational::new(n, d).expect("non-zero denom constructs"))
}

/// A valid `ExactComplex`.
fn exact_complex() -> impl Strategy<Value = ExactComplex> {
    (rational(), rational()).prop_map(|(re, im)| ExactComplex { re, im })
}

proptest! {
    #[test]
    fn rational_new_is_reduced(n in numer(), d in denom()) {
        let r = Rational::new(n, d).expect("constructs");
        // Normalized: gcd(|num|, den) == 1, and the denominator is positive.
        let g = gcd(r.numerator().unsigned_abs(), r.denominator() as u128);
        prop_assert_eq!(g, 1, "not fully reduced");
        prop_assert!(r.denominator() > 0, "denominator must be positive");
    }

    #[test]
    fn rational_add_commutes(a in rational(), b in rational()) {
        let ab = a.checked_add(b).expect("add");
        let ba = b.checked_add(a).expect("add");
        prop_assert_eq!(ab, ba);
    }

    #[test]
    fn rational_add_associates(a in rational(), b in rational(), c in rational()) {
        let left = a.checked_add(b).and_then(|x| x.checked_add(c));
        let right = b.checked_add(c).and_then(|y| a.checked_add(y));
        prop_assert_eq!(left, right);
    }

    #[test]
    fn rational_mul_commutes(a in rational(), b in rational()) {
        prop_assert_eq!(a.checked_mul(b), b.checked_mul(a));
    }

    #[test]
    fn rational_mul_associates(a in rational(), b in rational(), c in rational()) {
        let left = a.checked_mul(b).and_then(|x| x.checked_mul(c));
        let right = b.checked_mul(c).and_then(|y| a.checked_mul(y));
        prop_assert_eq!(left, right);
    }

    #[test]
    fn rational_distributes(a in rational(), b in rational(), c in rational()) {
        // a * (b + c) == a*b + a*c
        let left = b.checked_add(c).and_then(|s| a.checked_mul(s));
        let right = a.checked_mul(b).and_then(|ab| a.checked_mul(c).and_then(|ac| ab.checked_add(ac)));
        prop_assert_eq!(left, right);
    }

    #[test]
    fn rational_zero_is_additive_identity(a in rational()) {
        prop_assert_eq!(a.checked_add(Rational::ZERO), Ok(a));
        prop_assert_eq!(Rational::ZERO.checked_add(a), Ok(a));
    }

    #[test]
    fn rational_one_is_multiplicative_identity(a in rational()) {
        prop_assert_eq!(a.checked_mul(Rational::ONE), Ok(a));
    }

    #[test]
    fn rational_additive_inverse(a in rational()) {
        // a + (-a) == 0
        let neg = Rational::new(-a.numerator(), a.denominator()).expect("neg");
        prop_assert_eq!(a.checked_add(neg), Ok(Rational::ZERO));
    }

    #[test]
    fn rational_multiplicative_inverse_nonzero(a in rational()) {
        prop_assume!(!a.is_zero());
        let inv = Rational::new(a.denominator(), a.numerator()).expect("inv");
        prop_assert_eq!(a.checked_mul(inv), Ok(Rational::ONE));
    }

    #[test]
    fn rational_zero_denominator_rejected(n in numer()) {
        prop_assert!(Rational::new(n, 0).is_err());
    }

    #[test]
    fn exact_complex_add_commutes(a in exact_complex(), b in exact_complex()) {
        prop_assert_eq!(a.checked_add(b), b.checked_add(a));
    }

    #[test]
    fn exact_complex_mul_commutes(a in exact_complex(), b in exact_complex()) {
        prop_assert_eq!(a.checked_mul(b), b.checked_mul(a));
    }

    #[test]
    fn exact_complex_distributes(a in exact_complex(), b in exact_complex(), c in exact_complex()) {
        let left = b.checked_add(c).and_then(|s| a.checked_mul(s));
        let right = a.checked_mul(b).and_then(|ab| a.checked_mul(c).and_then(|ac| ab.checked_add(ac)));
        prop_assert_eq!(left, right);
    }

    #[test]
    fn exact_complex_i_squared_is_minus_one(_seed in any::<u8>()) {
        // I * I == -1
        prop_assert_eq!(ExactComplex::I.checked_mul(ExactComplex::I), Ok(ExactComplex::real(Rational::MINUS_ONE)));
    }

    #[test]
    fn exact_complex_additive_inverse(a in exact_complex()) {
        let neg = ExactComplex {
            re: Rational::new(-a.re.numerator(), a.re.denominator()).expect("neg re"),
            im: Rational::new(-a.im.numerator(), a.im.denominator()).expect("neg im"),
        };
        prop_assert_eq!(a.checked_add(neg), Ok(ExactComplex::ZERO));
    }

    #[test]
    fn exact_complex_to_complex64_matches_parts(z in exact_complex()) {
        // The f64 projection must equal the exact parts evaluated in f64.
        let c = z.to_complex64();
        let re = z.re.numerator() as f64 / z.re.denominator() as f64;
        let im = z.im.numerator() as f64 / z.im.denominator() as f64;
        prop_assert_eq!(c.re, re);
        prop_assert_eq!(c.im, im);
    }
}

// PauliSum: adding the zero sum is the identity.
proptest! {
    #[test]
    fn pauli_sum_add_zero_is_identity(
        n_sites in 1usize..4,
        seed in any::<u64>(),
    ) {
        // Build a deterministic sum from the seed.
        let mut sum = PauliSum::zero(n_sites);
        let words = seed % 3;
        for w in 0..words {
            let mut factors = vec![Pauli::I; n_sites];
            factors[(w as usize) % n_sites] = Pauli::X;
            let coeff = ExactComplex::real(Rational::new((w as i128) + 1, 2).unwrap());
            sum.add_term(coeff, PauliWord::new(factors)).unwrap();
        }
        let zero = PauliSum::zero(n_sites);
        prop_assert_eq!(sum.checked_add(&zero), Ok(sum.clone()));
    }
}

/// Greatest common divisor (mirrors the crate's private helper) for the
/// reduction-law test above.
fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

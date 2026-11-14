//! Property-based tests for the CAS layer (ScalarExpr algebra):
//! integrate/differentiate round-trips, solve-verifies, like-term combining.
//! Generate random polynomials over symbol "x".

use proptest::prelude::*;
use quantumforge::analytic::{ExactComplex, Rational, ScalarExpr};

fn rational() -> impl Strategy<Value = Rational> {
    (-20i128..20)
        .prop_flat_map(|n| (Just(n), 1i128..10))
        .prop_map(|(n, d)| Rational::new(n, d).unwrap())
}

fn exact() -> impl Strategy<Value = ExactComplex> {
    rational().prop_map(ExactComplex::real)
}

/// Random polynomial in x: sum of up to 4 terms, degrees 0..4.
fn poly() -> impl Strategy<Value = ScalarExpr> {
    proptest::collection::vec((exact(), 0u32..4), 1..5).prop_map(|terms| {
        let built: Vec<ScalarExpr> = terms
            .into_iter()
            .map(|(c, e)| {
                if e == 0 {
                    ScalarExpr::exact(c)
                } else {
                    ScalarExpr::mul([
                        ScalarExpr::exact(c),
                        ScalarExpr::pow(ScalarExpr::symbol("x"), e as i32).unwrap(),
                    ])
                    .unwrap()
                }
            })
            .collect();
        ScalarExpr::add(built).unwrap()
    })
}

proptest! {
    #[test]
    fn integrate_then_differentiate_roundtrips(p in poly()) {
        let a = p.integrate("x").unwrap();
        let d = a.differentiate("x").unwrap();
        prop_assert_eq!(d.canonicalize(), p.canonicalize(), "roundtrip failed for {}", p);
    }

    #[test]
    fn add_combines_like_terms(c in exact()) {
        let a = ScalarExpr::exact(c);
        let sum = ScalarExpr::add([a.clone(), a]).unwrap();
        let expected = ScalarExpr::exact(c.checked_mul(ExactComplex::real(Rational::new(2, 1).unwrap())).unwrap());
        prop_assert_eq!(sum.canonicalize(), expected.canonicalize());
    }

    #[test]
    fn degree_of_product_is_sum(a_deg in 0u32..4, b_deg in 0u32..4) {
        let a = ScalarExpr::pow(ScalarExpr::symbol("x"), a_deg as i32).unwrap();
        let b = ScalarExpr::pow(ScalarExpr::symbol("x"), b_deg as i32).unwrap();
        let prod = ScalarExpr::mul([a, b]).unwrap();
        prop_assert_eq!(prod.degree("x"), Some((a_deg + b_deg) as i32));
    }

    #[test]
    fn solve_verifies_roots(c in rational()) {
        // a*(x - c) = a*x - a*c  => root c
        let a = ScalarExpr::exact(ExactComplex::real(Rational::new(3, 1).unwrap()));
        let _root = ScalarExpr::exact(ExactComplex::real(c));
        let expr = ScalarExpr::add([
            ScalarExpr::mul([a.clone(), ScalarExpr::symbol("x")]).unwrap(),
            ScalarExpr::mul([a, ScalarExpr::exact(ExactComplex::real(c.checked_neg().unwrap()))]).unwrap(),
        ])
        .unwrap();
        let sol = expr.solve("x");
        prop_assert_eq!(sol.len(), 1);
        // substituting the solution back into the expression yields zero
        let v = expr.compose("x", &sol[0]).unwrap();
        prop_assert_eq!(v.canonicalize(), ScalarExpr::zero(), "root {} does not satisfy", c);
    }

    #[test]
    fn simplify_is_idempotent(p in poly()) {
        let s = quantumforge::analytic::Simplifier::default().simplify(&p).unwrap();
        let s2 = quantumforge::analytic::Simplifier::default().simplify(&s).unwrap();
        prop_assert_eq!(s, s2);
    }
}

//! Example: exact verification of known polynomial identities.
//!
//! This demonstrates QuantumForge's core differentiator: exact verification
//! of algebraic identities that floating-point comparison cannot do reliably.
//!
//! Run with: cargo run --example exact_identities

use quantumforge::analytic::{ExactComplex, Rational, ScalarExpr};

fn neg_one() -> ScalarExpr {
    ScalarExpr::exact(ExactComplex::real(Rational::new(-1, 1).unwrap()))
}

fn main() {
    println!("=== Exact Identity Verification ===\n");

    // Identity 1: (a+b)^2 = a^2 + 2ab + b^2
    let a = ScalarExpr::symbol("a");
    let b = ScalarExpr::symbol("b");
    let lhs = ScalarExpr::pow(ScalarExpr::add([a.clone(), b.clone()]).unwrap(), 2).unwrap();
    let rhs = ScalarExpr::add([
        ScalarExpr::pow(a.clone(), 2).unwrap(),
        ScalarExpr::mul([
            ScalarExpr::exact(quantumforge::analytic::ExactComplex::real(
                Rational::new(2, 1).unwrap(),
            )),
            a.clone(),
            b.clone(),
        ])
        .unwrap(),
        ScalarExpr::pow(b.clone(), 2).unwrap(),
    ])
    .unwrap();
    let result = ScalarExpr::verify_identity(&lhs, &rhs).unwrap();
    println!("(a+b)^2 = a^2 + 2ab + b^2  =>  {}", result);

    // Identity 2: (a-b)(a+b) = a^2 - b^2
    let lhs2 = ScalarExpr::mul([
        ScalarExpr::add([a.clone(), ScalarExpr::mul([neg_one(), b.clone()]).unwrap()]).unwrap(),
        ScalarExpr::add([a.clone(), b.clone()]).unwrap(),
    ])
    .unwrap();
    let rhs2 = ScalarExpr::add([
        ScalarExpr::pow(a.clone(), 2).unwrap(),
        ScalarExpr::mul([neg_one(), ScalarExpr::pow(b.clone(), 2).unwrap()]).unwrap(),
    ])
    .unwrap();
    let result2 = ScalarExpr::verify_identity(&lhs2, &rhs2).unwrap();
    println!("(a-b)(a+b) = a^2 - b^2  =>  {}", result2);

    // Identity 3: a^3 - b^3 = (a-b)(a^2 + ab + b^2)
    let lhs3 = ScalarExpr::add([
        ScalarExpr::pow(a.clone(), 3).unwrap(),
        ScalarExpr::mul([neg_one(), ScalarExpr::pow(b.clone(), 3).unwrap()]).unwrap(),
    ])
    .unwrap();
    let rhs3 = ScalarExpr::mul([
        ScalarExpr::add([a.clone(), ScalarExpr::mul([neg_one(), b.clone()]).unwrap()]).unwrap(),
        ScalarExpr::add([
            ScalarExpr::pow(a.clone(), 2).unwrap(),
            ScalarExpr::mul([a.clone(), b.clone()]).unwrap(),
            ScalarExpr::pow(b.clone(), 2).unwrap(),
        ])
        .unwrap(),
    ])
    .unwrap();
    let result3 = ScalarExpr::verify_identity(&lhs3, &rhs3).unwrap();
    println!("a^3 - b^3 = (a-b)(a^2+ab+b^2)  =>  {}", result3);

    // Identity 4: Non-polynomial expressions are rejected (hard gate)
    let trig_lhs = ScalarExpr::sin(a.clone());
    let trig_rhs = ScalarExpr::sin(a.clone());
    match ScalarExpr::verify_identity(&trig_lhs, &trig_rhs) {
        Err(e) => println!(
            "sin(a) = sin(a)  =>  Err({}) (non-polynomial, correct rejection)",
            e
        ),
        Ok(_) => println!("UNEXPECTED: trig accepted"),
    }

    // CSE demonstration
    let expr = ScalarExpr::add([
        ScalarExpr::mul([a.clone(), b.clone()]).unwrap(),
        ScalarExpr::mul([a.clone(), b.clone()]).unwrap(),
        ScalarExpr::pow(a.clone(), 2).unwrap(),
    ])
    .unwrap();
    let cse_result = expr.cse();
    println!("\nCSE of a*b + a*b + a^2:");
    println!("  original: {}", expr);
    println!("  CSE'd:    {:?}", cse_result);

    println!("\n=== All identities verified exactly (no floating point) ===");
}

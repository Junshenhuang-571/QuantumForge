//! Tests for `numeric_ir`: the typed SSA lowering of `ScalarExpr`.
//!
//! The module docs state that `ScalarExpr::evaluate` and
//! `NumericProgram::evaluate` "must agree; if you change one, update the other
//! and add a consistency test". That test did not exist, and the module sat at
//! 3/183 lines covered. This file is it.

use std::collections::HashMap;

use quantumforge::Complex64;
use quantumforge::analytic::{ExactComplex, Rational, ScalarExpr};
use quantumforge::numeric_ir::{NumericBuilder, NumericIrError, NumericProgram};

fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::add([a, b]).expect("add")
}

fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::mul([a, b]).expect("mul")
}

fn sub(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::subtract(a, b).expect("subtract")
}

/// `ScalarExpr` has no division; the inverse is `pow(b, -1)`.
fn div(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    mul(a, ScalarExpr::pow(b, -1).expect("inverse"))
}

/// A real exact constant as a `ScalarExpr`.
fn exact(value: i128) -> ScalarExpr {
    ScalarExpr::exact(ExactComplex::real(
        Rational::new(value, 1).expect("non-zero denominator"),
    ))
}

/// The module's central claim: lowering then interpreting must match interpreting
/// the symbolic expression directly.
#[test]
fn numeric_ir_agrees_with_scalar_expr() {
    let x = ScalarExpr::symbol("x");
    let y = ScalarExpr::symbol("y");
    let expressions = vec![
        add(x.clone(), y.clone()),
        mul(x.clone(), y.clone()),
        sub(x.clone(), y.clone()),
        div(add(x.clone(), y.clone()), sub(x.clone(), y.clone())),
        add(mul(x.clone(), x.clone()), mul(y.clone(), y.clone())),
        ScalarExpr::sqrt(add(mul(x.clone(), x.clone()), exact(1))),
        ScalarExpr::sin(x.clone()),
        ScalarExpr::cos(x.clone()),
        ScalarExpr::exp(x.clone()),
        ScalarExpr::abs(sub(x.clone(), y.clone())),
        add(
            ScalarExpr::pow(x.clone(), 3).expect("pow"),
            ScalarExpr::pow(y.clone(), 2).expect("pow"),
        ),
        ScalarExpr::log(add(mul(x.clone(), x.clone()), exact(1))),
        // sin^2 + cos^2 == 1, a good check that both paths agree numerically.
        add(
            mul(ScalarExpr::sin(x.clone()), ScalarExpr::sin(x.clone())),
            mul(ScalarExpr::cos(x.clone()), ScalarExpr::cos(x.clone())),
        ),
    ];

    let points = [
        vec![0.5, 0.25],
        vec![1.0, 2.0],
        vec![-0.75, 3.5],
        vec![2.5, -1.25],
    ];

    for expression in &expressions {
        let program =
            NumericProgram::compile_scalar(expression, &["x", "y"]).expect("lowers to real IR");
        assert_eq!(program.input_count(), 2);
        assert!(program.instruction_count() > 0);
        for point in &points {
            let mut values = HashMap::new();
            values.insert("x".to_string(), point[0]);
            values.insert("y".to_string(), point[1]);
            let symbolic = expression
                .eval_f64(&values)
                .expect("evaluates symbolically");
            let numeric = program.evaluate(point).expect("evaluates numerically");
            assert!(
                (symbolic - numeric).abs() < 1e-12,
                "{expression} at {point:?}: symbolic {symbolic} vs numeric {numeric}"
            );
        }
    }
}

/// A single-symbol expression, so the input order is unambiguous.
#[test]
fn single_symbol_lowering() {
    let expression = add(
        mul(ScalarExpr::symbol("t"), ScalarExpr::symbol("t")),
        exact(1),
    );
    let program = NumericProgram::compile_scalar(&expression, &["t"]).expect("lowers");
    assert_eq!(program.input_count(), 1);
    let mut values = HashMap::new();
    values.insert("t".to_string(), 3.0);
    let symbolic = expression.eval_f64(&values).expect("evaluates");
    let numeric = program.evaluate(&[3.0]).expect("evaluates");
    assert!((symbolic - numeric).abs() < 1e-12);
    assert_eq!(numeric, 10.0);
}

#[test]
fn builder_emits_a_readable_instruction_stream() {
    // (2 + x) * 3
    let mut builder = NumericBuilder::new(1);
    let x = builder.input(0).expect("input");
    let two = builder.constant(2.0).expect("constant");
    let sum = builder.add(two, x).expect("add");
    let three = builder.constant(3.0).expect("constant");
    let product = builder.multiply(sum, three).expect("multiply");
    let program = builder.finish(product).expect("finish");

    assert_eq!(program.instruction_count(), 5);
    assert_eq!(program.input_count(), 1);
    assert_eq!(program.output().index(), 4);
    // `Register` has no public constructor, so compare the debug form rather
    // than building expected values.
    assert_eq!(
        format!("{:?}", program.instructions()),
        "[Input(0), Constant(2.0), Add(Register(1), Register(0)), \
         Constant(3.0), Multiply(Register(2), Register(3))]"
    );
    assert_eq!(program.evaluate(&[4.0]).expect("evaluates"), 18.0);
}

#[test]
fn builder_unary_and_binary_operations() {
    let mut builder = NumericBuilder::new(1);
    let x = builder.input(0).expect("input");
    let squared = builder.multiply(x, x).expect("square");
    let rooted = builder.sqrt(squared).expect("sqrt");
    let negated = builder.negate(rooted).expect("negate");
    let program = builder.finish(negated).expect("finish");
    assert_eq!(program.evaluate(&[4.0]).expect("evaluates"), -4.0);

    let mut builder = NumericBuilder::new(2);
    let a = builder.input(0).expect("a");
    let b = builder.input(1).expect("b");
    let quotient = builder.divide(a, b).expect("divide");
    let difference = builder.subtract(quotient, b).expect("subtract");
    let program = builder.finish(difference).expect("finish");
    assert_eq!(program.evaluate(&[6.0, 2.0]).expect("evaluates"), 1.0);

    let mut builder = NumericBuilder::new(1);
    let x = builder.input(0).expect("input");
    let powered = builder.powi(x, 4).expect("powi");
    let program = builder.finish(powered).expect("finish");
    assert_eq!(program.evaluate(&[3.0]).expect("evaluates"), 81.0);
}

#[test]
fn error_cases_are_rejected_not_guessed() {
    let expression = add(ScalarExpr::symbol("x"), ScalarExpr::symbol("y"));
    let program = NumericProgram::compile_scalar(&expression, &["x", "y"]).expect("lowers");

    // Wrong input count.
    assert!(matches!(
        program.evaluate(&[1.0]),
        Err(NumericIrError::InvalidInputCount {
            expected: 2,
            actual: 1
        })
    ));
    // Non-finite input.
    assert!(matches!(
        program.evaluate(&[f64::NAN, 1.0]),
        Err(NumericIrError::NonFiniteValue)
    ));
    // Division by zero is rejected rather than returning infinity. Built with
    // the builder's `divide`, which lowers to `NumericInstruction::Divide` — the
    // symbolic `pow(y, -1)` route lowers to `PowerI` and yields `inf` instead.
    let mut builder = NumericBuilder::new(2);
    let a = builder.input(0).expect("a");
    let b = builder.input(1).expect("b");
    let quotient = builder.divide(a, b).expect("divide");
    let program = builder.finish(quotient).expect("finish");
    assert!(matches!(
        program.evaluate(&[1.0, 0.0]),
        Err(NumericIrError::DivisionByZero)
    ));
    // Overflow to infinity mid-program.
    let big = mul(ScalarExpr::symbol("x"), ScalarExpr::symbol("x"));
    let program = NumericProgram::compile_scalar(&big, &["x"]).expect("lowers");
    assert!(matches!(
        program.evaluate(&[1e308]),
        Err(NumericIrError::NonFiniteValue)
    ));

    // Unknown symbol in the expression.
    let stray = ScalarExpr::symbol("z");
    assert!(matches!(
        NumericProgram::compile_scalar(&stray, &["x"]),
        Err(NumericIrError::UnknownSymbol(name)) if name == "z"
    ));
    // Duplicate symbol in the input order.
    assert!(matches!(
        NumericProgram::compile_scalar(&expression, &["x", "x"]),
        Err(NumericIrError::DuplicateInputSymbol(name)) if name == "x"
    ));
    // Complex-valued constants cannot be lowered to real IR.
    let complex = ScalarExpr::exact(ExactComplex {
        re: Rational::new(1, 1).expect("ok"),
        im: Rational::new(1, 1).expect("ok"),
    });
    assert!(matches!(
        NumericProgram::compile_scalar(&complex, &[]),
        Err(NumericIrError::ComplexValueUnsupported)
    ));
    // `UnsupportedScalarExpression` is deliberately NOT asserted: `ScalarExpr`
    // is a closed enum of thirteen variants and `lower_scalar` handles all
    // thirteen, so no expression can reach it. See the variant's docs.
    // Out-of-range input index in the builder.
    let mut builder = NumericBuilder::new(1);
    assert!(matches!(
        builder.input(1),
        Err(NumericIrError::InputOutOfRange(1))
    ));
}

#[test]
fn constants_need_no_inputs() {
    let program = NumericProgram::compile_scalar(&exact(7), &[]).expect("lowers");
    assert_eq!(program.input_count(), 0);
    assert_eq!(program.evaluate(&[]).expect("evaluates"), 7.0);
    assert!(matches!(
        program.evaluate(&[1.0]),
        Err(NumericIrError::InvalidInputCount {
            expected: 0,
            actual: 1
        })
    ));
}

/// The symbolic and numeric paths must also agree on `evaluate` (complex),
/// not just `eval_f64`.
#[test]
fn complex_evaluate_agrees_too() {
    let expression = add(
        mul(ScalarExpr::symbol("x"), ScalarExpr::symbol("x")),
        exact(2),
    );
    let program = NumericProgram::compile_scalar(&expression, &["x"]).expect("lowers");
    let mut values = HashMap::new();
    values.insert("x".to_string(), Complex64::new(1.5, 0.0));
    let symbolic = expression.evaluate(&values).expect("evaluates");
    let numeric = program.evaluate(&[1.5]).expect("evaluates");
    assert!((symbolic.re - numeric).abs() < 1e-12);
    assert_eq!(numeric, 4.25);
}

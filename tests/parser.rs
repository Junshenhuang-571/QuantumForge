use quantumforge::Complex64;
use quantumforge::analytic::{ExactComplex, Rational, ScalarExpr, Simplifier};
use quantumforge::parser::{parse_circuit, parse_scalar_expr};
use quantumforge::{DensityMatrix, NoiseChannel, StateVector};
use std::collections::HashMap;

#[test]
fn parser_preserves_exact_rationals_and_operator_precedence() {
    let expression = parse_scalar_expr("1/3 + 0.25").unwrap();
    assert_eq!(
        expression,
        ScalarExpr::exact(ExactComplex::real(Rational::new(7, 12).unwrap()))
    );
    let powered = parse_scalar_expr("2^3 + 1").unwrap();
    assert_eq!(
        Simplifier::default().simplify(&powered).unwrap(),
        ScalarExpr::exact(ExactComplex::real(Rational::new(9, 1).unwrap()))
    );
    let negative_power = parse_scalar_expr("2^-3").unwrap();
    assert_eq!(
        Simplifier::default().simplify(&negative_power).unwrap(),
        ScalarExpr::exact(ExactComplex::real(Rational::new(1, 8).unwrap()))
    );

    let expression = parse_scalar_expr("-x^2 + 2 x").unwrap();
    assert_eq!(expression.to_string(), "-1 * x^2 + 2 * x");
}

#[test]
fn parser_supports_complex_exact_arithmetic_and_wolfram_heads() {
    let quotient = parse_scalar_expr("(1+I)/(1-I)").unwrap();
    assert_eq!(quotient, ScalarExpr::exact(ExactComplex::I));

    let trig_identity = parse_scalar_expr("Sin[Pi/4]^2 + Cos[Pi/4]^2").unwrap();
    let simplified = Simplifier::default().simplify(&trig_identity).unwrap();
    let value = simplified.evaluate(&HashMap::new()).unwrap();
    assert!((value.re - 1.0).abs() < 1e-12);
    assert!(value.im.abs() < 1e-12);

    let expression = parse_scalar_expr("Log[Exp[x]]").unwrap();
    assert!(matches!(expression, ScalarExpr::Log(_)));
    assert!(parse_scalar_expr("E + Pi + I").is_ok());
}

#[test]
fn parser_reports_malformed_or_unsupported_input_with_offsets() {
    for source in ["", "(1+2", "Sin[]", "F[x]", "2^1.5", "1/0", "1.2.3", "x+"] {
        let error = parse_scalar_expr(source).unwrap_err();
        assert!(error.offset() <= source.len(), "{source:?}: {error}");
        assert!(!error.message().is_empty());
    }
    let error = parse_scalar_expr("@").unwrap_err();
    assert_eq!(error.offset(), 0);
    assert_eq!(error.message(), "unsupported character '@'");
}

#[test]
fn parser_evaluates_complex_functions_with_the_existing_scalar_semantics() {
    let expression = parse_scalar_expr("Exp[I Pi]").unwrap();
    let value = expression.evaluate(&HashMap::new()).unwrap();
    assert!((value.re + 1.0).abs() < 1e-12);
    assert!(value.im.abs() < 1e-12);
    let expression = parse_scalar_expr("Sin[I]").unwrap();
    let value: Complex64 = expression.evaluate(&HashMap::new()).unwrap();
    assert!((value.im - 1.0_f64.sinh()).abs() < 1e-12);
}

#[test]
fn parser_builds_seeded_measurement_and_feedforward_circuits() {
    let source = "Circuit[H[0], Measure[0,0], IfBit[0,1,X,1], Measure[1,1]]";
    let circuit = parse_circuit(source).unwrap();
    let state = StateVector::zero(2).unwrap();
    let first = circuit.run_shots(&state, 256, 731).unwrap();
    let second = circuit.run_shots(&state, 256, 731).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.counts.values().sum::<usize>(), 256);
    assert_eq!(first.counts.len(), 2);
    assert!(first.counts.contains_key(&vec![0, 0]));
    assert!(first.counts.contains_key(&vec![1, 1]));
}

#[test]
fn parser_handles_rotation_angles_entanglers_and_density_channels() {
    let circuit =
        parse_circuit("Circuit[Ry[Pi/2,0], CNOT[0,1], AmplitudeDamping[1,0.25]]").unwrap();
    let initial = DensityMatrix::from_state(&StateVector::zero(2).unwrap());
    let output = circuit.run_density(initial).unwrap();
    assert!((output.trace().re - 1.0).abs() < 1e-12);
    assert!(matches!(
        circuit.operations().last(),
        Some(quantumforge::CircuitOp::Channel {
            channel: NoiseChannel::AmplitudeDamping { gamma },
            target: 1
        }) if (*gamma - 0.25).abs() < f64::EPSILON
    ));
}

#[test]
fn circuit_parser_reports_invalid_operation_syntax() {
    for source in [
        "",
        "Circuit[X[]]",
        "Circuit[CNOT[0]]",
        "Circuit[RX[0.5,0]]",
        "Circuit[Measure[0,0], IfBit[0,2,X,1]]",
        "Circuit[H[0],]",
        "Circuit[H[0]",
    ] {
        let error = parse_circuit(source).unwrap_err();
        assert!(error.offset() <= source.len(), "{source:?}: {error}");
        assert!(!error.message().is_empty());
    }
}

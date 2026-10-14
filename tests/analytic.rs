use quantumforge::Complex64;
use quantumforge::analytic::{
    AnalyticError, Assumptions, ExactComplex, ExprPattern, NamedConstant, NumericPauliError, Pauli,
    PauliSum, PauliWord, Rational, RewriteRule, ScalarExpr, Simplifier, SymbolicPauliSum,
    UnboundSymbol,
};
use quantumforge::quantum::{HilbertSpace, Ket, SparseKet};
use std::collections::HashMap;

#[test]
fn pauli_commutator_is_exact() {
    let x = PauliSum::term(ExactComplex::ONE, PauliWord::new(vec![Pauli::X])).unwrap();
    let y = PauliSum::term(ExactComplex::ONE, PauliWord::new(vec![Pauli::Y])).unwrap();
    let expected = PauliSum::term(
        ExactComplex {
            re: Rational::ZERO,
            im: Rational::new(2, 1).unwrap(),
        },
        PauliWord::new(vec![Pauli::Z]),
    )
    .unwrap();
    assert_eq!(x.commutator(&y).unwrap(), expected);
}

#[test]
fn square_root_simplification_respects_real_and_positive_assumptions() {
    let x = ScalarExpr::symbol("x");
    let expression = ScalarExpr::sqrt(ScalarExpr::pow(x.clone(), 2).unwrap());
    let mut real = Assumptions::default();
    real.set_real("x");
    assert_eq!(
        Simplifier::new(real).simplify(&expression).unwrap(),
        ScalarExpr::abs(x.clone())
    );
    let mut positive = Assumptions::default();
    positive.set_positive("x");
    assert_eq!(Simplifier::new(positive).simplify(&expression).unwrap(), x);
}

#[test]
fn exact_pauli_sum_compiles_to_sparse_operator_and_matches_state_action() {
    let two = Rational::new(2, 1).unwrap();
    let half = Rational::new(1, 2).unwrap();
    let mut hamiltonian = PauliSum::zero(2);
    hamiltonian
        .add_term(
            ExactComplex::real(half),
            PauliWord::new(vec![Pauli::Y, Pauli::X]),
        )
        .unwrap();
    hamiltonian
        .add_term(
            ExactComplex::real(two),
            PauliWord::new(vec![Pauli::Z, Pauli::I]),
        )
        .unwrap();
    let operator = hamiltonian.to_sparse_operator().unwrap();
    let dense = operator.to_dense().unwrap();
    let space = HilbertSpace::qubits(2).unwrap();
    for basis in 0..space.dimension() {
        let ket = Ket::basis(space.clone(), basis).unwrap();
        let actual = operator.apply(&ket).unwrap();
        let expected = dense.apply(&ket).unwrap();
        assert_eq!(actual, expected);
    }

    let state = Ket::new(
        space.clone(),
        vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.0, 0.5),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.0, -0.5),
        ],
    )
    .unwrap();
    let sparse_state = SparseKet::from_ket(&state);
    assert_eq!(
        hamiltonian.expectation_ket(&state).unwrap(),
        state.inner(&dense.apply(&state).unwrap()).unwrap()
    );
    assert_eq!(
        hamiltonian.expectation_sparse_ket(&sparse_state).unwrap(),
        hamiltonian.expectation_ket(&state).unwrap()
    );
    hamiltonian.to_sparse_observable(1e-12).unwrap();
}

#[test]
fn pauli_y_action_uses_standard_complex_phase_convention() {
    let space = HilbertSpace::qubits(1).unwrap();
    let y = PauliWord::new(vec![Pauli::Y]);
    let y_zero = y.apply_ket(&Ket::basis(space.clone(), 0).unwrap()).unwrap();
    assert_eq!(y_zero.amplitudes()[0], Complex64::ZERO);
    assert_eq!(y_zero.amplitudes()[1], Complex64::I);
    let y_one = y.apply_ket(&Ket::basis(space, 1).unwrap()).unwrap();
    assert_eq!(y_one.amplitudes()[0], Complex64::new(0.0, -1.0));
    assert_eq!(y_one.amplitudes()[1], Complex64::ZERO);
}

#[test]
fn symbolic_pauli_sum_evaluates_coefficients_and_validates_observables() {
    let mut hamiltonian = SymbolicPauliSum::zero(2);
    hamiltonian
        .add_term(
            ScalarExpr::symbol("g"),
            PauliWord::new(vec![Pauli::X, Pauli::I]),
        )
        .unwrap();
    hamiltonian
        .add_term(
            ScalarExpr::exact(ExactComplex::real(Rational::new(3, 2).unwrap())),
            PauliWord::new(vec![Pauli::I, Pauli::Z]),
        )
        .unwrap();
    let mut bindings = HashMap::new();
    bindings.insert("g".to_string(), Complex64::new(0.25, 0.0));
    let observable = hamiltonian
        .evaluate_sparse_observable(&bindings, 1e-12)
        .unwrap();
    let state = Ket::new(
        HilbertSpace::qubits(2).unwrap(),
        vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
        ],
    )
    .unwrap();
    let expected = state
        .expectation(
            &quantumforge::quantum::Observable::new(
                observable.operator().to_dense().unwrap(),
                1e-12,
            )
            .unwrap(),
        )
        .unwrap();
    assert!((expected.re - 0.25).abs() < 1e-12);
    assert!(expected.im.abs() < 1e-12);

    assert_eq!(
        hamiltonian
            .evaluate_sparse_operator(&HashMap::new())
            .unwrap_err(),
        NumericPauliError::UnboundSymbol(UnboundSymbol("g".to_string()))
    );
    bindings.insert("g".to_string(), Complex64::I);
    assert_eq!(
        hamiltonian
            .evaluate_sparse_observable(&bindings, 1e-12)
            .unwrap_err(),
        NumericPauliError::LinearAlgebra(quantumforge::quantum::LinearError::NotHermitian)
    );
}

#[test]
fn named_constants_and_complex_transcendentals_evaluate() {
    let values = HashMap::new();
    assert_eq!(
        ScalarExpr::pi().evaluate(&values).unwrap(),
        Complex64::new(std::f64::consts::PI, 0.0)
    );
    assert_eq!(
        ScalarExpr::e().evaluate(&values).unwrap(),
        Complex64::new(std::f64::consts::E, 0.0)
    );
    let imaginary_unit = ScalarExpr::exact(ExactComplex::I);
    let sine = ScalarExpr::sin(imaginary_unit.clone())
        .evaluate(&values)
        .unwrap();
    assert!(sine.re.abs() < 1e-14);
    assert!((sine.im - 1.0_f64.sinh()).abs() < 1e-14);
    let cosine = ScalarExpr::cos(imaginary_unit).evaluate(&values).unwrap();
    assert!((cosine.re - 1.0_f64.cosh()).abs() < 1e-14);
    assert!(cosine.im.abs() < 1e-14);
    let euler_phase = ScalarExpr::exp(
        ScalarExpr::mul([ScalarExpr::exact(ExactComplex::I), ScalarExpr::pi()]).unwrap(),
    )
    .evaluate(&values)
    .unwrap();
    assert!((euler_phase.re + 1.0).abs() < 1e-14);
    assert!(euler_phase.im.abs() < 1e-14);
}

#[test]
fn elementary_simplifier_handles_special_angles_parity_and_exp_log() {
    let simplifier = Simplifier::default();
    let rational = |n, d| ScalarExpr::exact(ExactComplex::real(Rational::new(n, d).unwrap()));
    let angle = |n, d| ScalarExpr::mul([rational(n, d), ScalarExpr::pi()]).unwrap();
    assert_eq!(
        simplifier.simplify(&ScalarExpr::sin(angle(1, 1))).unwrap(),
        ScalarExpr::zero()
    );
    assert_eq!(
        simplifier.simplify(&ScalarExpr::cos(angle(1, 1))).unwrap(),
        rational(-1, 1)
    );
    assert_eq!(
        simplifier.simplify(&ScalarExpr::sin(angle(1, 2))).unwrap(),
        ScalarExpr::one()
    );
    assert_eq!(
        simplifier.simplify(&ScalarExpr::cos(angle(1, 2))).unwrap(),
        ScalarExpr::zero()
    );
    let half_sqrt_two =
        ScalarExpr::mul([rational(1, 2), ScalarExpr::sqrt(rational(2, 1))]).unwrap();
    assert_eq!(
        simplifier.simplify(&ScalarExpr::sin(angle(1, 4))).unwrap(),
        half_sqrt_two
    );
    assert_eq!(
        simplifier.simplify(&ScalarExpr::cos(angle(3, 4))).unwrap(),
        ScalarExpr::mul([rational(-1, 1), half_sqrt_two.clone()]).unwrap()
    );
    assert_eq!(
        simplifier
            .simplify(&ScalarExpr::sin(
                ScalarExpr::mul([rational(-1, 1), ScalarExpr::symbol("x"),]).unwrap()
            ))
            .unwrap(),
        ScalarExpr::mul([rational(-1, 1), ScalarExpr::sin(ScalarExpr::symbol("x")),]).unwrap()
    );
    assert_eq!(
        simplifier
            .simplify(&ScalarExpr::exp(ScalarExpr::zero()))
            .unwrap(),
        ScalarExpr::one()
    );
    assert_eq!(
        simplifier
            .simplify(&ScalarExpr::exp(ScalarExpr::one()))
            .unwrap(),
        ScalarExpr::e()
    );
    assert_eq!(
        simplifier
            .simplify(&ScalarExpr::log(ScalarExpr::e()))
            .unwrap(),
        ScalarExpr::one()
    );
    assert_eq!(NamedConstant::Pi.to_string(), "Pi");

    let x = ScalarExpr::symbol("real_x");
    let mut assumptions = Assumptions::default();
    assumptions.set_real("real_x");
    assert_eq!(
        Simplifier::new(assumptions)
            .simplify(&ScalarExpr::log(ScalarExpr::exp(x.clone())))
            .unwrap(),
        x
    );
}

#[test]
fn transcendental_nodes_participate_in_structural_rewrites() {
    let mut simplifier = Simplifier::default();
    simplifier.add_rule(RewriteRule::new(
        ExprPattern::Sin(Box::new(ExprPattern::Any("x".to_string()))),
        ScalarExpr::pow(ScalarExpr::symbol("x"), 2).unwrap(),
    ));
    assert_eq!(
        simplifier
            .simplify(&ScalarExpr::sin(ScalarExpr::symbol("theta")))
            .unwrap(),
        ScalarExpr::pow(ScalarExpr::symbol("theta"), 2).unwrap()
    );
}

#[test]
fn symbolic_differentiation_handles_chain_product_and_power_rules() {
    let x = ScalarExpr::symbol("x");
    let expression = ScalarExpr::add([
        ScalarExpr::mul([ScalarExpr::sin(x.clone()), ScalarExpr::exp(x.clone())]).unwrap(),
        ScalarExpr::pow(x.clone(), 3).unwrap(),
    ])
    .unwrap();
    let derivative = expression.differentiate("x").unwrap();
    let value = 0.37;
    let mut bindings = HashMap::new();
    bindings.insert("x".to_string(), Complex64::new(value, 0.0));
    let derivative_value = derivative.evaluate(&bindings).unwrap().re;
    let expected = value.exp() * (value.sin() + value.cos()) + 3.0 * value * value;
    assert!((derivative_value - expected).abs() < 1e-12);

    let epsilon = 1e-6;
    let evaluate_at = |x_value| {
        let mut local = HashMap::new();
        local.insert("x".to_string(), Complex64::new(x_value, 0.0));
        expression.evaluate(&local).unwrap().re
    };
    let finite_difference =
        (evaluate_at(value + epsilon) - evaluate_at(value - epsilon)) / (2.0 * epsilon);
    assert!((derivative_value - finite_difference).abs() < 1e-8);

    let log_derivative = ScalarExpr::log(x.clone()).differentiate("x").unwrap();
    bindings.insert("x".to_string(), Complex64::new(2.0, 0.0));
    assert!((log_derivative.evaluate(&bindings).unwrap().re - 0.5).abs() < 1e-12);
    let sqrt_derivative = ScalarExpr::sqrt(x.clone()).differentiate("x").unwrap();
    bindings.insert("x".to_string(), Complex64::new(4.0, 0.0));
    assert!((sqrt_derivative.evaluate(&bindings).unwrap().re - 0.25).abs() < 1e-12);
    assert_eq!(
        ScalarExpr::abs(x.clone()).differentiate("x"),
        Err(AnalyticError::NondifferentiableAbsoluteValue)
    );
    let mut positive = Assumptions::default();
    positive.set_positive("x");
    let smooth_absolute = Simplifier::new(positive)
        .simplify(&ScalarExpr::abs(x))
        .unwrap();
    assert_eq!(
        smooth_absolute.differentiate("x").unwrap(),
        ScalarExpr::one()
    );
}

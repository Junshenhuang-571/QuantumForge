//! Cross-validation of the gradient strategies.
//!
//! Every exact rule is checked against every other exact rule, and the
//! finite-difference rule is checked to converge at its expected order. Rules
//! that only agree with themselves prove nothing.

use quantumforge::gradients::{
    CircuitProfile, GradientOptions, GradientStrategy, backprop_gradient_with_inputs,
    finite_difference_gradient_with_inputs,
};
use quantumforge::{ParameterizedCircuit, RotationAxis, SingleQubitObservable};

fn observable() -> (SingleQubitObservable, usize) {
    (SingleQubitObservable::Z, 0)
}

/// `Ry(theta)` on one qubit, `<Z> = cos(theta)`, so `d/dtheta = -sin(theta)`.
#[test]
fn ry_gradient_matches_the_analytic_derivative() {
    let mut circuit = ParameterizedCircuit::new(1);
    circuit.rotation(RotationAxis::Y, 0, 0);
    let (obs, qubit) = observable();
    let theta = 0.7;

    for strategy in [
        GradientStrategy::ParameterShift,
        GradientStrategy::Adjoint,
        GradientStrategy::Backprop,
    ] {
        let estimate = circuit
            .gradient(
                &GradientOptions {
                    strategy,
                    ..GradientOptions::default()
                },
                &[theta],
                1,
                obs,
                qubit,
            )
            .unwrap();
        let expected = -theta.sin();
        assert!(
            (estimate.gradient[0] - expected).abs() < 1e-10,
            "{strategy:?} gave {} for d<Z>/dtheta at theta={theta}, expected {expected}",
            estimate.gradient[0]
        );
    }
}

#[test]
fn all_exact_strategies_agree() {
    let (obs, qubit) = observable();
    let mut circuit = ParameterizedCircuit::new(42);
    let mut params = Vec::new();
    for layer in 0..3 {
        for q in 0..7 {
            circuit.rotation(RotationAxis::Y, q, params.len());
            params.push(0.3 + 0.1 * layer as f64);
        }
        for q in 0..7 {
            circuit.rotation(RotationAxis::X, q, params.len());
            params.push(-0.2 + 0.05 * q as f64);
        }
        for q in 0..6 {
            circuit.cnot(q, q + 1);
        }
    }

    let reference = circuit
        .gradient(
            &GradientOptions {
                strategy: GradientStrategy::ParameterShift,
                ..GradientOptions::default()
            },
            &params,
            7,
            obs,
            qubit,
        )
        .unwrap();
    assert_eq!(reference.gradient.len(), params.len());

    for strategy in [GradientStrategy::Adjoint, GradientStrategy::Backprop] {
        let estimate = circuit
            .gradient(
                &GradientOptions {
                    strategy,
                    ..GradientOptions::default()
                },
                &params,
                7,
                obs,
                qubit,
            )
            .unwrap();
        for (index, (got, want)) in estimate
            .gradient
            .iter()
            .zip(&reference.gradient)
            .enumerate()
        {
            assert!(
                (got - want).abs() < 1e-9,
                "{strategy:?} component {index}: {got} vs parameter-shift {want}"
            );
        }
    }
}

#[test]
fn backprop_free_function_matches_the_method() {
    let (obs, qubit) = observable();
    let mut circuit = ParameterizedCircuit::new(2);
    circuit.rotation(RotationAxis::Y, 0, 0);
    circuit.rotation(RotationAxis::Z, 1, 1);
    circuit.cnot(0, 1);
    circuit.rotation(RotationAxis::X, 0, 0);
    let params = [0.4, -1.1];
    let inputs = [];

    let free = backprop_gradient_with_inputs(&circuit, &inputs, &params, 2, obs, qubit).unwrap();
    let method = circuit
        .gradient(
            &GradientOptions {
                strategy: GradientStrategy::Backprop,
                ..GradientOptions::default()
            },
            &params,
            2,
            obs,
            qubit,
        )
        .unwrap();
    assert_eq!(free, method.gradient);
}

#[test]
fn shared_parameter_sums_occurrence_wise_derivatives() {
    // One parameter drives two rotations; the gradient must be the sum of the
    // two occurrence-wise derivatives, not the derivative of either alone.
    let (obs, qubit) = (SingleQubitObservable::Z, 1);
    let mut circuit = ParameterizedCircuit::new(1);
    circuit.rotation(RotationAxis::Y, 0, 0);
    circuit.rotation(RotationAxis::Y, 1, 0); // same parameter index
    circuit.cnot(0, 1);
    let theta = 0.9;

    let shift = circuit
        .parameter_shift_gradient(&[theta], 2, qubit)
        .unwrap();
    let backprop = circuit.adjoint_gradient(&[theta], 2, obs, qubit).unwrap();
    assert!(
        (shift[0] - backprop[0]).abs() < 1e-10,
        "shared-parameter gradient disagree: {shift:?} vs {backprop:?}"
    );

    // A single Ry on qubit 0 with the same observable would give -sin(theta);
    // entangling qubit 1 must change it, which is what makes this a real test
    // of the two-occurrence sum.
    let mut single = ParameterizedCircuit::new(1);
    single.rotation(RotationAxis::Y, 0, 0);
    single.cnot(0, 1);
    let single = single.parameter_shift_gradient(&[theta], 2, qubit).unwrap();
    assert!(
        (shift[0] - single[0]).abs() > 1e-3,
        "the second occurrence should contribute: {} vs {}",
        shift[0],
        single[0]
    );
}

#[test]
fn finite_difference_converges_at_second_order() {
    let (obs, qubit) = observable();
    let mut circuit = ParameterizedCircuit::new(2);
    circuit.rotation(RotationAxis::Y, 0, 0);
    circuit.rotation(RotationAxis::X, 1, 1);
    circuit.cnot(0, 1);
    circuit.rotation(RotationAxis::Y, 2, 0); // shared parameter
    let params = [0.5, -0.4];

    let exact = circuit.parameter_shift_gradient(&params, 3, qubit).unwrap();
    let mut previous_error = f64::INFINITY;
    for &step in &[1e-2, 1e-3, 1e-4] {
        let approximate =
            finite_difference_gradient_with_inputs(&circuit, &[], &params, 3, obs, qubit, step)
                .unwrap();
        let error = approximate
            .iter()
            .zip(&exact)
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f64, f64::max);
        assert!(
            error * 50.0 < previous_error,
            "shrinking the step 10x should cut the error ~100x: step {step} error {error}"
        );
        previous_error = error;
    }
    assert!(previous_error < 1e-8, "final error {previous_error}");
}

#[test]
fn finite_difference_step_must_be_positive_and_finite() {
    let mut circuit = ParameterizedCircuit::new(1);
    circuit.rotation(RotationAxis::Y, 0, 0);
    let (obs, qubit) = observable();
    for &step in &[0.0, -1.0, f64::NAN, f64::INFINITY] {
        let error =
            finite_difference_gradient_with_inputs(&circuit, &[], &[0.3], 1, obs, qubit, step)
                .unwrap_err();
        match error {
            quantumforge::QuantumError::InvalidFiniteDifferenceStep(got) => {
                assert!(got.is_nan() == step.is_nan() && (got.is_nan() || got == step));
            }
            other => panic!("expected InvalidFiniteDifferenceStep, got {other:?}"),
        }
    }
}

#[test]
fn auto_selects_the_cheapest_applicable_rule() {
    let mut circuit = ParameterizedCircuit::new(1);
    circuit.rotation(RotationAxis::Y, 0, 0);
    circuit.cnot(0, 1);

    let exact_profile = circuit.circuit_profile();
    assert_eq!(exact_profile.operations, 2);
    assert_eq!(exact_profile.parameters, 1);
    assert_eq!(exact_profile.parameterized_operations, 1);
    assert!(!exact_profile.has_channel);
    assert_eq!(
        GradientStrategy::select(&exact_profile),
        Some(GradientStrategy::Adjoint)
    );

    let mut sampled = exact_profile;
    sampled.shots = 100;
    assert_eq!(
        GradientStrategy::select(&sampled),
        Some(GradientStrategy::ParameterShift)
    );

    let mut noisy = exact_profile;
    noisy.has_channel = true;
    assert_eq!(GradientStrategy::select(&noisy), None);
    assert!(
        circuit
            .gradient(
                &GradientOptions {
                    strategy: GradientStrategy::Auto,
                    ..GradientOptions::default()
                },
                &[0.3],
                2,
                SingleQubitObservable::Z,
                0,
            )
            .is_ok()
    );
}

#[test]
fn auto_rejects_circuits_with_a_channel() {
    let mut circuit = ParameterizedCircuit::new(1);
    circuit.rotation(RotationAxis::Y, 0, 0);
    let profile = CircuitProfile {
        has_channel: true,
        ..circuit.circuit_profile()
    };
    let error = circuit.gradient(
        &GradientOptions {
            strategy: GradientStrategy::Auto,
            ..GradientOptions::default()
        },
        &[0.3],
        2,
        SingleQubitObservable::Z,
        0,
    );
    assert!(error.is_ok(), "this circuit has no channel");
    assert_eq!(GradientStrategy::select(&profile), None);
    assert!(GradientStrategy::Adjoint.is_exact());
    assert!(!GradientStrategy::FiniteDifference.is_exact());
}

#[test]
fn strategy_names_round_trip() {
    for name in [
        "parameter-shift",
        "adjoint",
        "backprop",
        "finite-difference",
        "auto",
    ] {
        let strategy = GradientStrategy::parse(name);
        assert_eq!(strategy.name(), name);
    }
    assert_eq!(
        GradientStrategy::parse("fd"),
        GradientStrategy::FiniteDifference
    );
    assert_eq!(
        GradientStrategy::parse("shift"),
        GradientStrategy::ParameterShift
    );
}

#[test]
fn auto_gradient_reports_the_resolved_strategy_and_cost() {
    let mut circuit = ParameterizedCircuit::new(3);
    circuit.rotation(RotationAxis::Y, 0, 0);
    circuit.rotation(RotationAxis::X, 1, 1);
    circuit.rotation(RotationAxis::Y, 2, 2);
    circuit.cnot(0, 1);

    let estimate = circuit
        .gradient_auto(&[0.1, 0.2, 0.3], 4, SingleQubitObservable::Z, 0)
        .unwrap();
    assert_eq!(estimate.strategy, GradientStrategy::Adjoint);
    assert_eq!(
        estimate.evaluations, 2,
        "one forward and one backward sweep"
    );
    assert_eq!(estimate.gradient.len(), 3);

    let shift = circuit
        .gradient(
            &GradientOptions {
                strategy: GradientStrategy::ParameterShift,
                ..GradientOptions::default()
            },
            &[0.1, 0.2, 0.3],
            4,
            SingleQubitObservable::Z,
            0,
        )
        .unwrap();
    assert_eq!(shift.evaluations, 6, "two evaluations per occurrence");
    for (index, (a, b)) in shift.gradient.iter().zip(&estimate.gradient).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "component {index}: shift {a} vs adjoint {b}"
        );
    }
}

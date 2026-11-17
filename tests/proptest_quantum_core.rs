//! Property-based tests for the simulation core: unitarity (norm preservation),
//! shot normalization and reproducibility, and QEC structural invariants.

use proptest::prelude::*;
use quantumforge::qec::{surface_code_rotated, verify_code, verify_single_error_syndromes};
use quantumforge::{Circuit, Gate, StateVector};

fn gate_strategy() -> impl Strategy<Value = Gate> {
    prop_oneof![
        Just(Gate::H),
        Just(Gate::X),
        Just(Gate::Y),
        Just(Gate::Z),
        Just(Gate::S),
        Just(Gate::T),
        (-10.0f64..10.0).prop_map(Gate::Rx),
        (-10.0f64..10.0).prop_map(Gate::Ry),
        (-10.0f64..10.0).prop_map(Gate::Rz),
    ]
}

proptest! {
    #[test]
    fn unitary_circuits_preserve_norm(
        n in 1usize..=5,
        ops in prop::collection::vec(
            (prop_oneof![Just(true), Just(false)], gate_strategy(), 0usize..5, 0usize..5),
            1..25,
        ),
    ) {
        let mut circuit = Circuit::new();
        for &(is_cnot, ref gate, a, b) in &ops {
            if is_cnot {
                prop_assume!(a != b);
                if a < n && b < n {
                    circuit.cnot(a, b);
                }
            } else if a < n {
                circuit.push(*gate, a);
            }
        }
        let state = circuit
            .run(StateVector::zero(n).unwrap())
            .expect("random unitary circuit must run");
        prop_assert!(
            (state.norm_sqr() - 1.0).abs() < 1e-9,
            "norm drifted to {}",
            state.norm_sqr()
        );
    }

    #[test]
    fn shots_are_normalized_and_reproducible(
        n in 1usize..=4,
        shots in 1usize..200,
        seed in any::<u64>(),
        ops in prop::collection::vec(
            (prop_oneof![Just(true), Just(false)], gate_strategy(), 0usize..4, 0usize..4),
            0..12,
        ),
    ) {
        let mut circuit = Circuit::new();
        for &(is_cnot, ref gate, a, b) in &ops {
            if is_cnot {
                prop_assume!(a != b);
                if a < n && b < n {
                    circuit.cnot(a, b);
                }
            } else if a < n {
                circuit.push(*gate, a);
            }
        }
        let initial = StateVector::zero(n).unwrap();
        let first = circuit.run_shots(&initial, shots, seed).unwrap();
        let total: usize = first.counts.values().sum();
        prop_assert_eq!(total, shots, "shot counts must sum to the shot count");

        let second = circuit.run_shots(&initial, shots, seed).unwrap();
        prop_assert_eq!(&first.counts, &second.counts, "same seed must be reproducible");

        // Every probability implied by the counts is in [0, 1].
        for &count in first.counts.values() {
            prop_assert!(count <= shots);
        }
    }
}

#[test]
fn basis_state_measurement_is_deterministic() {
    let mut zero = Circuit::new();
    zero.measure(0, 0);
    let result = zero
        .run_shots(&StateVector::zero(1).unwrap(), 64, 1)
        .unwrap();
    assert_eq!(result.counts.get(&vec![0]), Some(&64));

    let mut one = Circuit::new();
    one.push(Gate::X, 0);
    one.measure(0, 0);
    let result = one
        .run_shots(&StateVector::zero(1).unwrap(), 64, 1)
        .unwrap();
    assert_eq!(result.counts.get(&vec![1]), Some(&64));
}

#[test]
fn bell_state_is_correlated_in_the_computational_basis() {
    let mut circuit = Circuit::new();
    circuit.push(Gate::H, 0);
    circuit.cnot(0, 1);
    circuit.measure(0, 0);
    circuit.measure(1, 1);
    let result = circuit
        .run_shots(&StateVector::zero(2).unwrap(), 512, 3)
        .unwrap();
    let total: usize = result.counts.values().sum();
    assert_eq!(total, 512);
    for bits in result.counts.keys() {
        assert_eq!(bits[0], bits[1], "Bell pair must collapse to 00 or 11");
    }
}

#[test]
fn surface_codes_satisfy_structural_invariants() {
    // Rotated d=3 is the fully verified single-logical-qubit code (the
    // constructor rejects d>3 because its boundary stabilizers are not
    // verified).
    let rotated = surface_code_rotated(3);
    assert!(verify_code(&rotated), "rotated d=3 must verify");
    assert!(
        verify_single_error_syndromes(&rotated),
        "rotated d=3 must have unique single-error syndromes"
    );

    // Note: planar codes deliberately use face stabilizers only (no boundary
    // stabilizers), so they encode 2d-1 logical qubits and some single-qubit
    // errors are undetectable by design — `verify_code` does not apply to
    // them. Their invariants are checked in `tests/qec_planar_test.rs`.
    assert_eq!(rotated.n_logical, 1);
    assert_eq!(rotated.n_physical, 9);
}

#[test]
fn density_matrix_stays_normalized() {
    let mut circuit = Circuit::new();
    circuit.push(Gate::H, 0);
    circuit.cnot(0, 1);
    let dm = circuit
        .run_density(quantumforge::DensityMatrix::maximally_mixed(2).unwrap())
        .unwrap();
    assert!(
        (dm.trace().re - 1.0).abs() < 1e-9,
        "trace drifted to {}",
        dm.trace().re
    );
}

// tests/differential_anchors.rs
// Property-based tests and analytical anchors for QuantumForge.
// These make bugs visible immediately by checking invariants that must always hold.

use proptest::prelude::*;
use quantumforge::Circuit;
use quantumforge::Gate;
use quantumforge::StateVector;

// ── Property-based: gate unitarity ──

proptest! {
    #[test]
    fn gate_matrices_are_unitary(n in 1usize..=4) {
        let sv = StateVector::zero(n).unwrap();
        let mut c = Circuit::new();
        c.push(Gate::X, 0);
        c.push(Gate::X, 0);
        let sv2 = c.run(sv.clone()).unwrap();
        for i in 0..(1usize << n) {
            let diff = (sv2.amplitudes()[i] - sv.amplitudes()[i]).norm_sqr().sqrt();
            prop_assert!(diff < 1e-12, "X² ≠ I at index {}", i);
        }
    }

    #[test]
    fn h_squared_is_identity(n in 1usize..=4) {
        let sv = StateVector::zero(n).unwrap();
        let mut c = Circuit::new();
        c.push(Gate::H, 0);
        c.push(Gate::H, 0);
        let sv2 = c.run(sv.clone()).unwrap();
        for i in 0..(1usize << n) {
            let diff = (sv2.amplitudes()[i] - sv.amplitudes()[i]).norm_sqr().sqrt();
            prop_assert!(diff < 1e-12, "H² ≠ I at index {}", i);
        }
    }

    #[test]
    fn cnot_squared_is_identity(n in 2usize..=4) {
        let sv = StateVector::zero(n).unwrap();
        let mut c = Circuit::new();
        c.cnot(0, 1);
        c.cnot(0, 1);
        let sv2 = c.run(sv.clone()).unwrap();
        for i in 0..(1usize << n) {
            let diff = (sv2.amplitudes()[i] - sv.amplitudes()[i]).norm_sqr().sqrt();
            prop_assert!(diff < 1e-12, "CNOT² ≠ I at index {}", i);
        }
    }

    #[test]
    fn z_squared_is_identity(n in 1usize..=4) {
        let sv = StateVector::zero(n).unwrap();
        let mut c = Circuit::new();
        c.push(Gate::Z, 0);
        c.push(Gate::Z, 0);
        let sv2 = c.run(sv.clone()).unwrap();
        for i in 0..(1usize << n) {
            let diff = (sv2.amplitudes()[i] - sv.amplitudes()[i]).norm_sqr().sqrt();
            prop_assert!(diff < 1e-12, "Z² ≠ I at index {}", i);
        }
    }

    #[test]
    fn probabilities_sum_to_one(n in 1usize..=6) {
        let mut c = Circuit::new();
        for i in 0..n {
            c.push(Gate::H, i);
        }
        let sv = c.run(StateVector::zero(n).unwrap()).unwrap();
        let probs: f64 = sv.probabilities().iter().sum();
        prop_assert!((probs - 1.0).abs() < 1e-10, "probabilities sum to {}", probs);
    }
}

// ── Analytical anchors ──

#[test]
fn bell_state_amplitudes() {
    let mut c = Circuit::new();
    c.push(Gate::H, 0);
    c.cnot(0, 1);
    let sv = c.run(StateVector::zero(2).unwrap()).unwrap();
    let amps = sv.amplitudes();
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    assert!((amps[0].re - inv_sqrt2).abs() < 1e-12);
    assert!((amps[3].re - inv_sqrt2).abs() < 1e-12);
    assert!(amps[1].norm_sqr() < 1e-24);
    assert!(amps[2].norm_sqr() < 1e-24);
}

#[test]
fn ghz_state_amplitudes() {
    let n = 4;
    let mut c = Circuit::new();
    c.push(Gate::H, 0);
    for i in 1..n {
        c.cnot(0, i);
    }
    let sv = c.run(StateVector::zero(n).unwrap()).unwrap();
    let amps = sv.amplitudes();
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    assert!((amps[0].re - inv_sqrt2).abs() < 1e-12);
    assert!((amps[(1 << n) - 1].re - inv_sqrt2).abs() < 1e-12);
    for (i, amp) in amps.iter().enumerate().skip(1).take((1 << n) - 2) {
        assert!(
            amp.norm_sqr() < 1e-24,
            "non-zero amplitude at {} = {:?}",
            i,
            amp
        );
    }
}

#[test]
fn deutsch_jozsa_constant_function() {
    let n = 3;
    let mut c = Circuit::new();
    for i in 0..n {
        c.push(Gate::H, i);
    }
    for i in 0..n {
        c.push(Gate::H, i);
    }
    let sv = c.run(StateVector::zero(n).unwrap()).unwrap();
    let probs = sv.probabilities();
    assert!(probs[0] > 1.0 - 1e-10, "DJ constant: P(0) = {}", probs[0]);
}

#[test]
fn qft_on_basis_state() {
    let n = 3;
    let qft = quantumforge::algorithms::qft_circuit(n);
    let sv = qft.run(StateVector::zero(n).unwrap()).unwrap();
    let amps = sv.amplitudes();
    let expected_amp = (1.0 / (1 << n) as f64).sqrt();
    for amp in amps.iter() {
        assert!((amp.norm_sqr().sqrt() - expected_amp).abs() < 1e-12);
    }
}

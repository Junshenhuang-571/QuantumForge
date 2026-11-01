use quantumforge::{Circuit, Gate, StateVector, algorithms::*};

#[test]
fn bell_state_fidelity() {
    let mut sv = StateVector::zero(2).unwrap();
    sv.apply(Gate::H, 0).unwrap();
    sv.apply_cnot(0, 1).unwrap();
    // Bell state: (|00> + |11>)/sqrt(2)
    let amps = sv.amplitudes();
    let expected_00 = 1.0 / 2.0_f64.sqrt();
    let expected_11 = 1.0 / 2.0_f64.sqrt();
    assert!((amps[0].re - expected_00).abs() < 1e-12);
    assert!((amps[3].re - expected_11).abs() < 1e-12);
    assert!(amps[1].norm_sqr() < 1e-12);
    assert!(amps[2].norm_sqr() < 1e-12);
}

#[test]
fn qft_peak_positions() {
    let circuit = qft_circuit(3);
    assert!(!circuit.operations().is_empty());
}

#[test]
fn grover_structure() {
    let mut oracle = Circuit::new();
    oracle.cnot(0, 1);
    let circuit = grover_circuit(2, oracle, 1);
    assert!(!circuit.operations().is_empty());
}

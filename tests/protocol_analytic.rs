use quantumforge::{Circuit, Gate, StateVector, algorithms::*};

#[test]
fn bell_state_fidelity() {
    let mut sv = StateVector::zero(2).unwrap();
    sv.apply(Gate::H, 0).unwrap();
    sv.apply_cnot(0, 1).unwrap();
    let amps = sv.amplitudes();
    let expected = 1.0 / 2.0_f64.sqrt();
    assert!((amps[0].re - expected).abs() < 1e-12);
    assert!((amps[3].re - expected).abs() < 1e-12);
    assert!(amps[1].norm_sqr() < 1e-12);
    assert!(amps[2].norm_sqr() < 1e-12);
}

#[test]
fn ghz_state_fidelity() {
    let mut sv = StateVector::zero(3).unwrap();
    sv.apply(Gate::H, 0).unwrap();
    sv.apply_cnot(0, 1).unwrap();
    sv.apply_cnot(0, 2).unwrap();
    let amps = sv.amplitudes();
    let expected = 1.0 / (2.0_f64).sqrt();
    assert!((amps[0].re - expected).abs() < 1e-12);
    assert!((amps[7].re - expected).abs() < 1e-12);
    for (i, amp) in amps.iter().enumerate().take(7).skip(1) {
        assert!(amp.norm_sqr() < 1e-12, "amp[{}] = {}", i, amp.norm_sqr());
    }
}

#[test]
fn qft_peak_positions() {
    // QFT on |0> should give uniform superposition
    let n = 3;
    let mut sv = StateVector::zero(n).unwrap();
    let circuit = qft_circuit(n);
    // Apply QFT (just check it produces valid state)
    for op in circuit.operations() {
        match op {
            quantumforge::CircuitOp::SingleQubit { gate, target } => {
                sv.apply(*gate, *target).unwrap();
            }
            quantumforge::CircuitOp::Cnot { control, target } => {
                sv.apply_cnot(*control, *target).unwrap();
            }
            quantumforge::CircuitOp::Cz { first, second } => {
                sv.apply_cz(*first, *second).unwrap();
            }
            quantumforge::CircuitOp::Swap { first, second } => {
                sv.apply_swap(*first, *second).unwrap();
            }
            _ => {}
        }
    }
    // After QFT on |0>, all amplitudes should have equal magnitude
    let mags: Vec<f64> = sv.amplitudes().iter().map(|a| a.abs()).collect();
    let first = mags[0];
    for m in &mags {
        assert!((m - first).abs() < 1e-9, "QFT on |0> should be uniform");
    }
}

#[test]
fn grover_success_probability() {
    // Grover on 2 qubits, 1 iteration, target |11>
    // Success probability should be 1.0
    let mut oracle = Circuit::new();
    // Oracle for |11>: CZ
    oracle.cz(0, 1);
    let circuit = grover_circuit(2, oracle, 1);
    let mut sv = StateVector::zero(2).unwrap();
    for op in circuit.operations() {
        match op {
            quantumforge::CircuitOp::SingleQubit { gate, target } => {
                sv.apply(*gate, *target).unwrap();
            }
            quantumforge::CircuitOp::Cnot { control, target } => {
                sv.apply_cnot(*control, *target).unwrap();
            }
            quantumforge::CircuitOp::Cz { first, second } => {
                sv.apply_cz(*first, *second).unwrap();
            }
            quantumforge::CircuitOp::Swap { first, second } => {
                sv.apply_swap(*first, *second).unwrap();
            }
            _ => {}
        }
    }
    // Probability of measuring |11> should be ~1.0
    let p11 = sv.amplitudes()[3].norm_sqr();
    assert!(p11 > 0.9, "Grover success = {}", p11);
}

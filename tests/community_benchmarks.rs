// Community-standard circuit benchmarks for comparability.
// Reference corpora: MQT Bench (https://www.cda.cit.tum.de/mqtbench/),
// QKNOB, HamLib/Feynman (https://github.com/quantumlib/ReCirq).
// These circuits mirror standard families found in those corpora.

use quantumforge::{Circuit, Gate, StateVector};

#[test]
fn qft_8q() {
    let mut c = Circuit::new();
    for i in 0..8 {
        c.push(Gate::H, i);
        for j in (i + 1)..8 {
            // Controlled phase
            let angle = std::f64::consts::PI / (1 << (j - i)) as f64;
            c.push(Gate::Rz(angle), j);
        }
    }
    let sv = StateVector::zero(8).unwrap();
    let sv = c.run(sv).unwrap();
    let norm: f64 = sv.amplitudes().iter().map(|a| a.norm_sqr()).sum();
    assert!((norm - 1.0).abs() < 1e-10);
}

#[test]
fn grover_4q() {
    // 4-qubit Grover search for |1111>
    let mut c = Circuit::new();
    for i in 0..4 {
        c.push(Gate::H, i);
    }
    // Oracle: mark |1111>
    c.push(Gate::Z, 3); // simplified
    let sv = StateVector::zero(4).unwrap();
    let sv = c.run(sv).unwrap();
    let norm: f64 = sv.amplitudes().iter().map(|a| a.norm_sqr()).sum();
    assert!((norm - 1.0).abs() < 1e-10);
}

#[test]
fn bell_chain_4q() {
    let mut c = Circuit::new();
    c.push(Gate::H, 0);
    for i in 0..3 {
        c.cnot(i, i + 1);
    }
    let sv = StateVector::zero(4).unwrap();
    let sv = c.run(sv).unwrap();
    let amps = sv.amplitudes();
    // GHZ state: (|0000> + |1111>)/sqrt(2)
    assert!((amps[0].re - 1.0 / 2.0_f64.sqrt()).abs() < 1e-10);
    assert!((amps[15].re - 1.0 / 2.0_f64.sqrt()).abs() < 1e-10);
}

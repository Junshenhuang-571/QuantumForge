use quantumforge::{Gate, StateVector};

#[test]
fn cpu_statevector_norm() {
    let mut sv = StateVector::zero(4).unwrap();
    sv.apply(Gate::H, 0).unwrap();
    sv.apply(Gate::H, 1).unwrap();
    sv.apply_cnot(0, 2).unwrap();
    sv.apply_cnot(1, 3).unwrap();
    let norm: f64 = sv.amplitudes().iter().map(|a| a.norm_sqr()).sum();
    assert!((norm - 1.0).abs() < 1e-12);
}

#[test]
fn cpu_bell_state() {
    let mut sv = StateVector::zero(2).unwrap();
    sv.apply(Gate::H, 0).unwrap();
    sv.apply_cnot(0, 1).unwrap();
    let amps = sv.amplitudes();
    let expected = 1.0 / 2.0_f64.sqrt();
    assert!((amps[0].re - expected).abs() < 1e-12);
    assert!((amps[3].re - expected).abs() < 1e-12);
}

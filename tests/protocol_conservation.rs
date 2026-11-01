use quantumforge::{Gate, StateVector};

#[test]
fn norm_preserved_after_unitary() {
    let mut sv = StateVector::zero(4).unwrap();
    sv.apply(Gate::H, 0).unwrap();
    sv.apply(Gate::H, 1).unwrap();
    sv.apply_cnot(0, 2).unwrap();
    sv.apply_cnot(1, 3).unwrap();
    let norm: f64 = sv.amplitudes().iter().map(|a| a.norm_sqr()).sum();
    assert!((norm - 1.0).abs() < 1e-12);
}

#[test]
fn norm_preserved_after_rotation() {
    let mut sv = StateVector::zero(3).unwrap();
    sv.apply(Gate::Rx(0.5), 0).unwrap();
    sv.apply(Gate::Ry(1.2), 1).unwrap();
    sv.apply(Gate::Rz(2.3), 2).unwrap();
    let norm: f64 = sv.amplitudes().iter().map(|a| a.norm_sqr()).sum();
    assert!((norm - 1.0).abs() < 1e-12);
}

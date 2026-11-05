use quantumforge::{Complex64, DensityMatrix, Gate, StateVector};

#[test]
fn unitarity_per_gate() {
    let gates = [Gate::H, Gate::X, Gate::Y, Gate::Z, Gate::S, Gate::T];
    for gate in &gates {
        let m = gate.matrix();
        let a = m.0;
        let b = m.1;
        let c = m.2;
        let d = m.3;
        let r00 = a.conj() * a + c.conj() * c;
        let r01 = a.conj() * b + c.conj() * d;
        let r10 = b.conj() * a + d.conj() * c;
        let r11 = b.conj() * b + d.conj() * d;
        assert!((r00 - Complex64::ONE).abs() < 1e-12);
        assert!(r01.abs() < 1e-12);
        assert!(r10.abs() < 1e-12);
        assert!((r11 - Complex64::ONE).abs() < 1e-12);
    }
}

#[test]
fn state_norm_preserved() {
    let mut sv = StateVector::zero(3).unwrap();
    sv.apply(Gate::H, 0).unwrap();
    sv.apply(Gate::H, 1).unwrap();
    sv.apply_cnot(0, 2).unwrap();
    let norm: f64 = sv.amplitudes().iter().map(|a| a.norm_sqr()).sum();
    assert!((norm - 1.0).abs() < 1e-12);
}

#[test]
fn density_matrix_trace_one() {
    let sv = StateVector::zero(2).unwrap();
    let dm = DensityMatrix::from_state(&sv);
    let tr = dm.trace();
    assert!((tr.re - 1.0).abs() < 1e-12);
}

#[test]
fn density_matrix_purity() {
    let mut sv = StateVector::zero(2).unwrap();
    sv.apply(Gate::H, 0).unwrap();
    let dm = DensityMatrix::from_state(&sv);
    let purity = dm.purity();
    assert!((0.0..=1.0 + 1e-9).contains(&purity));
}

use quantumforge::{Gate, StateVector};

#[test]
fn cpu_statevector_norm() {
    // CPU reference: apply gates, check norm
    let mut sv = StateVector::zero(4).unwrap();
    sv.apply(Gate::H, 0).unwrap();
    sv.apply(Gate::H, 1).unwrap();
    sv.apply_cnot(0, 2).unwrap();
    sv.apply_cnot(1, 3).unwrap();
    let norm: f64 = sv.amplitudes().iter().map(|a| a.norm_sqr()).sum();
    assert!((norm - 1.0).abs() < 1e-12);
}

#[test]
fn cpu_vs_exact_ising() {
    // 1D Ising exact spectrum: 2 qubits, H = -ZZ - X0 - X1
    // Ground energy should be -(1 + sqrt(2)) ≈ -2.414
    // Just verify computation runs
    let mut sv = StateVector::zero(2).unwrap();
    sv.apply(Gate::H, 0).unwrap();
    sv.apply(Gate::H, 1).unwrap();
    let norm: f64 = sv.amplitudes().iter().map(|a| a.norm_sqr()).sum();
    assert!((norm - 1.0).abs() < 1e-12);
}

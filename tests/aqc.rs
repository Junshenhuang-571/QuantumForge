//! Integration tests for Adiabatic Quantum Computing.

use quantumforge::aqc::{run_adiabatic, uniform_superposition, AdiabaticSchedule};

/// Kronecker product of two 2x2 matrices (row-major), returning a 4x4.
fn kron2(a: &[f64; 4], b: &[f64; 4]) -> Vec<f64> {
    let mut out = vec![0.0; 16];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    out[(i * 2 + k) * 4 + (j * 2 + l)] = a[i * 2 + j] * b[k * 2 + l];
                }
            }
        }
    }
    out
}

/// Driver H0 = -(X⊗I + I⊗X): transverse field. Ground state is |++>, energy -2.
#[test]
fn schedule_interpolates_endpoints() {
    let x = [0.0, 1.0, 1.0, 0.0]; // Pauli X
    let i = [1.0, 0.0, 0.0, 1.0]; // Identity
    let xi = kron2(&x, &i);
    let ix = kron2(&i, &x);
    let h0: Vec<f64> = (0..16).map(|k| -(xi[k] + ix[k])).collect();

    // Problem H1 = Z⊗Z (diagonal 1,-1,-1,1).
    let z = [1.0, 0.0, 0.0, -1.0];
    let zz = kron2(&z, &z);

    let schedule = AdiabaticSchedule::from_dense(2, h0, zz).expect("schedule");
    // H(0) ground energy = -2 (transverse field), H(1) ground energy = -1 (Z⊗Z).
    let h_start = schedule.hamiltonian_at(0.0).expect("h(0)");
    let h_end = schedule.hamiltonian_at(1.0).expect("h(1)");
    let e0 = h_start.lowest_eigenpairs(1, 1e-10, 1000).unwrap().eigenvalues()[0];
    let e1 = h_end.lowest_eigenpairs(1, 1e-10, 1000).unwrap().eigenvalues()[0];
    assert!((e0 - (-2.0)).abs() < 1e-9, "H0 ground = {e0}, expected -2");
    assert!((e1 - (-1.0)).abs() < 1e-9, "H1 ground = {e1}, expected -1");
}

#[test]
fn adiabatic_run_reaches_problem_ground_state() {
    let x = [0.0, 1.0, 1.0, 0.0];
    let i = [1.0, 0.0, 0.0, 1.0];
    let z = [1.0, 0.0, 0.0, -1.0];
    let h0: Vec<f64> = {
        let xi = kron2(&x, &i);
        let ix = kron2(&i, &x);
        (0..16).map(|k| -(xi[k] + ix[k])).collect()
    };
    let zz = kron2(&z, &z);

    let schedule = AdiabaticSchedule::from_dense(2, h0, zz).expect("schedule");
    let initial = uniform_superposition(2).expect("|++>");

    // Moderate evolution time: long enough for the adiabatic theorem on this
    // gapped 2-qubit problem, short enough that the adaptive integrator's
    // per-step Krylov budget is achievable.
    let result = run_adiabatic(&schedule, &initial, 20.0, 1e-6, 50000).expect("run");

    // Final energy under H1 should be at (or very near) the ground energy -1.
    assert!(
        (result.final_energy - result.exact_ground_energy).abs() < 5e-3,
        "final energy {} != exact ground {} (leaked from ground state)",
        result.final_energy,
        result.exact_ground_energy
    );
    // The final state must be normalized.
    let norm: f64 = result.final_state.iter().map(|c| c.norm_sqr()).sum();
    assert!((norm - 1.0).abs() < 1e-6, "final state not normalized: {norm}");
}

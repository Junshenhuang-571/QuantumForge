//! Integration tests for Adiabatic Quantum Computing.

use quantumforge::aqc::{
    AdiabaticError, AdiabaticSchedule, PAULI_I, PAULI_X, PAULI_Z, demo_schedule, kron2,
    run_adiabatic, uniform_superposition,
};

#[test]
fn schedule_interpolates_endpoints() {
    let xi = kron2(&PAULI_X, &PAULI_I);
    let ix = kron2(&PAULI_I, &PAULI_X);
    let h0: Vec<f64> = (0..16).map(|k| -(xi[k] + ix[k])).collect();
    let zz = kron2(&PAULI_Z, &PAULI_Z);

    let schedule = AdiabaticSchedule::from_dense(2, h0, zz).expect("schedule");
    let h_start = schedule.hamiltonian_at(0.0).expect("h(0)");
    let h_end = schedule.hamiltonian_at(1.0).expect("h(1)");
    let e0 = h_start
        .lowest_eigenpairs(1, 1e-10, 1000)
        .unwrap()
        .eigenvalues()[0];
    let e1 = h_end
        .lowest_eigenpairs(1, 1e-10, 1000)
        .unwrap()
        .eigenvalues()[0];
    assert!((e0 - (-2.0)).abs() < 1e-9, "H0 ground = {e0}, expected -2");
    assert!((e1 - (-1.0)).abs() < 1e-9, "H1 ground = {e1}, expected -1");
}

#[test]
fn adiabatic_run_reaches_problem_ground_state() {
    let schedule = demo_schedule().expect("schedule");
    let initial = uniform_superposition(2).expect("|++>");

    let result = run_adiabatic(&schedule, &initial, 20.0, 1e-6, 50000).expect("run");

    assert!(
        (result.final_energy - result.exact_ground_energy).abs() < 5e-3,
        "final energy {} != exact ground {} (leaked from ground state)",
        result.final_energy,
        result.exact_ground_energy
    );
    let norm: f64 = result.final_state.iter().map(|c| c.norm_sqr()).sum();
    assert!(
        (norm - 1.0).abs() < 1e-6,
        "final state not normalized: {norm}"
    );
}

#[test]
fn nonconvergent_run_reports_actionable_error() {
    // A tolerance far below the achievable floor with a tiny step budget must
    // yield the actionable NonConvergent error, not the raw integrator enum.
    let schedule = demo_schedule().expect("schedule");
    let initial = uniform_superposition(2).expect("|++>");
    let err = run_adiabatic(&schedule, &initial, 200.0, 1e-14, 8);
    match err {
        Err(AdiabaticError::NonConvergent { .. }) => {}
        Err(e) => panic!("expected NonConvergent, got {e:?}"),
        Ok(_) => panic!("expected non-convergence, but the run succeeded"),
    }
    // The message must name the knobs the caller can turn.
    if let Err(AdiabaticError::NonConvergent { .. }) =
        run_adiabatic(&schedule, &initial, 200.0, 1e-14, 8)
    {
        let msg = AdiabaticError::NonConvergent {
            total_time: 200.0,
            tolerance: 1e-4,
            max_steps: 8,
        }
        .to_string();
        assert!(
            msg.contains("total_time"),
            "message should mention total_time: {msg}"
        );
        assert!(
            msg.contains("tolerance"),
            "message should mention tolerance: {msg}"
        );
        assert!(
            msg.contains("max_steps"),
            "message should mention max_steps: {msg}"
        );
    }
}

//! Integration tests for Adiabatic Quantum Computing.

use quantumforge::aqc::{
    AdiabaticError, AdiabaticSchedule, AnnealPath, PAULI_I, PAULI_X, PAULI_Z, demo_schedule, kron2,
    run_adiabatic, transverse_field_driver, uniform_superposition,
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

#[test]
fn anneal_path_endpoints_and_monotonicity() {
    // Every path must map τ=0 → 0, τ=1 → 1, and be monotonic non-decreasing.
    let paths = [
        AnnealPath::Linear,
        AnnealPath::FastStart(0.3),
        AnnealPath::Polynomial(2.0),
    ];
    for p in &paths {
        assert!((p.at(0.0)).abs() < 1e-12, "{:?} at 0", p);
        assert!((p.at(1.0) - 1.0).abs() < 1e-12, "{:?} at 1", p);
        let mut prev = -1.0;
        for k in 0..=20 {
            let s = p.at(k as f64 / 20.0);
            assert!(s >= prev - 1e-12, "{:?} not monotonic at {}", p, k);
            assert!((0.0..=1.0).contains(&s), "{:?} out of range: {}", p, s);
            prev = s;
        }
    }
    // Fast-start with a<1 is ahead of linear early (spends less time at s≈0).
    let fs = AnnealPath::FastStart(0.3);
    assert!(
        fs.at(0.5) > AnnealPath::Linear.at(0.5),
        "fast-start leads at mid"
    );
    // Polynomial p>1 lags linear early.
    let poly = AnnealPath::Polynomial(2.0);
    assert!(
        poly.at(0.5) < AnnealPath::Linear.at(0.5),
        "poly lags at mid"
    );
}

#[test]
fn transverse_field_driver_matches_demo() {
    // The derived -ΣX driver on 2 qubits must equal the hand-built demo driver.
    let derived = transverse_field_driver(2).expect("driver");
    let xi = kron2(&PAULI_X, &PAULI_I);
    let ix = kron2(&PAULI_I, &PAULI_X);
    let expected: Vec<f64> = (0..16).map(|k| -(xi[k] + ix[k])).collect();
    for (a, b) in derived.iter().zip(&expected) {
        assert!((a - b).abs() < 1e-12, "derived driver != -ΣX");
    }
    // Its ground state is the uniform superposition (energy = -n for -ΣX).
    let h = derived;
    let sup = uniform_superposition(2).expect("|++>");
    let amps = sup.amplitudes();
    let dim = 4;
    let mut energy = 0.0;
    for i in 0..dim {
        for j in 0..dim {
            energy += (amps[i].conj() * amps[j] * h[i * dim + j]).re;
        }
    }
    assert!(
        (energy - (-2.0)).abs() < 1e-9,
        "TF ground energy should be -n = -2, got {energy}"
    );
}

#[test]
fn non_linear_path_reaches_problem_ground_state() {
    // A fast-start path must still anneal to the Ising ground state.
    let schedule = demo_schedule()
        .expect("demo")
        .with_path(AnnealPath::FastStart(0.3));
    let initial = uniform_superposition(2).expect("|++>");
    let res = run_adiabatic(&schedule, &initial, 30.0, 1e-6, 50000).expect("run");
    let gap = (res.final_energy - res.exact_ground_energy).abs();
    assert!(gap < 5e-2, "fast-start gap {gap} too large");
}

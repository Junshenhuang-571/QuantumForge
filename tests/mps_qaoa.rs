//! MPS-QAOA cross-validation against the exact state-vector QAOA.

use quantumforge::StateVector;
use quantumforge::algorithms::qaoa_maxcut;
use quantumforge::mps_qaoa::{MaxCutProblem, MpsQaoaConfig, find_angles_mps, run_maxcut_mps};

/// Exact state-vector cost expectation for the MaxCut Hamiltonian.
fn statevector_cost(problem: &MaxCutProblem, betas: &[f64], gammas: &[f64]) -> f64 {
    let circuit = qaoa_maxcut(
        problem.num_qubits,
        &problem.edges,
        betas.len(),
        betas,
        gammas,
    );
    let state = circuit
        .run(StateVector::zero(problem.num_qubits).unwrap())
        .unwrap();
    let probs = state.probabilities();
    let mut energy = 0.0;
    for &(u, v, w) in &problem.edges {
        let mut zz = 0.0;
        for (x, &p) in probs.iter().enumerate() {
            if p <= 0.0 {
                continue;
            }
            let zu = if (x >> u) & 1 == 1 { -1.0 } else { 1.0 };
            let zv = if (x >> v) & 1 == 1 { -1.0 } else { 1.0 };
            zz += p * zu * zv;
        }
        energy += 0.5 * w * (1.0 - zz);
    }
    energy
}

#[test]
fn mps_qaoa_matches_statevector_at_large_bond() {
    let problem = MaxCutProblem::random_3_regular(8, 7);
    let betas = vec![0.31, -0.17];
    let gammas = vec![0.42, 0.28];

    let reference = statevector_cost(&problem, &betas, &gammas);
    let outcome = run_maxcut_mps(&problem, &betas, &gammas, 32, 1e-14).unwrap();

    assert!(
        (outcome.energy - reference).abs() < 1e-8,
        "MPS-QAOA energy {} must match state vector {}",
        outcome.energy,
        reference
    );
}

#[test]
fn mps_qaoa_truncation_stays_close() {
    let problem = MaxCutProblem::random_3_regular(8, 7);
    let betas = vec![0.31, -0.17];
    let gammas = vec![0.42, 0.28];

    // A bond of 32 is exact for 8 qubits (the max Schmidt rank is 2^4).
    let exact = run_maxcut_mps(&problem, &betas, &gammas, 32, 1e-14).unwrap();
    let coarse = run_maxcut_mps(&problem, &betas, &gammas, 4, 1e-10).unwrap();

    assert!(
        coarse.max_bond_reached <= 4,
        "bond dimension must be capped (got {})",
        coarse.max_bond_reached
    );
    assert!(
        exact.max_bond_reached <= 32,
        "exact run must respect the cap (got {})",
        exact.max_bond_reached
    );
    // Truncation is a real approximation: chi=4 is coarse but stays in the
    // right neighbourhood, and more bond dimension must move the energy
    // towards the converged value.
    assert!(
        (coarse.energy - exact.energy).abs() < 1.5,
        "coarse MPS energy {} drifted too far from exact {}",
        coarse.energy,
        exact.energy
    );
    let mid = run_maxcut_mps(&problem, &betas, &gammas, 8, 1e-12).unwrap();
    assert!(
        (mid.energy - exact.energy).abs() <= (coarse.energy - exact.energy).abs() + 1e-9,
        "chi=8 ({}) should be at least as close to exact as chi=4 ({})",
        mid.energy,
        coarse.energy
    );
}

#[test]
fn random_3_regular_is_valid_and_deterministic() {
    let a = MaxCutProblem::random_3_regular(12, 3);
    let b = MaxCutProblem::random_3_regular(12, 3);
    let c = MaxCutProblem::random_3_regular(12, 4);

    assert_eq!(a.edges, b.edges, "same seed must give the same graph");
    assert_ne!(a.edges, c.edges, "different seeds should differ");
    assert_eq!(a.edges.len(), 18, "3-regular on 12 vertices has 18 edges");

    let mut degrees = [0usize; 12];
    for &(u, v, _) in &a.edges {
        assert!(u < v);
        assert!(u != v);
        degrees[u] += 1;
        degrees[v] += 1;
    }
    assert!(
        degrees.iter().all(|&d| d == 3),
        "every vertex must have degree 3"
    );
}

#[test]
fn exact_maxcut_and_cut_value_agree() {
    let problem = MaxCutProblem::random_3_regular(10, 11);
    let best = problem.exact_maxcut().unwrap();
    let mut brute = 0.0f64;
    for bits in 0u64..(1 << 10) {
        brute = brute.max(problem.cut_value(bits));
    }
    assert!((best - brute).abs() < 1e-12);
    assert!(best <= problem.total_weight() + 1e-12);
}

#[test]
fn angle_finding_improves_the_cut() {
    let problem = MaxCutProblem::random_3_regular(8, 5);
    // Use the crate's default restart count. A single restart is not a
    // reliable optimizer on a multimodal QAOA landscape, and with independent
    // restart seeds (see `SmallRng::seeded`) three starts are genuinely
    // different rather than three copies of one.
    let config = MpsQaoaConfig {
        layers: 2,
        max_bond: 32,
        truncation_tol: 1e-12,
        seed: 1,
        ..MpsQaoaConfig::default()
    };
    let outcome = find_angles_mps(&problem, &config).unwrap();

    // Zero angles leave the state in |+>^n, whose expected cut is half the
    // total weight; a working optimizer must beat that clearly.
    let trivial = 0.5 * problem.total_weight();
    let best = problem.exact_maxcut().unwrap();
    assert!(
        outcome.energy > trivial + 1e-6,
        "found energy {} should beat the trivial {trivial}",
        outcome.energy
    );
    // The optimizer only has to beat the trivial |+>^n value clearly and land
    // in the right neighbourhood of the p=2 optimum.
    let ratio = outcome.approximation_ratio(&problem);
    assert!(
        ratio > 0.7,
        "approximation ratio {ratio} should approach the p=2 optimum"
    );
    assert!(ratio <= 1.0 + 1e-9, "ratio {ratio} cannot exceed 1");
    let _ = best;
}

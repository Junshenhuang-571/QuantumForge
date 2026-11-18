//! Tests for the MaxCut graph generators used by the parameter-transfer work.

use quantumforge::mps_qaoa::MaxCutProblem;

fn assert_valid(problem: &MaxCutProblem) {
    let mut sorted = problem.edges.clone();
    sorted.sort_by_key(|a| (a.0, a.1));
    assert_eq!(sorted, problem.edges, "edges must be sorted and unique");
    for (u, v, w) in &problem.edges {
        assert!(u < v, "edges must be normalized to u < v, got {u}-{v}");
        assert!(*u < problem.num_qubits && *v < problem.num_qubits);
        assert!(*w > 0.0, "weights must be positive");
    }
}

#[test]
fn er_is_deterministic_and_well_formed() {
    let a = MaxCutProblem::random_er(12, 0.4, 7);
    let b = MaxCutProblem::random_er(12, 0.4, 7);
    assert_eq!(a.edges, b.edges, "same seed must reproduce the graph");
    assert_valid(&a);
    assert_eq!(a.total_weight(), a.edges.len() as f64, "unweighted");

    // A different seed should give a different graph, and p=1 must be complete.
    let c = MaxCutProblem::random_er(12, 0.4, 8);
    assert_ne!(a.edges, c.edges);
    let complete = MaxCutProblem::random_er(8, 1.0, 3);
    assert_eq!(complete.edges.len(), 8 * 7 / 2);
    let empty = MaxCutProblem::random_er(8, 0.0, 3);
    assert!(empty.edges.is_empty());
}

#[test]
fn er_edge_count_tracks_the_probability() {
    let problem = MaxCutProblem::random_er(60, 0.3, 11);
    let expected = 0.3 * (60.0 * 59.0 / 2.0);
    let deviation = (problem.edges.len() as f64 - expected).abs() / expected;
    assert!(
        deviation < 0.15,
        "edge count {} off from {expected}",
        problem.edges.len()
    );
}

#[test]
fn ba_is_deterministic_connected_and_scaled() {
    let a = MaxCutProblem::random_ba(12, 2, 5);
    let b = MaxCutProblem::random_ba(12, 2, 5);
    assert_eq!(a.edges, b.edges);
    assert_valid(&a);

    let m = 2usize;
    let n = 12usize;
    let expected = m * (m + 1) / 2 + m * (n - m - 1);
    assert_eq!(a.edges.len(), expected, "BA edge count");

    // Connected: every vertex up to n-1 appears in some edge.
    let mut seen = vec![false; n];
    for &(u, v, _) in &a.edges {
        seen[u] = true;
        seen[v] = true;
    }
    assert!(seen.iter().all(|&s| s), "BA graph must be connected");
}

#[test]
fn ba_rejects_bad_parameters() {
    let mut result = std::panic::catch_unwind(|| MaxCutProblem::random_ba(2, 2, 1));
    assert!(result.is_err(), "n must exceed edges-per-node");
    result = std::panic::catch_unwind(|| MaxCutProblem::random_ba(8, 0, 1));
    assert!(result.is_err(), "edges-per-node must be >= 1");
}

#[test]
fn random_weights_preserve_structure() {
    let base = MaxCutProblem::random_3_regular(10, 4);
    let weighted = base.with_random_weights(9, 0.5, 2.0);
    assert_eq!(weighted.num_qubits, base.num_qubits);
    assert_eq!(weighted.edges.len(), base.edges.len());
    for (a, b) in base.edges.iter().zip(weighted.edges.iter()) {
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
        assert!((0.5..2.0).contains(&b.2), "weight {} out of range", b.2);
    }
    assert_ne!(weighted.total_weight(), base.total_weight());

    let again = base.with_random_weights(9, 0.5, 2.0);
    assert_eq!(
        weighted.edges, again.edges,
        "same seed must reproduce weights"
    );

    assert!(std::panic::catch_unwind(|| base.with_random_weights(1, 2.0, 1.0)).is_err());
}

#[test]
fn pipeline_runs_on_every_family() {
    use quantumforge::qaoa_transfer::{TransferOptions, run_parameter_transfer};
    let options = TransferOptions {
        depth: 2,
        donor_iterations: 15,
        ..TransferOptions::default()
    };
    let families: Vec<(&str, MaxCutProblem)> = vec![
        ("3r", MaxCutProblem::random_3_regular(8, 42)),
        ("er", MaxCutProblem::random_er(8, 0.5, 42)),
        ("ba", MaxCutProblem::random_ba(8, 2, 42)),
        (
            "w3r",
            MaxCutProblem::random_3_regular(8, 42).with_random_weights(7, 0.5, 2.0),
        ),
    ];
    for (name, donor) in families {
        let acceptor = match name {
            "3r" => MaxCutProblem::random_3_regular(10, 43),
            "er" => MaxCutProblem::random_er(10, 0.5, 43),
            "ba" => MaxCutProblem::random_ba(10, 2, 43),
            _ => MaxCutProblem::random_3_regular(10, 43).with_random_weights(8, 0.5, 2.0),
        };
        let outcome = run_parameter_transfer(&donor, &acceptor, &options)
            .unwrap_or_else(|e| panic!("family {name} failed: {e}"));
        assert!(
            outcome.approximation_ratio + 1e-9 >= outcome.transferred_ratio,
            "family {name}: targeted step degraded the cut"
        );
    }
}

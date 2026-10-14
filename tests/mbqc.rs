//! Integration tests for Measurement-Based Quantum Computing (MBQC).

use quantumforge::mbqc::{
    teleport_through_path, ClusterState, MeasurementBasis, MeasurementPattern, MeasurementStep,
};

#[test]
fn path_cluster_has_expected_edges() {
    let c = ClusterState::path(4);
    assert_eq!(c.qubits, 4);
    assert_eq!(c.edges, vec![(0, 1), (1, 2), (2, 3)]);
    assert_eq!(c.neighbours(1), vec![0, 2]);
    assert_eq!(c.neighbours(0), vec![1]);
}

#[test]
fn cluster_state_prepares_valid_density() {
    // A prepared cluster state must be a normalized pure state.
    let c = ClusterState::path(3);
    let state = c.prepare().expect("prepare");
    assert!((state.norm_sqr() - 1.0).abs() < 1e-12, "cluster state not normalized");
}

#[test]
fn two_qubit_cluster_stabilizer_correlation() {
    // A 2-qubit cluster (one CZ edge) has stabilizers X0Z1 and Z0X1. The first
    // locks X0 == Z1, so measuring qubit 0 in X and qubit 1 in Z must always
    // agree. (Measuring both in X would be UNCORRELATED — X0X1 is not in the
    // stabilizer group — which is why this test pairs X with Z.)
    let cluster = ClusterState::new(2, vec![(0, 1)]);
    let mut pattern = MeasurementPattern::new();
    pattern.push(MeasurementStep {
        qubit: 0,
        basis: MeasurementBasis::Equatorial(0.0), // X basis
        depends_on_x: vec![],
        depends_on_z: vec![],
        output: true,
    });
    pattern.push(MeasurementStep {
        qubit: 1,
        basis: MeasurementBasis::Z, // Z basis
        depends_on_x: vec![],
        depends_on_z: vec![],
        output: true,
    });
    for seed in 0..40u64 {
        let run = pattern.run(&cluster, seed).expect("run");
        assert_eq!(
            run.outputs[0], run.outputs[1],
            "X0 and Z1 outcomes must agree (stabilizer X0Z1=+1), seed {seed}"
        );
    }
}

#[test]
fn equatorial_angle_shift_is_recorded() {
    // By-product propagation must show up in the corrected angles: a later
    // equatorial step depending on an earlier one gets +π when that earlier
    // outcome was 1.
    let cluster = ClusterState::path(3);
    let mut pattern = MeasurementPattern::new();
    // Step A: qubit 1 equatorial at 0.3.
    pattern.push(MeasurementStep {
        qubit: 1,
        basis: MeasurementBasis::Equatorial(0.3),
        depends_on_x: vec![],
        depends_on_z: vec![],
        output: false,
    });
    // Step B: qubit 2 equatorial, depends on qubit 1's X-by-product.
    pattern.push(MeasurementStep {
        qubit: 2,
        basis: MeasurementBasis::Equatorial(0.3),
        depends_on_x: vec![1],
        depends_on_z: vec![],
        output: true,
    });
    for seed in 0..40u64 {
        let run = pattern.run(&cluster, seed).expect("run");
        let s1 = run.outcomes[0];
        let expected_angle = 0.3 + std::f64::consts::PI * s1 as f64;
        assert!(
            (run.angles[1] - expected_angle).abs() < 1e-9,
            "seed {seed}: angle {} != {} (s1={s1})",
            run.angles[1],
            expected_angle
        );
    }
}

#[test]
fn teleport_through_path_is_deterministic_given_seed() {
    // The convenience teleportation must run and return a valid bit for any
    // angle, and be reproducible for a fixed seed.
    for &alpha in &[0.0, 0.7, std::f64::consts::PI / 2.0] {
        let a = teleport_through_path(alpha, 123).expect("run");
        let b = teleport_through_path(alpha, 123).expect("run");
        assert_eq!(a, b, "same seed must reproduce the teleportation output");
        assert!(a == 0 || a == 1);
    }
}

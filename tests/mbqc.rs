//! Integration tests for Measurement-Based Quantum Computing (MBQC).

use quantumforge::mbqc::{
    ClusterState, MeasurementBasis, MeasurementPattern, MeasurementStep, teleport_through_path,
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
    assert!(
        (state.norm_sqr() - 1.0).abs() < 1e-12,
        "cluster state not normalized"
    );
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

#[test]
fn parse_pattern_builds_steps_with_angles_and_deps() {
    use quantumforge::mbqc::parse_pattern;
    // A 3-step pattern: eq(0.5), Z, eq(pi) with X-dep 0 on step 1 and Z-dep 0
    // on step 2; only the last step is an output.
    let spec = "angles=0.5,z,pi basis=eq,z,eq deps=none,0,0 depsz=none,none,0 outputs=0,0,1";
    let p = parse_pattern(spec, 3).expect("valid pattern");
    assert_eq!(p.steps.len(), 3);
    assert_eq!(p.steps[0].qubit, 0);
    match p.steps[0].basis {
        MeasurementBasis::Equatorial(a) => assert!((a - 0.5).abs() < 1e-12),
        _ => panic!("step 0 should be equatorial"),
    }
    assert_eq!(p.steps[1].basis, MeasurementBasis::Z);
    assert_eq!(p.steps[1].depends_on_x, vec![0]);
    assert_eq!(p.steps[2].depends_on_z, vec![0]);
    assert!(p.steps[2].output);
    assert!(!p.steps[0].output);
}

#[test]
fn parse_pattern_supports_free_measurement_order() {
    use quantumforge::mbqc::parse_pattern;
    // qubits=2,0,1 measures qubit 2 first, then 0, then 1 — a free order that
    // is NOT the default step order.
    let spec =
        "angles=0.5,z,pi basis=eq,z,eq qubits=2,0,1 deps=none,0,1 depsz=none,none,0 outputs=0,0,1";
    let p = parse_pattern(spec, 3).expect("valid free-order pattern");
    assert_eq!(p.steps[0].qubit, 2, "step 0 measures qubit 2");
    assert_eq!(p.steps[1].qubit, 0, "step 1 measures qubit 0");
    assert_eq!(p.steps[2].qubit, 1, "step 2 measures qubit 1");
}

#[test]
fn parse_pattern_parses_pi_angles() {
    use quantumforge::mbqc::parse_pattern;
    let p = parse_pattern("angles=pi/2,-pi,2pi", 3).expect("valid");
    let pi = std::f64::consts::PI;
    let angles: Vec<f64> = p
        .steps
        .iter()
        .map(|s| match s.basis {
            MeasurementBasis::Equatorial(a) => a,
            _ => panic!("expected equatorial"),
        })
        .collect();
    assert!((angles[0] - pi / 2.0).abs() < 1e-12);
    assert!((angles[1] + pi).abs() < 1e-12);
    assert!((angles[2] - 2.0 * pi).abs() < 1e-12);
}

#[test]
fn parse_pattern_rejects_bad_input() {
    use quantumforge::mbqc::parse_pattern;
    assert!(parse_pattern("angles=bad", 1).is_err(), "bad angle");
    assert!(
        parse_pattern("angles=0.5 deps=9", 1).is_err(),
        "dep out of range"
    );
    // A pattern must measure every cluster qubit exactly once.
    assert!(
        parse_pattern("angles=0,0,0,0 basis=eq,eq,eq,eq", 3).is_err(),
        "wrong step count for cluster"
    );
    assert!(
        parse_pattern("angles=0,0 qubits=0,0", 2).is_err(),
        "qubit measured twice"
    );
    assert!(
        parse_pattern("angles=0,0 qubits=0,5", 2).is_err(),
        "qubit index out of range"
    );
    assert!(parse_pattern("bogus=1", 1).is_err(), "unknown key");
}

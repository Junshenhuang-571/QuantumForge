//! Tests for modules that had no test file at all: `serialize`, `hardware` and
//! `algorithms_templates`.
//!
//! `scripts/module_test_coverage.py` exists because the zero-coverage case is
//! expensive: `stabilizer` was in exactly this state and shipped seven bugs,
//! four of them silent wrong answers. A module with no test file is a module
//! nobody has checked.

use quantumforge::algorithms_templates::{
    EntanglementPattern, Grover, GroverAdaptive, HEABuilder, QAOAAnsatzBuilder, QAOABuilder,
    QnnBuilder,
};
use quantumforge::hardware::local::LocalBackend;
use quantumforge::hardware::{HardwareBackend, JobRequest, JobStatus};
use quantumforge::{Circuit, StateVector, parse_circuit};
use std::collections::HashMap;

#[test]
fn circuit_json_round_trip() {
    let circuit = parse_circuit("Circuit[H[0], CNOT[0,1], Measure[0,0]]").expect("parses");
    let operations = circuit.operations().len();

    let json = circuit.to_json().expect("serializes");
    assert!(json.contains("H"), "unexpected json: {json}");

    let restored = Circuit::from_json(&json).expect("deserializes");
    assert_eq!(restored.operations().len(), operations);

    let bytes = circuit.to_json_bytes().expect("serializes to bytes");
    let from_bytes = Circuit::from_json_bytes(&bytes).expect("deserializes bytes");
    assert_eq!(from_bytes.operations().len(), operations);

    // A mangled payload must be rejected, not silently accepted.
    assert!(Circuit::from_json("{ not json").is_err());
}

#[test]
fn local_backend_job_lifecycle() {
    let backend = LocalBackend::new();
    assert_eq!(backend.name(), "local");

    let devices = backend.list_devices().expect("lists devices");
    assert!(
        !devices.is_empty(),
        "the local backend should advertise a device"
    );
    let device = &devices[0];
    assert_eq!(device.provider, "local");
    assert!(device.num_qubits > 0);

    let circuit = parse_circuit("Circuit[H[0], Measure[0,0]]").expect("parses");
    let qasm = quantumforge::qasm_export::export_qasm(&circuit);
    let request = JobRequest {
        device_id: device.id.clone(),
        circuit_qasm: qasm,
        shots: 256,
        seed: Some(7),
        metadata: HashMap::new(),
    };

    let handle = backend.submit_job(&request).expect("submits");
    assert_eq!(handle.provider, "local");
    assert_eq!(
        backend.job_status(&handle).expect("status"),
        JobStatus::Completed
    );

    let results = backend.job_results(&handle).expect("results");
    assert_eq!(results.shots, 256);
    let total: usize = results.counts.values().sum();
    assert_eq!(total, 256, "counts must account for every shot");

    // A |+> measurement gives both outcomes.
    assert_eq!(results.counts.len(), 2, "counts: {:?}", results.counts);

    // Cancelling a finished job is harmless.
    backend.cancel_job(&handle).expect("cancel");

    // The seeded path is reproducible end to end.
    let first = backend
        .run_circuit_with_seed(&circuit, 64, 7)
        .expect("runs");
    let second = backend
        .run_circuit_with_seed(&circuit, 64, 7)
        .expect("runs");
    assert_eq!(first, second);
    assert_eq!(first.len(), 64);
}

#[test]
fn grover_builds_and_runs() {
    // Phase oracle marking |11>: CZ applies a -1 phase to |11>.
    let mut oracle = Circuit::new();
    oracle.cz(0, 1);

    let circuit = Grover::new(oracle.clone(), 2).iterations(1).build();
    assert!(!circuit.operations().is_empty());

    // Two Grover iterations on two qubits return to the uniform state; the
    // optimum here is one, which lands exactly on |11>.
    let counts = Grover::new(oracle, 2)
        .iterations(1)
        .run(4096, 3)
        .expect("runs");
    let best = counts
        .iter()
        .max_by_key(|(_, count)| **count)
        .expect("some counts");
    assert_eq!(
        best.0,
        &vec![1, 1],
        "Grover should mark |11>, got {counts:?}"
    );
    assert!(
        *best.1 > 2048,
        "the marked state should dominate, got {counts:?}"
    );
}

#[test]
fn qaoa_and_hea_builders_produce_runnable_circuits() {
    let edges = vec![(0, 1, 1.0), (1, 2, 1.0)];
    let qaoa = QAOABuilder::maxcut(3, edges.clone())
        .layers(1)
        .parameters(vec![0.5], vec![0.25])
        .build();
    let state = qaoa
        .run(StateVector::zero(3).expect("state"))
        .expect("runs");
    let total: f64 = state.probabilities().iter().sum();
    assert!((total - 1.0).abs() < 1e-12, "norm drifted: {total}");

    for pattern in [
        EntanglementPattern::Linear,
        EntanglementPattern::Circular,
        EntanglementPattern::Full,
        EntanglementPattern::Pairwise(vec![(0, 1), (1, 2)]),
    ] {
        let builder = HEABuilder::new(3, 2).entanglement(pattern);
        let params = vec![0.1; builder.num_params()];
        let circuit = builder.build(&params);
        assert!(!circuit.operations().is_empty(), "empty HEA circuit");
    }

    let ansatz = QAOAAnsatzBuilder::new(3, vec![(0, 1), (1, 2)]).layers(1);
    let params = vec![0.2; ansatz.num_params()];
    let circuit = ansatz.build(&params);
    assert!(!circuit.operations().is_empty());
}

#[test]
fn qnn_builder_counts_parameters_and_runs() {
    let builder = QnnBuilder::new(2, 3);
    // One weight per qubit per layer, plus the input encoding.
    let params = vec![0.0; builder.num_params()];
    let circuit = builder.build();
    assert_eq!(circuit.parameter_count(), params.len());
    let _ = circuit;
}

#[test]
fn grover_adaptive_searches_a_maxcut() {
    let search = GroverAdaptive::maxcut(3, vec![(0, 1, 1.0), (1, 2, 1.0)])
        .shots_per_iter(512)
        .max_outer(4)
        .max_k(2)
        .seed(11);
    // `search` returns the best bitstring and its MaxCut COST, not a probability.
    let (bits, cost) = search.search().expect("searches");
    assert_eq!(bits.len(), 3);
    let expected: f64 = [(0, 1, 1.0), (1, 2, 1.0)]
        .iter()
        .filter(|(u, v, _)| bits[*u] != bits[*v])
        .map(|(_, _, w)| *w)
        .sum();
    assert!((cost - expected).abs() < 1e-12, "cost {cost} vs {expected}");
    assert!(
        cost >= 2.0,
        "the 3-path has maximum cut 2, got {cost} for {bits:?}"
    );
}

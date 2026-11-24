use quantumforge::qasm::parse_qasm;
use std::fs;

#[test]
fn parse_mqt_ghz() {
    let qasm = fs::read_to_string("tests/ghz_mqt.qasm").unwrap();
    let circuit = parse_qasm(&qasm).unwrap();
    println!("Gates: {}", circuit.operations.len());
    let sv = quantumforge::StateVector::zero(4).unwrap();
    let sv = circuit.run(sv).unwrap();
    let norm: f64 = sv.amplitudes().iter().map(|a| a.norm_sqr()).sum();
    println!("Norm: {norm}");
    assert!((norm - 1.0).abs() < 1e-10);
}

const HEADER: &str = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n";

/// `;` terminates a statement in OpenQASM, not a line, so several statements may
/// share a physical line. Only the first used to be parsed and the rest were
/// silently dropped, while the CLI still reported success.
#[test]
fn statements_sharing_a_line_are_all_parsed() {
    let shared = format!("{HEADER}qreg q[2];\nh q[0];cx q[0],q[1];x q[1];\n");
    let circuit = parse_qasm(&shared).expect("parses");
    assert_eq!(
        circuit.operations().len(),
        3,
        "all three gates on the line must be parsed"
    );
}

#[test]
fn shared_line_matches_one_statement_per_line() {
    let shared = format!("{HEADER}qreg q[2];\nh q[0];cx q[0],q[1];x q[1];\n");
    let split = format!("{HEADER}qreg q[2];\nh q[0];\ncx q[0],q[1];\nx q[1];\n");
    let shared_circuit = parse_qasm(&shared).expect("parses");
    let split_circuit = parse_qasm(&split).expect("parses");
    assert_eq!(
        shared_circuit.to_json().expect("serializes"),
        split_circuit.to_json().expect("serializes")
    );

    // And the shared-line circuit must actually compute the intended unitary.
    let initial = quantumforge::StateVector::zero(2).unwrap();
    let state = shared_circuit.run(initial).expect("runs");
    let probabilities: Vec<f64> = state.probabilities();
    assert!(
        (probabilities[1] - 0.5).abs() < 1e-9 && (probabilities[2] - 0.5).abs() < 1e-9,
        "expected |01> and |10> at 50% each, got {probabilities:?}"
    );
}

#[test]
fn trailing_comment_after_a_shared_statement_is_ignored() {
    let qasm = format!("{HEADER}qreg q[2];\nh q[0]; cx q[0],q[1]; // entangle\n");
    let circuit = parse_qasm(&qasm).expect("parses");
    assert_eq!(circuit.operations().len(), 2);
}

/// A `gate` definition opens a brace, so its body must not be treated as a list
/// of top-level statements. The documented form spans several lines.
#[test]
fn multiline_gate_definition_is_left_intact() {
    let qasm = format!("{HEADER}qreg q[1];\ngate mygate a {{\n  x a;\n  h a;\n}}\nmygate q[0];\n");
    let circuit = parse_qasm(&qasm).expect("parses");
    assert_eq!(
        circuit.operations().len(),
        2,
        "the body must expand to X then H"
    );
}

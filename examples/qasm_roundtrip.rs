//! QASM 2.0 round-trip: import a circuit, export it back, re-import, and
//! confirm the operation sequence survives.
//!
//! The parser is line-oriented (one statement per line), and the exporter
//! re-emits canonical OpenQASM 2.0.
//!
//! Run with: cargo run --example qasm_roundtrip

use quantumforge::qasm::parse_qasm;
use quantumforge::qasm_export::export_qasm;

fn main() {
    let source = r#"OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"#;

    let circuit = parse_qasm(source).expect("source must parse");
    println!("imported: {} operations", circuit.operations().len());

    let exported = export_qasm(&circuit);
    println!("--- exported QASM ---\n{exported}");

    let reimported = parse_qasm(&exported).expect("exported QASM must re-parse");
    println!("re-imported: {} operations", reimported.operations().len());

    assert_eq!(
        circuit.operations().len(),
        reimported.operations().len(),
        "operation count must survive the round-trip"
    );
    assert!(
        exported.contains("cx q[0], q[1]"),
        "entangling gate must be preserved"
    );
    assert_eq!(
        exported.matches("measure").count(),
        2,
        "both measurements must be preserved"
    );
    // Note: whole-register measurement (`measure q -> c;`) is imported as a
    // single operation today; write per-qubit measures for exact round-trips.
    println!(
        "round-trip OK: {} operations preserved",
        reimported.operations().len()
    );
}

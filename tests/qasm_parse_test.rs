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
    println!("Norm: {}", norm);
    assert!((norm - 1.0).abs() < 1e-10);
}

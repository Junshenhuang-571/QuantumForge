use quantumforge::qasm::parse_qasm;
use quantumforge::StateVector;
use std::fs;
use std::time::Instant;

fn bench(name: &str, path: &str) {
    let qasm = fs::read_to_string(path).unwrap();
    let circuit = parse_qasm(&qasm).unwrap();
    let n = circuit.num_qubits();
    let gates = circuit.operations.len();
    let t0 = Instant::now();
    let sv = StateVector::zero(n).unwrap();
    let sv = circuit.run(sv).unwrap();
    let dt = t0.elapsed().as_secs_f64() * 1000.0;
    let norm: f64 = sv.amplitudes().iter().map(|a| a.norm_sqr()).sum();
    println!("{}, MQT Bench, {}, {}, {:.3}, {:.10}", name, n, gates, dt, norm);
}

fn main() {
    println!("Circuit, Source, Qubits, Gates, Time (ms), Norm");
    bench("GHZ", "tests/mqt_circuits/ghz_4.qasm");
    bench("GHZ", "tests/mqt_circuits/ghz_8.qasm");
    bench("Grover", "tests/mqt_circuits/grover_4.qasm");
    bench("BV", "tests/mqt_circuits/bv_4.qasm");
    bench("BV", "tests/mqt_circuits/bv_8.qasm");
    bench("DJ", "tests/mqt_circuits/dj_4.qasm");
    bench("DJ", "tests/mqt_circuits/dj_8.qasm");
    bench("QFT", "tests/mqt_circuits/qft_4.qasm");
    bench("QFT", "tests/mqt_circuits/qft_8.qasm");
}

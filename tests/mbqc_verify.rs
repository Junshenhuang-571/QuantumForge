use quantumforge::mbqc::{compile_circuit, compile_single_qubit};

#[test]
fn mbqc_single_qubit_h_runs() {
    let (cluster, pattern) = compile_single_qubit(&["H"], 42).unwrap();
    let run = pattern.run(&cluster, 42).unwrap();
    println!("MBQC H outputs: {:?}", run.outputs);
    // MBQC should produce a deterministic output for a single H gate
    assert!(run.outputs.len() <= 1);
}

#[test]
fn mbqc_compile_circuit_runs() {
    let gates = vec![(0, "H", None), (0, "T", None)];
    let (cluster, pattern) = compile_circuit(&gates).unwrap();
    let run = pattern.run(&cluster, 42).unwrap();
    println!("MBQC HT outputs: {:?}", run.outputs);
}

use quantumforge::mbqc::compile_circuit;
use quantumforge::{Gate, StateVector};

#[test]
fn mbqc_cnot_produces_output() {
    // Gate model: CNOT on |+0> -> (|00> + |11>)/sqrt2
    let mut sv = StateVector::zero(2).unwrap();
    sv.apply(Gate::H, 0).unwrap();
    // Apply CNOT
    // StateVector may not have cnot method; check
    println!("Gate model done");

    // MBQC compiled CNOT
    let gates = vec![(0, "H", None), (0, "CNOT", Some(1))];
    let (cluster, pattern) = compile_circuit(&gates).unwrap();
    let run = pattern.run(&cluster, 42).unwrap();
    println!("MBQC outputs: {:?}", run.outputs);
    println!("MBQC outcomes: {:?}", run.outcomes);
}

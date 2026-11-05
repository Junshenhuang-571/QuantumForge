use quantumforge::{StateVector, parse_circuit};

fn main() {
    let circuit = parse_circuit("Circuit[H[0], CNOT[0,1], Measure[0,0]]").unwrap();
    let initial = StateVector::zero(2).unwrap();
    let result = circuit.run_shots(&initial, 1000, 42).unwrap();

    println!("Bell-state measurement counts: {:?}", result.counts);
}

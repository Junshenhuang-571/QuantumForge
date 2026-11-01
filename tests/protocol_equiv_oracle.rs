use quantumforge::{Circuit, Gate, optimizer::*};

#[test]
fn optimization_preserves_equivalence() {
    // Test that self_optimize preserves unitary equivalence
    let mut circuit = Circuit::new();
    circuit.push(Gate::H, 0);
    circuit.push(Gate::H, 0); // cancel
    circuit.push(Gate::T, 1);
    circuit.push(Gate::Tdg, 1); // cancel
    circuit.push(Gate::Rz(0.5), 2);
    circuit.push(Gate::Rz(0.3), 2); // fuse
    circuit.push(Gate::I, 3); // remove
    circuit.cnot(0, 1);
    circuit.cnot(0, 1); // cancel

    let original = circuit.clone();
    let result = self_optimize(&mut circuit);

    assert!(
        circuits_equivalent(&original, &circuit, 4),
        "optimization broke equivalence"
    );
    assert!(result.optimized_ops <= result.original_ops);
    println!(
        "{} -> {} ops (2q: {} -> {}, depth: {} -> {})",
        result.original_ops,
        result.optimized_ops,
        result.original_two_qubit,
        result.optimized_two_qubit,
        result.original_depth,
        result.optimized_depth
    );
}

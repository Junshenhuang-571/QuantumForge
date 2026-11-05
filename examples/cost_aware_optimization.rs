use quantumforge::optimizer::{CircuitCostModel, OptimizeConfig, optimize_for_cost_model};
use quantumforge::{Circuit, Gate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut circuit = Circuit::new();
    circuit.push(Gate::H, 0);
    circuit.push(Gate::H, 0);
    circuit.cnot(0, 1);

    let model = CircuitCostModel {
        cnot_gate: 10.0,
        depth: 2.0,
        ..CircuitCostModel::default()
    };
    let (optimized, decision) = optimize_for_cost_model(&circuit, &OptimizeConfig::all(), &model)?;

    println!(
        "candidate cost: {} -> {}",
        decision.original_cost, decision.candidate_cost
    );
    println!(
        "optimization applied: {}; selected operations: {}",
        decision.applied,
        optimized.operations().len()
    );

    Ok(())
}

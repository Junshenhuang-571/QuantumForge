//! SubsCoRe: adaptive observation cost control for VQE (arXiv:2502.01704).
//!
//! ```bash
//! cargo run --release --example subscore
//! ```
//!
//! Runs the same VQE problem three ways under an identical total shot budget:
//! a fixed-shot NFT baseline, SubsCoRe-Bound, and SubsCoRe-Center. The point of
//! the paper is that allocating shots adaptively — loosely early, accurately
//! near convergence — reaches a lower energy for the same quantum budget.

use quantumforge::analytic::{ExactComplex, Pauli, PauliSum, PauliWord, Rational};
use quantumforge::subscore::{ReadoutStrategy, ShotHamiltonian, SubsCoReOptions, run_subscore};
use quantumforge::{ParameterizedCircuit, RotationAxis};

fn real(value: i128) -> ExactComplex {
    ExactComplex::real(Rational::new(value, 1).unwrap())
}

/// 1-layer hardware-efficient ansatz: Ry then Rx on every qubit, CNOT chain.
fn ansatz(qubits: usize) -> (ParameterizedCircuit, Vec<f64>) {
    let mut circuit = ParameterizedCircuit::new(2 * qubits);
    let mut params = Vec::new();
    for qubit in 0..qubits {
        circuit.rotation(RotationAxis::Y, qubit, params.len());
        params.push(0.3);
    }
    for qubit in 0..qubits {
        circuit.rotation(RotationAxis::X, qubit, params.len());
        params.push(-0.2);
    }
    for qubit in 0..qubits - 1 {
        circuit.cnot(qubit, qubit + 1);
    }
    (circuit, params)
}

/// Transverse-field Ising Hamiltonian `-sum ZZ - sum X`.
fn tfim(qubits: usize) -> PauliSum {
    let mut sum = PauliSum::zero(qubits);
    for qubit in 0..qubits - 1 {
        let mut word = vec![Pauli::I; qubits];
        word[qubit] = Pauli::Z;
        word[qubit + 1] = Pauli::Z;
        let _ = sum.add_term(real(-1), PauliWord::new(word));
    }
    for qubit in 0..qubits {
        let mut word = vec![Pauli::I; qubits];
        word[qubit] = Pauli::X;
        let _ = sum.add_term(real(-1), PauliWord::new(word));
    }
    sum
}

fn main() {
    let qubits = 3;
    let budget = 20_000;
    let (circuit, params) = ansatz(qubits);
    let objective = ShotHamiltonian::new(circuit, tfim(qubits), qubits);
    let counts = objective.parameter_counts();
    let start = objective.exact_energy(&params).unwrap();

    println!("SubsCoRe: adaptive observation cost control for VQE (arXiv:2502.01704)");
    println!(
        "{qubits} qubits, {} parameters, total budget {budget} shots",
        params.len()
    );
    println!("energy at the starting point: {start:.4}");
    println!();
    println!(
        "{:<22} {:>8} {:>8} {:>10} {:>8}",
        "method", "iters", "shots", "final E", "vs start"
    );

    let mut runs: Vec<(&str, SubsCoReOptions)> = Vec::new();
    // Fixed-shot NFT baseline: a constant observation variance, so the shot
    // count per point never adapts.
    let fixed_shots = 512;
    runs.push((
        "fixed-shot NFT",
        SubsCoReOptions {
            total_shot_budget: budget,
            initial_shots: fixed_shots,
            initial_core_variance: 1.0 / fixed_shots as f64,
            core_min: 1.0 / fixed_shots as f64,
            core_slope_scale: 0.0,
            strategy: ReadoutStrategy::Center,
            seed: 11,
            ..SubsCoReOptions::default()
        },
    ));
    runs.push((
        "SubsCoRe-Bound",
        SubsCoReOptions {
            total_shot_budget: budget,
            strategy: ReadoutStrategy::Bound,
            seed: 11,
            ..SubsCoReOptions::default()
        },
    ));
    runs.push((
        "SubsCoRe-Center",
        SubsCoReOptions {
            total_shot_budget: budget,
            strategy: ReadoutStrategy::Center,
            seed: 11,
            ..SubsCoReOptions::default()
        },
    ));

    for (name, options) in runs {
        let outcome = run_subscore(
            |point, shots, seed| objective.observe(point, shots, seed),
            &params,
            &counts,
            &options,
        );
        let energy = objective.exact_energy(&outcome.best).unwrap();
        println!(
            "{name:<22} {:>8} {:>8} {:>10.4} {:>8.4}",
            outcome.iterations,
            outcome.shots_used,
            energy,
            energy - start
        );
        if name == "SubsCoRe-Center" {
            let first = outcome.shots_per_iteration.first().copied().unwrap_or(0);
            let last = outcome.shots_per_iteration.last().copied().unwrap_or(0);
            println!("  shots per iteration: {first} early -> {last} late (adaptive)");
        }
    }

    println!();
    println!("same budget throughout; the adaptive methods move shots from the");
    println!("early, rapidly-improving phase to the converging phase.");
}

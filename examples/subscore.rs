//! SubsCoRe: adaptive observation cost control for VQE (arXiv:2502.01704).
//!
//! ```bash
//! cargo run --release --example subscore
//! ```
//!
//! Runs the same VQE problem under one total shot budget and compares a
//! fixed-shot NFT baseline, SubsCoRe-Bound, SubsCoRe-Center with the
//! zero-temperature update (posterior-mean minimizer), and SubsCoRe-Center
//! with the finite-temperature update (Thompson sampling from the GP
//! posterior). Every number is averaged over several seeds: one shot budget on
//! a multimodal landscape is a coin flip.

use quantumforge::analytic::{ExactComplex, Pauli, PauliSum, PauliWord, Rational};
use quantumforge::subscore::{
    Observation, ReadoutStrategy, ShotHamiltonian, SubsCoReOptions, SubsCoReOutcome, UpdateRule,
    run_subscore,
};
use quantumforge::{ParameterizedCircuit, RotationAxis};

fn real(value: i128) -> ExactComplex {
    ExactComplex::real(Rational::new(value, 1).unwrap())
}

/// Hardware-efficient ansatz: per layer, Ry then Rx on every qubit, CNOT chain.
fn ansatz(qubits: usize, layers: usize) -> (ParameterizedCircuit, Vec<f64>) {
    let mut circuit = ParameterizedCircuit::new(2 * qubits * layers);
    let mut params = Vec::new();
    for _ in 0..layers {
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
    }
    (circuit, params)
}

/// Transverse-field Ising Hamiltonian `-sum ZZ -sum X`.
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
    let qubits = 4;
    let layers = 2;
    let budget = 40_000;
    let seeds = [1u64, 2, 3, 4, 5, 6];

    let (circuit, params) = ansatz(qubits, layers);
    let objective = ShotHamiltonian::new(circuit, tfim(qubits), qubits);
    let counts = objective.parameter_counts();
    let start = objective.exact_energy(&params).unwrap();
    let eta_sq = objective.single_shot_variance();

    println!("SubsCoRe: adaptive observation cost control for VQE (arXiv:2502.01704)");
    println!(
        "{qubits} qubits, {} parameters, {layers} layers, budget {budget} shots, {} seeds",
        params.len(),
        seeds.len()
    );
    println!("start energy {start:.4}, effective single-shot variance eta^2 = {eta_sq:.1}");
    println!();
    println!(
        "{:<26} {:>8} {:>8} {:>10} {:>8}",
        "method", "mean E", "best E", "vs start", "iters"
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
            initial_core_variance: eta_sq / fixed_shots as f64,
            core_min: eta_sq / fixed_shots as f64,
            core_slope_scale: 0.0,
            eta_sq,
            strategy: ReadoutStrategy::Center,
            ..SubsCoReOptions::default()
        },
    ));
    for (name, strategy, update_rule, exploration) in [
        (
            "Bound + mean",
            ReadoutStrategy::Bound,
            UpdateRule::Mean,
            1.0,
        ),
        (
            "Center + mean",
            ReadoutStrategy::Center,
            UpdateRule::Mean,
            1.0,
        ),
        (
            "Center + Thompson",
            ReadoutStrategy::Center,
            UpdateRule::Thompson,
            1.0,
        ),
        (
            "Center + guarded Thompson",
            ReadoutStrategy::Center,
            UpdateRule::GuardedThompson,
            1.0,
        ),
        (
            "Center + guarded Th. x1.5",
            ReadoutStrategy::Center,
            UpdateRule::GuardedThompson,
            1.5,
        ),
    ] {
        runs.push((
            name,
            SubsCoReOptions {
                total_shot_budget: budget,
                eta_sq,
                strategy,
                update_rule,
                exploration,
                initial_shots: 128,
                // The paper's Table 1 quotes the threshold as a shot count
                // (512 to start, 4096 at the floor); convert with eta^2.
                initial_core_variance: eta_sq / 512.0,
                core_min: eta_sq / 4096.0,
                ..SubsCoReOptions::default()
            },
        ));
    }

    for (name, options) in runs {
        let mut energies = Vec::new();
        let mut iterations = Vec::new();
        let mut early = 0usize;
        let mut late = 0usize;
        for &seed in &seeds {
            let outcome = run_trial(&objective, &params, &counts, &options, seed);
            energies.push(objective.exact_energy(&outcome.best).unwrap());
            iterations.push(outcome.iterations);
            early = outcome.shots_per_iteration.first().copied().unwrap_or(0);
            late = outcome.shots_per_iteration.last().copied().unwrap_or(0);
        }
        let mean = energies.iter().sum::<f64>() / energies.len() as f64;
        let best = energies.iter().cloned().fold(f64::INFINITY, f64::min);
        let mean_iters = iterations.iter().sum::<usize>() / iterations.len();
        println!(
            "{name:<26} {:>8.4} {:>8.4} {:>10.4} {:>8}",
            mean,
            best,
            mean - start,
            mean_iters
        );
        if name.contains("Thompson") {
            println!("  shots per iteration: {early} early -> {late} late");
        }
    }

    println!();
    println!("the shot budget is the thermostat: T = eta^2 / N. Center moves shots");
    println!("from the hot early phase to the cold converging phase, and a Thompson");
    println!("update spends the remaining temperature on exploration.");
}

/// One seeded trial.
fn run_trial(
    objective: &ShotHamiltonian,
    params: &[f64],
    counts: &[usize],
    options: &SubsCoReOptions,
    seed: u64,
) -> SubsCoReOutcome {
    let options = SubsCoReOptions {
        seed,
        ..options.clone()
    };
    run_subscore(
        |point, shots, run_seed| -> Observation { objective.observe(point, shots, run_seed) },
        params,
        counts,
        &options,
    )
}

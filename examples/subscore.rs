//! SubsCoRe: adaptive observation cost control for VQE (arXiv:2502.01704).
//!
//! ```bash
//! cargo run --release --example subscore
//! ```
//!
//! One campaign, seven rules, twelve seeds: a fixed-shot NFT baseline, the
//! paper's own update (GP-mean argmin) under both shot-allocation rules, the two
//! thermodynamic acquisitions, and posterior sampling. Every number is the exact
//! energy of the returned parameters, with the standard deviation across seeds.

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
    let qubits = 4;
    let layers = 2;
    let budget = 40_000;
    let seeds = [1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

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
        "{:<30} {:>9} {:>8} {:>8}",
        "method", "mean E", "sd", "iters"
    );

    let fixed_shots = 512.0;
    let mut rows: Vec<(&str, SubsCoReOptions)> = vec![(
        "fixed-shot NFT",
        SubsCoReOptions {
            total_shot_budget: budget,
            eta_sq,
            initial_shots: fixed_shots as usize,
            initial_core_variance: eta_sq / fixed_shots,
            core_min: eta_sq / fixed_shots,
            core_slope_scale: 0.0,
            strategy: ReadoutStrategy::Center,
            ..SubsCoReOptions::default()
        },
    )];
    for (name, strategy, update_rule) in [
        ("Bound + mean", ReadoutStrategy::Bound, UpdateRule::Mean),
        (
            "Bound + lower conf. bound",
            ReadoutStrategy::Bound,
            UpdateRule::LowerConfidenceBound,
        ),
        (
            "Bound + free energy",
            ReadoutStrategy::Bound,
            UpdateRule::FreeEnergy,
        ),
        ("Center + mean", ReadoutStrategy::Center, UpdateRule::Mean),
        (
            "Center + Thompson",
            ReadoutStrategy::Center,
            UpdateRule::Thompson,
        ),
        (
            "Center + guarded Thompson",
            ReadoutStrategy::Center,
            UpdateRule::GuardedThompson,
        ),
    ] {
        rows.push((
            name,
            SubsCoReOptions {
                total_shot_budget: budget,
                eta_sq,
                initial_shots: 128,
                // The paper's Table 1 quotes the threshold as a shot count
                // (512 to start, 4096 at the floor); convert with eta^2.
                initial_core_variance: eta_sq / 512.0,
                core_min: eta_sq / 4096.0,
                strategy,
                update_rule,
                ..SubsCoReOptions::default()
            },
        ));
    }

    for (name, options) in rows {
        let mut energies = Vec::new();
        let mut iterations = Vec::new();
        for &seed in &seeds {
            let outcome = run_trial(&objective, &params, &counts, &options, seed);
            energies.push(objective.exact_energy(&outcome.best).unwrap());
            iterations.push(outcome.iterations);
        }
        let n = energies.len() as f64;
        let mean = energies.iter().sum::<f64>() / n;
        let sd = (energies.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        println!(
            "{name:<30} {mean:>9.4} {sd:>8.3} {:>8}",
            iterations.iter().sum::<usize>() / iterations.len()
        );
    }

    println!();
    println!("the shot budget is a thermostat: the noise is a variance, so the");
    println!("temperature in energy units is T = eta/sqrt(N). Bound + mean cools");
    println!("adaptively; the lower-confidence-bound rule adds T*sigma with that same");
    println!("temperature and lands on the quench. The literal Gibbs entropy makes the");
    println!("bonus linear in ln sigma, which is worse. Posterior sampling uses the");
    println!("surrogate's ignorance instead, which is far hotter and loses.");
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

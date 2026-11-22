//! Noise-driven VQE: annealed Langevin versus plain gradient descent.
//!
//! ```bash
//! cargo run --release --example langevin_vqe
//! ```

use quantumforge::analytic::{ExactComplex, Pauli, PauliSum, PauliWord, Rational};
use quantumforge::noisy_optimizer::{
    LangevinVqeOptions, annealed_langevin_vqe, gradient_descent_vqe,
};
use quantumforge::{ParameterizedCircuit, RotationAxis};

fn real(value: i128) -> ExactComplex {
    ExactComplex::real(Rational::new(value, 1).unwrap())
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn sd(values: &[f64]) -> f64 {
    let m = mean(values);
    (values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / values.len() as f64).sqrt()
}

fn main() {
    // Frustrated 3-qubit TFIM on a triangle.
    let mut hamiltonian = PauliSum::zero(3);
    for (first, second) in [(0usize, 1usize), (1, 2), (2, 0)] {
        let mut word = vec![Pauli::I; 3];
        word[first] = Pauli::Z;
        word[second] = Pauli::Z;
        let _ = hamiltonian.add_term(real(-1), PauliWord::new(word));
    }
    for qubit in 0..3 {
        let mut word = vec![Pauli::I; 3];
        word[qubit] = Pauli::X;
        let _ = hamiltonian.add_term(real(-1), PauliWord::new(word));
    }

    let operator = hamiltonian.to_sparse_operator().expect("sparse operator");
    let exact = operator
        .lowest_eigenpairs(1, 1e-10, 200)
        .expect("ground state");
    let ground = exact.eigenvalues()[0];

    let mut circuit = ParameterizedCircuit::new(6);
    let mut index = 0;
    for _ in 0..2 {
        for qubit in 0..3 {
            circuit.rotation(RotationAxis::Y, qubit, index);
            index += 1;
        }
        for qubit in 0..3 {
            circuit.cnot(qubit, (qubit + 1) % 3);
        }
    }

    println!("frustrated 3-qubit TFIM on a triangle, exact ground energy {ground:.6}");
    println!("{} seeds, plain GD 400 steps vs Langevin 800 steps\n", 12);

    let mut plain = Vec::new();
    let mut langevin = Vec::new();
    for seed in 0..12u64 {
        let initial: Vec<f64> = (0..6)
            .map(|k| 0.15 * ((seed + 1) as f64) * (k as f64 + 1.0))
            .collect();
        let (_, plain_energy) =
            gradient_descent_vqe(&circuit, &hamiltonian, 3, &initial, 0.05, 400).expect("GD");
        let options = LangevinVqeOptions {
            learning_rate: 0.05,
            steps: 800,
            initial_temperature: 0.5,
            final_temperature: 0.005,
            seed,
        };
        let outcome =
            annealed_langevin_vqe(&circuit, &hamiltonian, 3, &initial, &options).expect("Langevin");
        plain.push(plain_energy);
        langevin.push(outcome.best_energy);
    }

    println!(
        "plain GD    : mean {:.6}, sd {:.6}, gap to ground {:.6}, within 0.01 {}/12",
        mean(&plain),
        sd(&plain),
        mean(&plain) - ground,
        plain.iter().filter(|e| **e < ground + 0.01).count()
    );
    println!(
        "Langevin    : mean {:.6}, sd {:.6}, gap to ground {:.6}, within 0.01 {}/12",
        mean(&langevin),
        sd(&langevin),
        mean(&langevin) - ground,
        langevin.iter().filter(|e| **e < ground + 0.01).count()
    );
    println!(
        "plain stuck at the shallow minimum: {}/12",
        plain.iter().filter(|e| **e > ground + 0.1).count()
    );
    println!(
        "Langevin stuck at the shallow minimum: {}/12",
        langevin.iter().filter(|e| **e > ground + 0.1).count()
    );
    let paired: Vec<f64> = langevin
        .iter()
        .zip(plain.iter())
        .map(|(langevin_energy, plain_energy)| langevin_energy - plain_energy)
        .collect();
    println!(
        "\npaired (Langevin - plain): mean {:.6}, sd {:.6}, better in {}/12 seeds",
        mean(&paired),
        sd(&paired),
        paired.iter().filter(|d| **d < 0.0).count()
    );
    println!("per-seed plain   : {plain:?}");
    println!("per-seed Langevin: {langevin:?}");
}

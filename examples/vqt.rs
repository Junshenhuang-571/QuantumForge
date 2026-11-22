//! Variational quantum thermalization on a 2-qubit interacting Hamiltonian.
//!
//! ```bash
//! cargo run --release --example vqt
//! ```

use quantumforge::analytic::{ExactComplex, Pauli, PauliSum, PauliWord, Rational};
use quantumforge::vqt::{VqtOptions, trace_distance, variational_thermalize};

fn real(value: i128) -> ExactComplex {
    ExactComplex::real(Rational::new(value, 1).unwrap())
}

fn main() {
    // H = -Z0 - Z1 - Z0Z1 - (1/2) X0X1, an interacting 2-qubit Hamiltonian.
    let mut hamiltonian = PauliSum::zero(2);
    let _ = hamiltonian.add_term(real(-1), PauliWord::new(vec![Pauli::Z, Pauli::I]));
    let _ = hamiltonian.add_term(real(-1), PauliWord::new(vec![Pauli::I, Pauli::Z]));
    let _ = hamiltonian.add_term(real(-1), PauliWord::new(vec![Pauli::Z, Pauli::Z]));
    let _ = hamiltonian.add_term(
        ExactComplex::real(Rational::new(-1, 2).unwrap()),
        PauliWord::new(vec![Pauli::X, Pauli::X]),
    );

    let options = VqtOptions {
        system_qubits: 2,
        layers: 4,
        temperature: 1.0,
        learning_rate: 0.05,
        steps: 400,
        finite_difference_step: 1e-3,
    };
    let outcome = variational_thermalize(&hamiltonian, &options).expect("VQT must run");

    println!(
        "variational quantum thermalization, T = {}",
        options.temperature
    );
    println!("  F prepared = {:.6}", outcome.free_energy);
    println!("  F exact    = {:.6}", outcome.exact_free_energy);
    println!(
        "  gap        = {:.6}  ({:.2}% of |F|)",
        outcome.free_energy - outcome.exact_free_energy,
        100.0 * (outcome.free_energy - outcome.exact_free_energy).abs()
            / outcome.exact_free_energy.abs()
    );
    println!("  trace dist = {:.6}", outcome.trace_distance);
    println!(
        "  S prepared = {:.6}, S exact = {:.6}",
        outcome.state.von_neumann_entropy().unwrap(),
        outcome.exact_state.von_neumann_entropy().unwrap()
    );

    let initial = outcome.history.first().copied().unwrap_or(f64::NAN);
    let middle = outcome.history[outcome.history.len() / 2];
    println!(
        "  F history  = {initial:.6} -> {middle:.6} -> {:.6}",
        outcome.free_energy
    );

    // The trace distance must be small; print the diagonal so the mixing is visible.
    println!("\n  prepared diagonal:");
    for index in 0..outcome.state.dimension() {
        println!(
            "    |{index:02b}>  rho = {:.6}   exact = {:.6}",
            outcome.state.entries()[index * outcome.state.dimension() + index].re,
            outcome.exact_state.entries()[index * outcome.exact_state.dimension() + index].re
        );
    }
    let _ = trace_distance;
}

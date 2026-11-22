//! Finite-temperature Gibbs states, free energy, and sampling.
//!
//! ```bash
//! cargo run --release --example thermal_states
//! ```

use quantumforge::analytic::{ExactComplex, Pauli, PauliSum, PauliWord, Rational};
use quantumforge::thermal::{
    expectation, free_energy, gibbs_probabilities, gibbs_state, mean_energy, sample_gibbs,
};

fn real(value: i128) -> ExactComplex {
    ExactComplex::real(Rational::new(value, 1).unwrap())
}

fn main() {
    // Non-interacting field H = (Z0 + Z1)/2, so the exact Gibbs state is a
    // product and can be checked by hand.
    let mut hamiltonian = PauliSum::zero(2);
    let _ = hamiltonian.add_term(real(1), PauliWord::new(vec![Pauli::Z, Pauli::I]));
    let _ = hamiltonian.add_term(real(1), PauliWord::new(vec![Pauli::I, Pauli::Z]));

    for beta in [0.5, 1.0, 2.0] {
        let rho = gibbs_state(&hamiltonian, beta).unwrap();
        println!("beta = {beta}");
        println!(
            "  <H>      = {:.6}",
            expectation(&rho, &hamiltonian).unwrap()
        );
        println!(
            "  <H> exact= {:.6}",
            mean_energy(&hamiltonian, beta).unwrap()
        );
        println!(
            "  F        = {:.6}",
            free_energy(&hamiltonian, beta).unwrap()
        );
        println!("  S        = {:.6}", rho.von_neumann_entropy().unwrap());
        println!("  purity   = {:.6}", rho.purity());
    }

    // The Gibbs sampler must reproduce the exact diagonal.
    let beta = 1.0;
    let exact = gibbs_probabilities(&hamiltonian, beta).unwrap();
    let shots = 40_000;
    let counts = sample_gibbs(&hamiltonian, beta, shots, 42).unwrap();
    println!("\nsampler check at beta = {beta} ({shots} shots)");
    println!("  {:<8} {:>12} {:>12}", "state", "sampled", "exact");
    for (index, exact_probability) in exact.iter().enumerate() {
        let bits = vec![u8::from(index & 1 != 0), u8::from(index & 2 != 0)];
        let sampled = counts.get(&bits).copied().unwrap_or(0) as f64 / shots as f64;
        println!(
            "  |{}{}>   {:>12.6} {:>12.6}",
            bits[1], bits[0], sampled, exact_probability
        );
    }

    // The single-qubit Gibbs state is exactly what the ThermalDamping channel
    // prepares; the collisional channel is the bath-side view of the same map.
    let mut single = PauliSum::zero(1);
    let _ = single.add_term(
        ExactComplex::real(Rational::new(-1, 2).unwrap()),
        PauliWord::new(vec![Pauli::Z]),
    );
    let rho = gibbs_state(&single, 1.0).unwrap();
    println!(
        "\nH = -Z/2 at beta = 1: p1 = {:.9}",
        rho.probability_one(0).unwrap()
    );
    println!(
        "thermal damping fixed point: p1 = 1/(1+e) = {:.9}",
        1.0 / (1.0 + std::f64::consts::E)
    );
}

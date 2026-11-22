//! Finite-temperature Gibbs states, free energy, and sampling.

use quantumforge::analytic::{ExactComplex, Pauli, PauliSum, PauliWord, Rational};
use quantumforge::thermal::{
    expectation, free_energy, gibbs_probabilities, gibbs_state, mean_energy, sample_gibbs,
};
use quantumforge::{DensityMatrix, NoiseChannel, StateVector};

fn real(value: i128) -> ExactComplex {
    ExactComplex::real(Rational::new(value, 1).unwrap())
}

/// H = (Delta/2) Z on one qubit, with `delta = 1`.
fn single_qubit_field() -> PauliSum {
    let half = ExactComplex::real(Rational::new(1, 2).unwrap());
    let mut sum = PauliSum::zero(1);
    let _ = sum.add_term(half, PauliWord::new(vec![Pauli::Z]));
    sum
}

#[test]
fn gibbs_state_of_a_single_qubit_matches_the_analytic_result() {
    let hamiltonian = single_qubit_field();
    let beta = 1.0;
    let rho = gibbs_state(&hamiltonian, beta).unwrap();
    // H = Z/2 puts |1> below |0>, so p1 = 1 / (1 + e^{-beta}).
    let expected = 1.0 / (1.0 + (-beta).exp());
    assert!((rho.probability_one(0).unwrap() - expected).abs() < 1e-12);
    // It is diagonal.
    assert!(rho.entries()[1].norm_sqr().sqrt() < 1e-12);
    // Entropy is the binary entropy of p1.
    let p = expected;
    let entropy = -(p * p.ln() + (1.0 - p) * (1.0 - p).ln());
    assert!((rho.von_neumann_entropy().unwrap() - entropy).abs() < 1e-12);
}

#[test]
fn gibbs_state_matches_the_thermal_damping_fixed_point() {
    // H = -Z/2 has |1> above |0> by Delta = 1, matching the ThermalDamping
    // convention (Delta = E_1 - E_0).
    let half = ExactComplex::real(Rational::new(-1, 2).unwrap());
    let mut hamiltonian = PauliSum::zero(1);
    let _ = hamiltonian.add_term(half, PauliWord::new(vec![Pauli::Z]));
    let exact = gibbs_state(&hamiltonian, 1.0).unwrap();
    let mut rho = DensityMatrix::from_state(&StateVector::zero(1).unwrap());
    for _ in 0..400 {
        rho.apply_channel(
            NoiseChannel::ThermalDamping {
                gamma_down: 0.2,
                beta_delta: 1.0,
            },
            0,
        )
        .unwrap();
    }
    assert!((rho.probability_one(0).unwrap() - exact.probability_one(0).unwrap()).abs() < 1e-12);
    // Both equal 1/(1+e^{1}) = 0.2689414213699951.
    assert!((exact.probability_one(0).unwrap() - 0.2689414213699951).abs() < 1e-12);
}

#[test]
fn free_energy_satisfies_f_equals_energy_minus_t_entropy() {
    let hamiltonian = single_qubit_field();
    let beta = 0.7;
    let rho = gibbs_state(&hamiltonian, beta).unwrap();
    let energy = expectation(&rho, &hamiltonian).unwrap();
    let entropy = rho.von_neumann_entropy().unwrap();
    let temperature = 1.0 / beta;
    let f = energy - temperature * entropy;
    assert!((f - free_energy(&hamiltonian, beta).unwrap()).abs() < 1e-8);
    // And the closed form F = -ln(2 cosh(beta/2)) / beta.
    let closed = -(2.0 * (beta / 2.0).cosh()).ln() / beta;
    assert!((f - closed).abs() < 1e-8);
    assert!((mean_energy(&hamiltonian, beta).unwrap() - energy).abs() < 1e-12);
}

#[test]
fn infinite_temperature_is_the_maximally_mixed_state() {
    let hamiltonian = single_qubit_field();
    let rho = gibbs_state(&hamiltonian, 0.0).unwrap();
    assert!((rho.probability_one(0).unwrap() - 0.5).abs() < 1e-12);
    assert!((rho.von_neumann_entropy().unwrap() - std::f64::consts::LN_2).abs() < 1e-12);
    let probabilities = gibbs_probabilities(&hamiltonian, 0.0).unwrap();
    assert!((probabilities[0] - 0.5).abs() < 1e-12);
    assert!((probabilities[1] - 0.5).abs() < 1e-12);
}

#[test]
fn gibbs_sampler_reproduces_the_exact_distribution() {
    let hamiltonian = single_qubit_field();
    let beta = 1.0;
    let shots = 40_000;
    let counts = sample_gibbs(&hamiltonian, beta, shots, 7).unwrap();
    let ones: usize = counts.get(&vec![1u8]).copied().unwrap_or(0);
    let observed = ones as f64 / shots as f64;
    let expected = 1.0 / (1.0 + (-beta).exp());
    // Standard error ~ 0.0024; allow 6 sigma.
    assert!(
        (observed - expected).abs() < 0.015,
        "{observed} vs {expected}"
    );
}

#[test]
fn non_interacting_gibbs_state_factorises() {
    // H = (Z0 + Z1)/2 at beta = 1 is a product state.
    let mut hamiltonian = PauliSum::zero(2);
    let mut word0 = vec![Pauli::I, Pauli::I];
    word0[0] = Pauli::Z;
    let mut word1 = vec![Pauli::I, Pauli::I];
    word1[1] = Pauli::Z;
    let _ = hamiltonian.add_term(real(1), PauliWord::new(word0));
    let _ = hamiltonian.add_term(real(1), PauliWord::new(word1));

    let rho = gibbs_state(&hamiltonian, 1.0).unwrap();
    let zz = {
        let mut word = vec![Pauli::I, Pauli::I];
        word[0] = Pauli::Z;
        word[1] = Pauli::Z;
        let mut sum = PauliSum::zero(2);
        let _ = sum.add_term(real(1), PauliWord::new(word));
        expectation(&rho, &sum).unwrap()
    };
    let mut h0 = PauliSum::zero(2);
    let _ = h0.add_term(real(1), PauliWord::new(vec![Pauli::Z, Pauli::I]));
    let z0 = expectation(&rho, &h0).unwrap();
    let mut h1 = PauliSum::zero(2);
    let _ = h1.add_term(real(1), PauliWord::new(vec![Pauli::I, Pauli::Z]));
    let z1 = expectation(&rho, &h1).unwrap();
    // <Z0 Z1> = <Z0> <Z1> for a product state.
    assert!((zz - z0 * z1).abs() < 1e-10, "{zz} vs {z0}*{z1}");
    // No coherences.
    assert!(rho.entries()[1].norm_sqr().sqrt() < 1e-12);
    assert!(rho.entries()[2].norm_sqr().sqrt() < 1e-12);
}

#[test]
fn x_field_gibbs_state_is_not_diagonal() {
    // H = X/2 has eigenvectors (|0> +- |1>)/sqrt(2); the Gibbs state at beta > 0
    // is a superposition mixture with non-zero off-diagonal.
    let mut hamiltonian = PauliSum::zero(1);
    let _ = hamiltonian.add_term(real(1), PauliWord::new(vec![Pauli::X]));
    let rho = gibbs_state(&hamiltonian, 1.0).unwrap();
    assert!(rho.entries()[1].norm_sqr().sqrt() > 1e-6);
    // Populations are equal by symmetry.
    assert!((rho.probability_one(0).unwrap() - 0.5).abs() < 1e-12);
}

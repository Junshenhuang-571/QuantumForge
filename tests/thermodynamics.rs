//! Tests for the thermodynamics of the noise channels and the optimizers.

use quantumforge::analytic::{ExactComplex, Pauli, PauliSum, PauliWord, Rational};
use quantumforge::collisional::{collision_rate, thermalize_collisions};
use quantumforge::dissipative::{entropy_production, heat_absorbed};
use quantumforge::noisy_optimizer::{langevin_samples, shot_noise_temperature};
use quantumforge::thermal::{exact_energy_expectation, exact_gibbs_weights};
use quantumforge::{Complex64, DensityMatrix, NoiseChannel, SmallRng, StateVector};

fn real(value: i128) -> ExactComplex {
    ExactComplex::real(Rational::new(value, 1).unwrap())
}

fn single_qubit_hamiltonian() -> PauliSum {
    // H = -Z/2, so E_0 = -1/2, E_1 = +1/2 and Delta = E_1 - E_0 = 1.
    let mut hamiltonian = PauliSum::zero(1);
    let _ = hamiltonian.add_term(
        ExactComplex::real(Rational::new(-1, 2).unwrap()),
        PauliWord::new(vec![Pauli::Z]),
    );
    hamiltonian
}

fn state_with_p1(prob_one: f64) -> DensityMatrix {
    let mut amplitudes = vec![Complex64::ZERO; 2];
    amplitudes[0] = Complex64::new((1.0 - prob_one).sqrt(), 0.0);
    amplitudes[1] = Complex64::new(prob_one.sqrt(), 0.0);
    DensityMatrix::from_state(&StateVector::from_amplitudes(amplitudes).unwrap())
}

fn diagonal_state_with_p1(prob_one: f64) -> DensityMatrix {
    let mut rho = state_with_p1(prob_one);
    let entries = rho.entries().to_vec();
    let mut diagonal = vec![Complex64::ZERO; 4];
    diagonal[0] = entries[0];
    diagonal[3] = entries[3];
    rho = DensityMatrix::from_entries(diagonal, 1);
    rho
}

#[test]
fn exact_gibbs_weights_are_exact_and_sum_to_one() {
    let base = Rational::new(1, 2).unwrap();
    let (partition, probabilities) = exact_gibbs_weights(&[0, 1], &base).unwrap();
    assert_eq!(partition, Rational::new(3, 2).unwrap());
    assert_eq!(probabilities[0], Rational::new(2, 3).unwrap());
    assert_eq!(probabilities[1], Rational::new(1, 3).unwrap());
    let sum = probabilities[0].checked_add(probabilities[1]).unwrap();
    assert_eq!(sum, Rational::new(1, 1).unwrap());
    assert_eq!(
        exact_energy_expectation(&[0, 1], &probabilities).unwrap(),
        Rational::new(1, 3).unwrap()
    );
}

#[test]
fn exact_gibbs_weights_handle_negative_energies() {
    let base = Rational::new(2, 1).unwrap();
    let (partition, probabilities) = exact_gibbs_weights(&[-1, 1], &base).unwrap();
    // weights 2 and 1/2, Z = 5/2, p = (1/5, 4/5)
    assert_eq!(partition, Rational::new(5, 2).unwrap());
    assert_eq!(probabilities[0], Rational::new(1, 5).unwrap());
    assert_eq!(probabilities[1], Rational::new(4, 5).unwrap());
}

#[test]
fn thermal_damping_has_non_negative_entropy_production() {
    let hamiltonian = single_qubit_hamiltonian();
    let mut rho = state_with_p1(0.9);
    let temperature = 1.0;
    for _ in 0..12 {
        let before = rho.clone();
        rho.apply_channel(
            NoiseChannel::ThermalDamping {
                gamma_down: 0.1,
                beta_delta: 1.0,
            },
            0,
        )
        .unwrap();
        let production = entropy_production(&before, &rho, &hamiltonian, temperature).unwrap();
        assert!(
            production >= -1e-12,
            "detailed-balance channel produced {production}"
        );
    }
}

#[test]
fn amplitude_damping_is_a_refrigerator_that_purifies() {
    let hamiltonian = single_qubit_hamiltonian();
    let mut rho = state_with_p1(0.9);
    let mut last_production = 0.0;
    for _ in 0..200 {
        let before = rho.clone();
        rho.apply_channel(NoiseChannel::AmplitudeDamping { gamma: 0.1 }, 0)
            .unwrap();
        last_production = entropy_production(&before, &rho, &hamiltonian, 1.0).unwrap();
    }
    // The state is pumped toward |0>, so the entropy falls: a driven process,
    // not a violation of the second law.
    assert!(rho.probability_one(0).unwrap() < 1e-6);
    assert!(
        last_production < 0.0,
        "refrigerator Sigma = {last_production}"
    );
}

#[test]
fn heat_absorbed_matches_the_energy_change() {
    let hamiltonian = single_qubit_hamiltonian();
    let before = state_with_p1(0.9);
    let after = state_with_p1(0.5);
    let heat = heat_absorbed(&before, &after, &hamiltonian).unwrap();
    // <H> = p1 - 1/2; going 0.9 -> 0.5 lowers the energy by 0.4.
    assert!((heat - (-0.4)).abs() < 1e-12, "heat {heat}");
}

#[test]
fn collisional_rate_and_fixed_point() {
    let phi = 0.35;
    assert!((collision_rate(phi) - 0.882421094).abs() < 1e-8);
    let bath = diagonal_state_with_p1(0.2689414213699951);
    let mut rho = diagonal_state_with_p1(0.9);
    thermalize_collisions(&mut rho, &bath, phi, 13).unwrap();
    let rate = collision_rate(phi);
    let predicted =
        bath.probability_one(0).unwrap() + (0.9 - bath.probability_one(0).unwrap()) * rate.powi(13);
    assert!(
        (rho.probability_one(0).unwrap() - predicted).abs() < 1e-10,
        "measured {}, predicted {}",
        rho.probability_one(0).unwrap(),
        predicted
    );
}

#[test]
fn shot_noise_temperature_is_the_estimator_sd() {
    let mut hamiltonian = PauliSum::zero(1);
    let _ = hamiltonian.add_term(real(1), PauliWord::new(vec![Pauli::Z]));
    // sum_w |c_w|^2 = 1, so sd = sqrt(1 / shots).
    assert!((shot_noise_temperature(&hamiltonian, 4).unwrap() - 0.5).abs() < 1e-12);
    assert!((shot_noise_temperature(&hamiltonian, 100).unwrap() - 0.1).abs() < 1e-12);
}

fn double_well(x: &[f64]) -> f64 {
    (x[0] * x[0] - 1.0).powi(2) + 0.5 * x[0]
}

fn double_well_gradient(x: &[f64], out: &mut [f64]) {
    out[0] = 4.0 * x[0].powi(3) - 4.0 * x[0] + 0.5;
}

#[test]
fn langevin_stationary_distribution_matches_boltzmann() {
    // Well minima: deep at x ~ -1.07 (f ~ -0.514), shallow at x ~ +0.93
    // (f ~ +0.483). The population ratio must be exp(Delta f / T).
    let temperature = 0.5;
    let samples = langevin_samples(double_well_gradient, &[0.0], temperature, 200_000, 0.01, 7);
    // Discard the transient.
    let steady = &samples[samples.len() / 2..];
    let deep = steady.iter().filter(|x| x[0] < 0.0).count() as f64;
    let shallow = steady.len() as f64 - deep;
    let measured = deep / shallow;
    let predicted = (0.997 / temperature).exp();
    // A loose bound: the claim is the exponential ratio, not the prefactor.
    assert!(
        (measured - predicted).abs() < 0.25 * predicted,
        "measured ratio {measured}, predicted {predicted}"
    );
}

#[test]
fn small_rng_draws_are_not_correlated_between_seeds() {
    let mut first = SmallRng::seeded(1);
    let mut second = SmallRng::seeded(2);
    let a: f64 = (0..200).map(|_| first.next_f64()).sum();
    let b: f64 = (0..200).map(|_| second.next_f64()).sum();
    assert!((a - b).abs() > 1.0, "seeds 1 and 2 gave {a} and {b}");
    let _ = double_well;
}

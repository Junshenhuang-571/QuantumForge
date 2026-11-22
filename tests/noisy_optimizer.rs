//! Noise-driven optimization: annealed Langevin dynamics.

use quantumforge::noisy_optimizer::{
    LangevinOptions, annealed_langevin, finite_difference_gradient, gradient_descent,
};

/// Asymmetric double well: the deep minimum is near x = -1.07, the shallow one
/// near x = +0.93.
fn double_well(x: &[f64]) -> f64 {
    let x = x[0];
    (x * x - 1.0).powi(2) + 0.5 * x
}

fn double_well_gradient(x: &[f64], out: &mut [f64]) {
    let x = x[0];
    out[0] = 4.0 * x.powi(3) - 4.0 * x + 0.5;
}

#[test]
fn plain_gradient_descent_is_trapped_in_the_shallow_well() {
    let (best, value) = gradient_descent(double_well, double_well_gradient, &[0.5], 0.01, 2_000);
    assert!(
        best[0] > 0.0,
        "gradient descent should stay in the shallow well: {}",
        best[0]
    );
    assert!(
        (best[0] - 0.93).abs() < 0.1,
        "shallow well at ~0.93, got {}",
        best[0]
    );
    assert!(
        value > 0.0,
        "the shallow well is above the deep one: {value}"
    );
}

#[test]
fn annealed_langevin_escapes_the_shallow_well() {
    let mut escapes = 0usize;
    let trials = 200usize;
    for seed in 0..trials {
        let options = LangevinOptions {
            learning_rate: 0.01,
            steps: 2_000,
            initial_temperature: 1.0,
            final_temperature: 0.01,
            seed: seed as u64,
        };
        let outcome = annealed_langevin(double_well, double_well_gradient, &[0.5], &options);
        if outcome.best[0] < 0.0 {
            escapes += 1;
        }
    }
    // The thermal kick must beat the barrier for a substantial fraction of
    // seeds, while plain gradient descent never does.
    assert!(
        escapes > trials / 5,
        "annealed Langevin escaped only {escapes}/{trials}"
    );
}

#[test]
fn langevin_converges_on_a_convex_problem() {
    let objective = |x: &[f64]| (x[0] - 3.0).powi(2);
    let gradient = |x: &[f64], out: &mut [f64]| out[0] = 2.0 * (x[0] - 3.0);
    let options = LangevinOptions {
        learning_rate: 0.05,
        steps: 500,
        initial_temperature: 0.1,
        final_temperature: 1e-4,
        seed: 1,
    };
    let outcome = annealed_langevin(objective, gradient, &[0.0], &options);
    assert!(
        (outcome.best[0] - 3.0).abs() < 1e-2,
        "best {}",
        outcome.best[0]
    );
    assert!(outcome.best_value < 1e-3);
}

#[test]
fn finite_difference_gradient_matches_the_analytic_one() {
    let mut numeric = [0.0f64];
    let mut analytic = [0.0f64];
    finite_difference_gradient(double_well, &[0.7], 1e-6, &mut numeric);
    double_well_gradient(&[0.7], &mut analytic);
    assert!((numeric[0] - analytic[0]).abs() < 1e-5);
}

fn single_qubit_hamiltonian() -> quantumforge::analytic::PauliSum {
    use quantumforge::analytic::{ExactComplex, Pauli, PauliSum, PauliWord, Rational};
    let mut hamiltonian = PauliSum::zero(1);
    hamiltonian
        .add_term(
            ExactComplex::real(Rational::new(-1, 1).unwrap()),
            PauliWord::new(vec![Pauli::Z]),
        )
        .unwrap();
    hamiltonian
}

#[test]
fn langevin_vqe_reaches_the_ground_state() {
    use quantumforge::noisy_optimizer::{LangevinVqeOptions, annealed_langevin_vqe};
    use quantumforge::{ParameterizedCircuit, RotationAxis};

    let hamiltonian = single_qubit_hamiltonian();
    let mut circuit = ParameterizedCircuit::new(1);
    circuit.rotation(RotationAxis::Y, 0, 0);

    let options = LangevinVqeOptions {
        learning_rate: 0.05,
        steps: 400,
        initial_temperature: 0.1,
        final_temperature: 1e-4,
        seed: 3,
    };
    let outcome =
        annealed_langevin_vqe(&circuit, &hamiltonian, 1, &[0.7], &options).expect("Langevin VQE");
    assert!(
        outcome.best_energy < -0.99,
        "best energy {}",
        outcome.best_energy
    );
    // Best-so-far is monotone by construction.
    assert!(
        outcome
            .energies
            .windows(2)
            .all(|pair| pair[1] <= pair[0] + 1e-12)
    );
}

#[test]
fn gradient_descent_vqe_matches_the_exact_ground_energy() {
    use quantumforge::noisy_optimizer::gradient_descent_vqe;
    use quantumforge::{ParameterizedCircuit, RotationAxis};

    let hamiltonian = single_qubit_hamiltonian();
    let mut circuit = ParameterizedCircuit::new(1);
    circuit.rotation(RotationAxis::Y, 0, 0);
    let (_, energy) =
        gradient_descent_vqe(&circuit, &hamiltonian, 1, &[0.7], 0.05, 400).expect("GD VQE");
    assert!((energy - (-1.0)).abs() < 1e-6, "energy {energy}");
}

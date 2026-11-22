//! Tests for variational quantum thermalization.

use quantumforge::analytic::{ExactComplex, Pauli, PauliSum, PauliWord, Rational};
use quantumforge::vqt::{VqtOptions, free_energy, trace_distance, variational_thermalize};

fn real(value: i128) -> ExactComplex {
    ExactComplex::real(Rational::new(value, 1).unwrap())
}

fn interacting_hamiltonian() -> PauliSum {
    let mut hamiltonian = PauliSum::zero(2);
    let _ = hamiltonian.add_term(real(-1), PauliWord::new(vec![Pauli::Z, Pauli::I]));
    let _ = hamiltonian.add_term(real(-1), PauliWord::new(vec![Pauli::I, Pauli::Z]));
    let _ = hamiltonian.add_term(real(-1), PauliWord::new(vec![Pauli::Z, Pauli::Z]));
    let _ = hamiltonian.add_term(
        ExactComplex::real(Rational::new(-1, 2).unwrap()),
        PauliWord::new(vec![Pauli::X, Pauli::X]),
    );
    hamiltonian
}

#[test]
fn vqt_matches_the_exact_gibbs_state() {
    let hamiltonian = interacting_hamiltonian();
    let options = VqtOptions {
        system_qubits: 2,
        layers: 4,
        temperature: 1.0,
        learning_rate: 0.05,
        steps: 400,
        finite_difference_step: 1e-3,
    };
    let outcome = variational_thermalize(&hamiltonian, &options).expect("VQT must run");

    // The variational free energy is a bound on the exact one: it cannot go below.
    assert!(
        outcome.free_energy >= outcome.exact_free_energy - 1e-4,
        "prepared {} below exact {}",
        outcome.free_energy,
        outcome.exact_free_energy
    );
    assert!(
        (outcome.free_energy - outcome.exact_free_energy).abs() < 1e-3,
        "free-energy gap {}",
        outcome.free_energy - outcome.exact_free_energy
    );
    assert!(
        outcome.trace_distance < 1e-3,
        "trace distance {}",
        outcome.trace_distance
    );
}

#[test]
fn free_energy_decreases_from_the_maximally_mixed_start() {
    let hamiltonian = interacting_hamiltonian();
    let options = VqtOptions {
        system_qubits: 2,
        layers: 4,
        temperature: 1.0,
        learning_rate: 0.05,
        steps: 400,
        finite_difference_step: 1e-3,
    };
    let outcome = variational_thermalize(&hamiltonian, &options).expect("VQT must run");
    // The purified start is close to maximally mixed, so F starts well above the
    // exact value and descends. (The ansatz entangling ring acts even at
    // theta = 0, so the start is not exactly I/4.)
    assert!(
        outcome.free_energy < outcome.history[0],
        "final {} should beat the start {}",
        outcome.free_energy,
        outcome.history[0]
    );
    assert!(outcome.free_energy <= outcome.exact_free_energy + 1e-3);
    // Fixed-step gradient descent is not monotone; the run must still descend.
    let lowest = outcome
        .history
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    assert!(lowest < outcome.history[0] - 0.5);
}

#[test]
fn trace_distance_is_a_metric_on_a_known_pair() {
    use quantumforge::DensityMatrix;
    use quantumforge::thermal::gibbs_state;

    let hamiltonian = interacting_hamiltonian();
    let first = gibbs_state(&hamiltonian, 1.0).unwrap();
    let second = gibbs_state(&hamiltonian, 2.0).unwrap();
    assert!(trace_distance(&first, &first).unwrap().abs() < 1e-12);
    let distance = trace_distance(&first, &second).unwrap();
    assert!(distance > 0.0 && distance < 1.0);
    assert!((distance - trace_distance(&second, &first).unwrap()).abs() < 1e-12);
    let _ = DensityMatrix::from_state(&quantumforge::StateVector::zero(2).unwrap());
}

#[test]
fn free_energy_rejects_a_non_positive_temperature() {
    let hamiltonian = interacting_hamiltonian();
    let state = quantumforge::thermal::gibbs_state(&hamiltonian, 1.0).unwrap();
    assert!(free_energy(&state, &hamiltonian, 0.0).is_err());
    assert!(free_energy(&state, &hamiltonian, -1.0).is_err());
}

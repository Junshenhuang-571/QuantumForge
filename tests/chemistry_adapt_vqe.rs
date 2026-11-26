//! Tests for the ADAPT-VQE module.

use quantumforge::analytic::{Pauli, PauliWord};
use quantumforge::chemistry::adapt_vqe::{
    AdaptVqeOptions, pauli_rotation_circuit, qubit_excitation_pool, run_adapt_vqe,
};
use quantumforge::chemistry::mapping::{FermionMapping, molecular_hamiltonian};
use quantumforge::chemistry::{Atom, Element, sto3g_integrals};

fn h2() -> Vec<Atom> {
    vec![
        Atom {
            element: Element::H,
            position: [0.0, 0.0, 0.0],
        },
        Atom {
            element: Element::H,
            position: [0.0, 0.0, 1.4],
        },
    ]
}

#[test]
fn pauli_rotation_matches_the_analytic_xx_action() {
    let word = PauliWord::new(vec![Pauli::X, Pauli::X]);
    let circuit = pauli_rotation_circuit(&word, 2).expect("XX rotation");
    let theta = 0.7;
    let state = circuit.run(&[theta], 2).expect("runs");
    let amplitudes = state.amplitudes();
    let cos = (theta / 2.0).cos();
    let sin = (theta / 2.0).sin();
    assert!((amplitudes[0b00].re - cos).abs() < 1e-12, "|00> amplitude");
    assert!((amplitudes[0b11].im + sin).abs() < 1e-12, "|11> amplitude");
    assert!(amplitudes[0b01].norm_sqr() < 1e-24);
    assert!(amplitudes[0b10].norm_sqr() < 1e-24);
}

#[test]
fn pauli_rotation_with_y_matches_the_analytic_yy_action() {
    // exp(-i theta YY/2) |00> = cos(theta/2)|00> + i sin(theta/2)|11>.
    let word = PauliWord::new(vec![Pauli::Y, Pauli::Y]);
    let circuit = pauli_rotation_circuit(&word, 2).expect("YY rotation");
    let theta = 0.4;
    let amplitudes = circuit
        .run(&[theta], 2)
        .expect("runs")
        .amplitudes()
        .to_vec();
    assert!((amplitudes[0b00].re - (theta / 2.0).cos()).abs() < 1e-12);
    assert!((amplitudes[0b11].im - (theta / 2.0).sin()).abs() < 1e-12);
}

#[test]
fn helium_hartree_fock_state_is_already_the_exact_state() {
    let atoms = vec![Atom {
        element: Element::He,
        position: [0.0, 0.0, 0.0],
    }];
    let integrals = sto3g_integrals(&atoms).expect("He integrals");
    let hamiltonian =
        molecular_hamiltonian(&integrals, FermionMapping::JordanWigner).expect("Hamiltonian");
    let options = AdaptVqeOptions {
        gradient_threshold: 1e-4,
        max_operators: 2,
        learning_rate: 0.1,
        optimisation_steps: 50,
    };
    let result = run_adapt_vqe(&hamiltonian, 0b11, 2, &qubit_excitation_pool(2), &options)
        .expect("ADAPT-VQE");
    assert!(
        result.operators.is_empty(),
        "no operator should be selected"
    );
    let h_core = integrals.kinetic[0][0] + integrals.nuclear[0][0];
    let expected = 2.0 * h_core + integrals.repulsion[0][0][0][0];
    assert!(
        (result.energy - expected).abs() < 1e-8,
        "{} vs {expected}",
        result.energy
    );
}

#[test]
fn h2_adapt_vqe_lowers_the_energy_towards_the_exact_value() {
    let integrals = sto3g_integrals(&h2()).expect("H2 integrals");
    let hamiltonian =
        molecular_hamiltonian(&integrals, FermionMapping::JordanWigner).expect("Hamiltonian");
    let operator = hamiltonian.to_sparse_operator().expect("operator");
    let dense = operator.to_dense().expect("dense");
    let exact = dense
        .hermitian_eigensystem(1e-12, 1000)
        .expect("eigensystem")
        .eigenvalues()[0];

    let hf_amplitudes = hamiltonian
        .to_sparse_operator()
        .expect("operator")
        .to_dense()
        .expect("dense")
        .entries()[0b11 * 16 + 0b11]
        .re;
    let hf_energy = hf_amplitudes;

    let options = AdaptVqeOptions {
        gradient_threshold: 1e-3,
        max_operators: 3,
        learning_rate: 0.05,
        optimisation_steps: 300,
    };
    let result = run_adapt_vqe(&hamiltonian, 0b11, 4, &qubit_excitation_pool(4), &options)
        .expect("ADAPT-VQE");

    assert!(
        result.energy < hf_energy - 1e-4,
        "ADAPT energy {} did not improve on HF {}",
        result.energy,
        hf_energy
    );
    assert!(
        result.energy >= exact - 1e-6,
        "variational energy {} fell below exact {}",
        result.energy,
        exact
    );
    assert!(!result.operators.is_empty());
    assert_eq!(result.operators.len(), result.parameters.len());
}

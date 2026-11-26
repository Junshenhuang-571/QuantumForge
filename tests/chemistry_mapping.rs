//! Tests for the Jordan-Wigner molecular-Hamiltonian builder.
//!
//! The independent oracle here is a fermionic Hamiltonian assembled directly
//! in the occupation basis. It shares the integral values with the builder but
//! not the Jordan-Wigner algebra, so a mapping error changes one spectrum and
//! not the other.

use quantumforge::Complex64;
use quantumforge::chemistry::mapping::{FermionMapping, molecular_hamiltonian};
use quantumforge::chemistry::{Atom, Element, sto3g_integrals};
use quantumforge::quantum::{HilbertSpace, SparseOperator};

fn h2(count: usize) -> Vec<Atom> {
    let mut atoms = vec![Atom {
        element: Element::H,
        position: [0.0, 0.0, 0.0],
    }];
    if count > 1 {
        atoms.push(Atom {
            element: Element::H,
            position: [0.0, 0.0, 1.4],
        });
    }
    atoms
}

/// Apply a Jordan-Wigner ladder operation to an occupation bitstring.
fn ladder(state: usize, orbital: usize, create: bool, qubits: usize) -> Option<(usize, f64)> {
    let mask = 1usize << orbital;
    let occupied = state & mask != 0;
    if occupied == create {
        return None;
    }
    let lower_mask = mask - 1;
    let crossings = (state & lower_mask).count_ones();
    let sign = if (crossings & 1) == 0 { 1.0 } else { -1.0 };
    let next = if create { state | mask } else { state & !mask };
    debug_assert!(next < (1 << qubits));
    Some((next, sign))
}

fn fermionic_spectrum(integrals: &quantumforge::chemistry::ElectronIntegrals) -> Vec<f64> {
    let n_spatial = integrals.overlap.len();
    let spin_count = 2 * n_spatial;
    let dim = 1 << spin_count;
    let mut entries = Vec::new();

    for state in 0..dim {
        let mut column = vec![0.0; dim];
        for p in 0..spin_count {
            for q in 0..spin_count {
                if p % 2 != q % 2 {
                    continue;
                }
                let h = integrals.kinetic[p / 2][q / 2] + integrals.nuclear[p / 2][q / 2];
                if let Some((after_annihilate, sign1)) = ladder(state, q, false, spin_count)
                    && let Some((after_create, sign2)) =
                        ladder(after_annihilate, p, true, spin_count)
                {
                    column[after_create] += h * sign1 * sign2;
                }
            }
        }
        for p in 0..spin_count {
            for q in 0..spin_count {
                for r in 0..spin_count {
                    for s in 0..spin_count {
                        if p % 2 != r % 2 || q % 2 != s % 2 {
                            continue;
                        }
                        let value = 0.5 * integrals.repulsion[p / 2][q / 2][r / 2][s / 2];
                        if value == 0.0 {
                            continue;
                        }
                        let mut applied = Some((state, 1.0));
                        for (orbital, create) in [(r, false), (s, false), (q, true), (p, true)] {
                            applied = applied.and_then(|(current, sign)| {
                                ladder(current, orbital, create, spin_count)
                                    .map(|(next, step)| (next, sign * step))
                            });
                        }
                        if let Some((final_state, sign)) = applied {
                            column[final_state] += value * sign;
                        }
                    }
                }
            }
        }
        for (row, value) in column.into_iter().enumerate() {
            if value.abs() > 1e-14 {
                entries.push((row, state, Complex64::new(value, 0.0)));
            }
        }
    }

    let space = HilbertSpace::qubits(spin_count).expect("valid space");
    let operator = SparseOperator::new(space.clone(), space, entries).expect("valid operator");
    let dense = operator.to_dense().expect("dense operator");
    let system = dense
        .hermitian_eigensystem(1e-12, 1000)
        .expect("eigenpairs");
    system.eigenvalues().to_vec()
}

fn pauli_spectrum(integrals: &quantumforge::chemistry::ElectronIntegrals) -> Vec<f64> {
    let hamiltonian =
        molecular_hamiltonian(integrals, FermionMapping::JordanWigner).expect("qubit Hamiltonian");
    let operator = hamiltonian.to_sparse_operator().expect("sparse operator");
    let dense = operator.to_dense().expect("dense operator");
    let system = dense
        .hermitian_eigensystem(1e-12, 1000)
        .expect("eigenpairs");
    system.eigenvalues().to_vec()
}

#[test]
fn helium_hamiltonian_is_hermitian_and_matches_the_fermionic_spectrum() {
    let atoms = vec![Atom {
        element: Element::He,
        position: [0.0, 0.0, 0.0],
    }];
    let integrals = sto3g_integrals(&atoms).expect("He integrals");
    let hamiltonian =
        molecular_hamiltonian(&integrals, FermionMapping::JordanWigner).expect("qubit Hamiltonian");

    assert_eq!(hamiltonian.qubits(), 2);
    for (word, coefficient) in hamiltonian.terms() {
        assert!(
            coefficient.to_complex64().im.abs() < 1e-9,
            "Hermitian Hamiltonian has a complex coefficient on {word:?}"
        );
    }

    let fermionic = fermionic_spectrum(&integrals);
    let qubit = pauli_spectrum(&integrals);
    for (a, b) in fermionic.iter().zip(qubit.iter()) {
        assert!((a - b).abs() < 1e-8, "fermion {a} vs qubit {b}");
    }
}

#[test]
fn h2_hamiltonian_matches_the_fermionic_spectrum() {
    let integrals = sto3g_integrals(&h2(2)).expect("H2 integrals");
    let fermionic = fermionic_spectrum(&integrals);
    let qubit = pauli_spectrum(&integrals);
    for (a, b) in fermionic.iter().zip(qubit.iter()) {
        assert!((a - b).abs() < 1e-8, "fermion {a} vs qubit {b}");
    }
}

#[test]
fn pauli_words_are_built_on_the_expected_qubit_count() {
    let integrals = sto3g_integrals(&h2(2)).expect("H2 integrals");
    let hamiltonian =
        molecular_hamiltonian(&integrals, FermionMapping::JordanWigner).expect("qubit Hamiltonian");
    assert_eq!(hamiltonian.qubits(), 4);
    for word in hamiltonian.terms().keys() {
        assert_eq!(word.qubits(), 4);
    }
}

# Quantum chemistry in QuantumForge

This module is a **bounded first slice**, not a general chemistry package. It
can build and solve electronic-structure Hamiltonians for hydrogen and helium
in the minimal STO-3G basis. It cannot produce a first-principles iron atom:
there are no effective core potentials, relativistic corrections, basis-set
extrapolation or active-space selection in the crate.

The design contract is in [`docs/quantum-chemistry-spec.md`](../quantum-chemistry-spec.md).
Read that before extending it: it names the seams, invariants and non-goals.

## What is implemented

| Step | API | Behaviour |
|---|---|---|
| Integrals | `chemistry::sto3g_integrals` | Overlap, kinetic, nuclear attraction and electron-repulsion tensors for contracted s-type STO-3G shells on H and He; distances in bohr, energies in hartree. |
| Fermion → qubit | `chemistry::mapping::molecular_hamiltonian` | Jordan–Wigner mapping of the electronic Hamiltonian to a `PauliSum` on `2 * n_spatial` qubits. |
| Adaptive VQE | `chemistry::adapt_vqe::run_adapt_vqe` | Grows an ansatz from a two-qubit Pauli excitation pool using parameter-shift gradients and the existing VQE optimiser. |

The contraction is renormalised so each basis function has unit self-overlap.
The nuclear-attraction integral includes the nuclear charge; without that
factor hydrogen looks correct and helium is wrong by a factor of two.

## Example: helium

```rust
use quantumforge::chemistry::sto3g_integrals;
use quantumforge::chemistry::mapping::{molecular_hamiltonian, FermionMapping};
use quantumforge::chemistry::adapt_vqe::{
    qubit_excitation_pool, run_adapt_vqe, AdaptVqeOptions,
};
use quantumforge::chemistry::{Atom, Element};

let atoms = [Atom { element: Element::He, position: [0.0, 0.0, 0.0] }];
let integrals = sto3g_integrals(&atoms).expect("integrals");
let hamiltonian = molecular_hamiltonian(&integrals, FermionMapping::JordanWigner)
    .expect("Hamiltonian");

let result = run_adapt_vqe(
    &hamiltonian,
    0b11, // both spin orbitals occupied
    2,
    &qubit_excitation_pool(2),
    &AdaptVqeOptions {
        gradient_threshold: 1e-4,
        max_operators: 2,
        learning_rate: 0.1,
        optimisation_steps: 50,
    },
).expect("ADAPT-VQE");
assert!(result.operators.is_empty()); // |11> is already the exact state here
```

## Unit and accuracy conventions

- Distances are bohr; energies are hartree.
- The simulator uses `f64`; the `PauliSum` coefficients are exact rationals
  approximated to 1e-12. Small discrepancies at that scale are expected.
- Angles are radians; qubit `q` is bit `q` of the state index, so site 0 is the
  least significant bit.
- The mapping uses chemists' `(pq|rs)` integrals and the operator ordering
  `a†_p a†_q a_s a_r`. Spin deltas are `σ_p = σ_r` and `σ_q = σ_s`.

## How it is verified

- `tests/chemistry_integrals.rs` checks the H2 overlap against an independent
  published STO-3G value, and the helium Coulomb integral against radial
  quadrature of the spherical average `1/max(r1,r2)`.
- `tests/chemistry_mapping.rs` builds the fermionic Hamiltonian directly in
  the occupation basis and compares its spectrum to the qubit Pauli sum.
- `tests/chemistry_adapt_vqe.rs` verifies the Pauli-rotation decomposition
  against the analytic action of `exp(-i θ P/2)`, checks that helium's Hartree–Fock
  state is exact in this basis, and checks that H2 improves on Hartree–Fock
  without falling below the exact eigenvalue.

Dense `hermitian_eigensystem` is used in these tests: sparse Lanczos returned
different lowest eigenvalues for identical dense matrices at this size.

## Moving toward heavier elements

Published iron calculations do not start from a bare iron atom. They run DFT
and CASSCF classically, choose a small active orbital space (often with an
effective core potential), and map only that embedded space to qubits. The
quantum part solves the reduced problem; the classical part supplies the
integrals, frozen-core contribution and reference energy.

The crate currently has no such front end. A practical next step is to import
an externally generated active-space Hamiltonian as Pauli terms together with
its classical reference energy, then solve it with the existing VQE path. See
`docs/research/` for the primary-source survey behind that statement.

# Quantum chemistry layer — spec

Goal: make electronic-structure problems constructible inside QuantumForge,
without pretending to replace a full classical chemistry package.

This spec covers three modules, in dependency order. Each is an independently
testable seam.

## 1. `chemistry::integrals`

**Interface.** A user supplies atom symbols, Cartesian coordinates (bohr), and a
basis-set name. The module returns one- and two-electron integrals over
contracted Gaussian basis functions: overlap `S`, kinetic `T`, nuclear
attraction `V`, and electron-repulsion tensors `(pq|rs)`.

**Scope of the first slice:** STO-3G on hydrogen and helium only, s-type shells.
This is the minimal basis needed for a real, verifiable H2/He demo. General
basis sets, p/d shells and effective core potentials are later work.

**Invariants:** symmetry `S_pq = S_qp`, `(pq|rs) = (qp|rs) = (pq|sr) =
(rs|pq)`; positivity of `S`; Hermiticity of the core Hamiltonian.

**Tests against independent truth:** published STO-3G integral values for H2 at
a fixed bond length, plus analytic one-centre results and numerical quadrature
for overlap/kinetic/nuclear terms. No test may derive its expected value from
the same formula as the implementation.

## 2. `chemistry::mapping`

**Interface.** Given the integrals, the number of electrons and a mapping name
(`JordanWigner` first, parity and Bravyi–Kitaev later), produce the fermionic
Hamiltonian as a qubit `PauliSum` on `2 * n_spatial` qubits.

**Invariants:** the qubit Hamiltonian is Hermitian; it commutes with the
particle-number operator; its lowest eigenvalue equals the lowest eigenvalue of
the fermionic Hamiltonian obtained by diagonalising the one-/two-electron
tensor directly in the occupation basis.

**Out of scope:** frozen-core and embedding corrections; these are supplied by
the caller as an additive constant or an effective integral set.

## 3. `chemistry::adapt_vqe`

**Interface.** Given a `PauliSum` Hamiltonian, a reference occupation and an
operator pool, return an ansatz (`ParameterizedCircuit`), the selected
operators, and the converged energy. The first pool is qubit excitations
(`X/Y` strings); UCC-style fermionic pools are later work.

**Invariants:** adding an operator never increases the energy after
re-optimisation; the result is invariant to permutation of equivalent pool
entries; for a system whose exact ground state is reachable by the pool, the
final energy matches the exact eigenvalue from `PauliSum::to_sparse_operator`.

**Validation target:** H2 and He in STO-3G, compared against the classical
exact diagonalisation inside this crate and against published energies where
available. No advantage claim: the classical reference is the oracle.

## Explicit non-goals

- No full iron atom, no relativistic or ECP physics, no basis-set extrapolation.
- No claim that this is a chemistry package; integrals are supplied by this
  crate only for the small basis it implements, and external integral files
  remain a supported input path.
- QFT/QPE remain broken and are not required by this path; they must be fixed
  separately before any phase-estimation-based feature is added.

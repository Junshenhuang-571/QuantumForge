//! Minimal quantum-chemistry integrals.
//!
//! This module computes contracted Gaussian integrals for the STO-3G basis on
//! hydrogen and helium. It is deliberately small: the point is to make a real
//! electronic-structure Hamiltonian constructible inside the crate, not to
//! replace a general chemistry package. All distances are in bohr and all
//! energies in hartree.
//!
//! The four tensors returned by [`crate::chemistry::sto3g_integrals`] are the overlap, kinetic,
//! nuclear-attraction and electron-repulsion integrals over contracted basis
//! functions, plus the classical nucleus-nucleus repulsion.

use std::error::Error;
use std::fmt;

/// A supported element in the first slice of the STO-3G basis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Element {
    /// Hydrogen, one contracted 1s function.
    H,
    /// Helium, one contracted 1s function.
    He,
}

/// An atom placed at a Cartesian position in bohr.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Atom {
    /// Element of the atom.
    pub element: Element,
    /// Position `(x, y, z)` in bohr.
    pub position: [f64; 3],
}

/// Errors produced while building molecular integrals.
#[derive(Clone, Debug, PartialEq)]
pub enum ChemistryError {
    /// The basis does not contain a definition for this element.
    UnsupportedElement(Element),
    /// A coordinate was not finite.
    NonFiniteCoordinate(usize),
    /// No atoms were supplied.
    EmptyMolecule,
    /// An integral could not be represented as an exact rational coefficient.
    IntegralConversion,
    /// The ADAPT-VQE loop could not construct or optimise an ansatz.
    AdaptVqe(String),
}

impl fmt::Display for ChemistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedElement(element) => {
                write!(f, "STO-3G integrals are not available for {element:?}")
            }
            Self::NonFiniteCoordinate(index) => {
                write!(f, "atom {index} has a non-finite coordinate")
            }
            Self::EmptyMolecule => write!(f, "a molecule needs at least one atom"),
            Self::IntegralConversion => {
                write!(
                    f,
                    "an integral could not be represented as an exact rational"
                )
            }
            Self::AdaptVqe(message) => write!(f, "ADAPT-VQE failed: {message}"),
        }
    }
}

impl Error for ChemistryError {}

/// The one- and two-electron integrals for a contracted Gaussian basis.
#[derive(Clone, Debug, PartialEq)]
pub struct ElectronIntegrals {
    /// Overlap integrals `S[p][q]`.
    pub overlap: Vec<Vec<f64>>,
    /// Kinetic-energy integrals `T[p][q]`.
    pub kinetic: Vec<Vec<f64>>,
    /// Nuclear-attraction integrals `V[p][q]`, summed over all nuclei.
    pub nuclear: Vec<Vec<f64>>,
    /// Electron-repulsion integrals `(pq|rs)`.
    pub repulsion: Vec<Vec<Vec<Vec<f64>>>>,
    /// Classical repulsion between the nuclei, in hartree.
    pub nuclear_repulsion: f64,
}

/// Gaussian exponents and contraction coefficients for a contracted 1s shell.
#[derive(Clone, Copy, Debug)]
struct Shell {
    exponents: [f64; 3],
    coefficients: [f64; 3],
}

const H_SHELL: Shell = Shell {
    exponents: [3.42525091, 0.62391373, 0.16885540],
    coefficients: [0.15432897, 0.53532814, 0.44463454],
};

const HE_SHELL: Shell = Shell {
    exponents: [6.36242139, 1.15892300, 0.31364979],
    coefficients: [0.15432897, 0.53532814, 0.44463454],
};

fn shell(element: Element) -> Result<Shell, ChemistryError> {
    match element {
        Element::H => Ok(H_SHELL),
        Element::He => Ok(HE_SHELL),
    }
}

fn distance_squared(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| (a[i] - b[i]).powi(2)).sum()
}

/// Abramowitz and Stegun 7.1.26; maximum absolute error 1.5e-7.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.5 * x);
    let polynomial = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - polynomial * (-x * x).exp())
}

/// The Boys function `F0(x)`, with the limiting value 1 at the origin.
fn boys0(x: f64) -> f64 {
    if x < 1e-14 {
        1.0
    } else {
        (std::f64::consts::PI).sqrt() * erf(x.sqrt()) / (2.0 * x.sqrt())
    }
}

fn primitive_norm(alpha: f64) -> f64 {
    (2.0 * alpha / std::f64::consts::PI).powf(0.75)
}

fn primitive_overlap(alpha: f64, a: [f64; 3], beta: f64, b: [f64; 3]) -> f64 {
    let p = alpha + beta;
    let rab2 = distance_squared(a, b);
    primitive_norm(alpha)
        * primitive_norm(beta)
        * (std::f64::consts::PI / p).powf(1.5)
        * (-(alpha * beta / p) * rab2).exp()
}

fn primitive_kinetic(alpha: f64, a: [f64; 3], beta: f64, b: [f64; 3]) -> f64 {
    let p = alpha + beta;
    let mu = alpha * beta / p;
    let rab2 = distance_squared(a, b);
    mu * (3.0 - 2.0 * mu * rab2) * primitive_overlap(alpha, a, beta, b)
}

fn primitive_nuclear(
    alpha: f64,
    a: [f64; 3],
    beta: f64,
    b: [f64; 3],
    center: [f64; 3],
    charge: f64,
) -> f64 {
    let p = alpha + beta;
    let mu = alpha * beta / p;
    let rab2 = distance_squared(a, b);
    let mut p_point = [0.0; 3];
    for i in 0..3 {
        p_point[i] = (alpha * a[i] + beta * b[i]) / p;
    }
    let rpc2 = distance_squared(p_point, center);
    let prefactor = -charge * 2.0 * std::f64::consts::PI / p
        * primitive_norm(alpha)
        * primitive_norm(beta)
        * (-mu * rab2).exp();
    prefactor * boys0(p * rpc2)
}

#[allow(clippy::too_many_arguments)] // four primitive Gaussian centres
fn primitive_repulsion(
    alpha: f64,
    a: [f64; 3],
    beta: f64,
    b: [f64; 3],
    gamma: f64,
    c: [f64; 3],
    delta: f64,
    d: [f64; 3],
) -> f64 {
    let p = alpha + beta;
    let q = gamma + delta;
    let mu = alpha * beta / p;
    let nu = gamma * delta / q;
    let mut pp = [0.0; 3];
    let mut qq = [0.0; 3];
    for i in 0..3 {
        pp[i] = (alpha * a[i] + beta * b[i]) / p;
        qq[i] = (gamma * c[i] + delta * d[i]) / q;
    }
    let prefactor = 2.0 * std::f64::consts::PI.powf(2.5) / (p * q * (p + q).sqrt())
        * primitive_norm(alpha)
        * primitive_norm(beta)
        * primitive_norm(gamma)
        * primitive_norm(delta)
        * (-mu * distance_squared(a, b)).exp()
        * (-nu * distance_squared(c, d)).exp();
    prefactor * boys0(p * q / (p + q) * distance_squared(pp, qq))
}

/// Build the STO-3G integrals for the given atoms.
///
/// The contraction coefficients are renormalised so that each returned basis
/// function has unit self-overlap. This is the convention used by quantum
/// chemistry packages; using raw textbook coefficients instead leaves a small
/// `~1e-6` error on the diagonal.
pub fn sto3g_integrals(atoms: &[Atom]) -> Result<ElectronIntegrals, ChemistryError> {
    if atoms.is_empty() {
        return Err(ChemistryError::EmptyMolecule);
    }
    for (index, atom) in atoms.iter().enumerate() {
        if atom.position.iter().any(|value| !value.is_finite()) {
            return Err(ChemistryError::NonFiniteCoordinate(index));
        }
        let _ = shell(atom.element)?;
    }

    let shells: Vec<Shell> = atoms
        .iter()
        .map(|atom| shell(atom.element))
        .collect::<Result<_, _>>()?;
    let n = atoms.len();
    let mut coefficients = vec![[0.0f64; 3]; n];
    for (index, current) in shells.iter().enumerate() {
        let mut self_overlap = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                self_overlap += current.coefficients[i]
                    * current.coefficients[j]
                    * primitive_overlap(
                        current.exponents[i],
                        atoms[index].position,
                        current.exponents[j],
                        atoms[index].position,
                    );
            }
        }
        let normalisation = self_overlap.sqrt().recip();
        for i in 0..3 {
            coefficients[index][i] = current.coefficients[i] * normalisation;
        }
    }

    let mut overlap = vec![vec![0.0; n]; n];
    let mut kinetic = vec![vec![0.0; n]; n];
    let mut nuclear = vec![vec![0.0; n]; n];
    let mut repulsion = vec![vec![vec![vec![0.0; n]; n]; n]; n];

    for p in 0..n {
        for q in 0..n {
            for i in 0..3 {
                for j in 0..3 {
                    let a = shells[p].exponents[i];
                    let b = shells[q].exponents[j];
                    let cp = coefficients[p][i];
                    let cq = coefficients[q][j];
                    let factor = cp * cq;
                    overlap[p][q] +=
                        factor * primitive_overlap(a, atoms[p].position, b, atoms[q].position);
                    kinetic[p][q] +=
                        factor * primitive_kinetic(a, atoms[p].position, b, atoms[q].position);
                    for nucleus in atoms {
                        nuclear[p][q] += factor
                            * primitive_nuclear(
                                a,
                                atoms[p].position,
                                b,
                                atoms[q].position,
                                nucleus.position,
                                match nucleus.element {
                                    Element::H => 1.0,
                                    Element::He => 2.0,
                                },
                            );
                    }
                }
            }
        }
    }

    for p in 0..n {
        for q in 0..n {
            for r in 0..n {
                for s in 0..n {
                    for i in 0..3 {
                        for j in 0..3 {
                            for k in 0..3 {
                                for l in 0..3 {
                                    repulsion[p][q][r][s] += coefficients[p][i]
                                        * coefficients[q][j]
                                        * coefficients[r][k]
                                        * coefficients[s][l]
                                        * primitive_repulsion(
                                            shells[p].exponents[i],
                                            atoms[p].position,
                                            shells[q].exponents[j],
                                            atoms[q].position,
                                            shells[r].exponents[k],
                                            atoms[r].position,
                                            shells[s].exponents[l],
                                            atoms[s].position,
                                        );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let mut nuclear_repulsion = 0.0;
    for p in 0..n {
        for q in (p + 1)..n {
            let charge = |element: Element| match element {
                Element::H => 1.0,
                Element::He => 2.0,
            };
            nuclear_repulsion += charge(atoms[p].element) * charge(atoms[q].element)
                / distance_squared(atoms[p].position, atoms[q].position).sqrt();
        }
    }

    Ok(ElectronIntegrals {
        overlap,
        kinetic,
        nuclear,
        repulsion,
        nuclear_repulsion,
    })
}

/// Fermion-to-qubit mapping and molecular Hamiltonian construction.
pub mod mapping {
    use super::{ChemistryError, ElectronIntegrals};
    use crate::analytic::{ExactComplex, Pauli, PauliSum, PauliWord, Rational};

    /// The fermion-to-qubit mapping used to build a Hamiltonian.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum FermionMapping {
        /// Jordan-Wigner: a Z string on all lower-indexed spin orbitals.
        JordanWigner,
    }

    /// Spin orbital `p` of spatial orbital `p / 2`, spin `p % 2`.
    fn spatial(spin_orbital: usize) -> usize {
        spin_orbital / 2
    }

    fn spin(spin_orbital: usize) -> usize {
        spin_orbital % 2
    }

    fn rational(value: f64) -> Result<Rational, ChemistryError> {
        const SCALE: i128 = 1_000_000_000_000;
        let scaled = (value * SCALE as f64).round() as i128;
        Rational::new(scaled, SCALE).map_err(|_| ChemistryError::IntegralConversion)
    }

    fn coefficient(value: f64) -> Result<ExactComplex, ChemistryError> {
        Ok(ExactComplex::real(rational(value)?))
    }

    type Terms = Vec<(ExactComplex, PauliWord)>;

    /// Jordan-Wigner ladder operator on `n` qubits.
    fn ladder(orbital: usize, n: usize, creation: bool) -> Result<Terms, ChemistryError> {
        let half = ExactComplex::real(
            Rational::new(1, 2).map_err(|_| ChemistryError::IntegralConversion)?,
        );
        let mut factors = vec![Pauli::I; n];
        for factor in factors.iter_mut().take(orbital) {
            *factor = Pauli::Z;
        }
        factors[orbital] = Pauli::X;
        let mut terms = vec![(half, PauliWord::new(factors.clone()))];
        factors[orbital] = Pauli::Y;
        let imaginary = if creation {
            ExactComplex::I
                .checked_neg()
                .map_err(|_| ChemistryError::IntegralConversion)?
        } else {
            ExactComplex::I
        };
        let coefficient = imaginary
            .checked_mul(half)
            .map_err(|_| ChemistryError::IntegralConversion)?;
        terms.push((coefficient, PauliWord::new(factors)));
        Ok(terms)
    }

    fn multiply_terms(left: &Terms, right: &Terms) -> Result<Terms, ChemistryError> {
        let mut result = Vec::new();
        for (a, word_a) in left {
            for (b, word_b) in right {
                let (phase, word) = word_a
                    .multiply(word_b)
                    .map_err(|_| ChemistryError::IntegralConversion)?;
                let coefficient = a
                    .checked_mul(*b)
                    .and_then(|value| value.checked_mul(phase))
                    .map_err(|_| ChemistryError::IntegralConversion)?;
                result.push((coefficient, word));
            }
        }
        Ok(result)
    }

    /// Builds the electronic Hamiltonian as a qubit Pauli sum using the given
    /// fermion-to-qubit mapping.
    ///
    /// The input integrals are chemists' `(pq|rs)` over spatial orbitals; the
    /// Hamiltonian is
    /// `sum_pq h_pq a^dag_p a_q + 1/2 sum_pqrs (pq|rs) a^dag_p a^dag_q a_s a_r`
    /// over spin orbitals. Frozen-core and embedding corrections must be added
    /// by the caller.
    pub fn molecular_hamiltonian(
        integrals: &ElectronIntegrals,
        mapping: FermionMapping,
    ) -> Result<PauliSum, ChemistryError> {
        match mapping {
            FermionMapping::JordanWigner => {}
        }

        let spatial_count = integrals.overlap.len();
        let spin_count = 2 * spatial_count;
        let mut sum = PauliSum::zero(spin_count);

        let add = |value: f64, terms: &Terms, sum: &mut PauliSum| -> Result<(), ChemistryError> {
            if value.abs() < 1e-14 {
                return Ok(());
            }
            let scalar = coefficient(value)?;
            for (term, word) in terms {
                let scaled = scalar
                    .checked_mul(*term)
                    .map_err(|_| ChemistryError::IntegralConversion)?;
                sum.add_term(scaled, word.clone())
                    .map_err(|_| ChemistryError::IntegralConversion)?;
            }
            Ok(())
        };

        // One-electron part: h_pq = T_pq + V_pq on the spatial orbitals, with
        // matching spin on both sides.
        for p in 0..spin_count {
            for q in 0..spin_count {
                if spin(p) != spin(q) {
                    continue;
                }
                let sp = spatial(p);
                let sq = spatial(q);
                let h = integrals.kinetic[sp][sq] + integrals.nuclear[sp][sq];
                let create = ladder(p, spin_count, true)?;
                let annihilate = ladder(q, spin_count, false)?;
                let terms = multiply_terms(&create, &annihilate)?;
                add(h, &terms, &mut sum)?;
            }
        }

        // Two-electron part. The integral is nonzero only when the two spins
        // on each Coulomb line agree: (p sigma, q tau | r sigma, s tau).
        for p in 0..spin_count {
            for q in 0..spin_count {
                for r in 0..spin_count {
                    for s in 0..spin_count {
                        if spin(p) != spin(r) || spin(q) != spin(s) {
                            continue;
                        }
                        let value =
                            integrals.repulsion[spatial(p)][spatial(q)][spatial(r)][spatial(s)];
                        if value.abs() < 1e-14 {
                            continue;
                        }
                        let mut terms = ladder(p, spin_count, true)?;
                        let second = multiply_terms(&terms, &ladder(q, spin_count, true)?)?;
                        terms = multiply_terms(&second, &ladder(s, spin_count, false)?)?;
                        let final_terms = multiply_terms(&terms, &ladder(r, spin_count, false)?)?;
                        let scalar = coefficient(0.5 * value)?;
                        let mut final_terms = final_terms;
                        for (term, _word) in &mut final_terms {
                            *term = term
                                .checked_mul(scalar)
                                .map_err(|_| ChemistryError::IntegralConversion)?;
                        }
                        add(1.0, &final_terms, &mut sum)?;
                    }
                }
            }
        }
        Ok(sum)
    }
}

/// Adaptive VQE with a qubit-excitation pool.
pub mod adapt_vqe {
    use super::ChemistryError;
    use crate::analytic::{Pauli, PauliSum, PauliWord};
    use crate::noisy_optimizer::gradient_descent_vqe;
    use crate::{Gate, ParameterizedCircuit, RotationAxis};

    /// Settings for an ADAPT-VQE run.
    #[derive(Clone, Debug, PartialEq)]
    pub struct AdaptVqeOptions {
        /// Stop when the largest pool gradient is below this value.
        pub gradient_threshold: f64,
        /// Maximum number of operators appended to the ansatz.
        pub max_operators: usize,
        /// Learning rate used for each re-optimisation.
        pub learning_rate: f64,
        /// Gradient-descent steps used for each re-optimisation.
        pub optimisation_steps: usize,
    }

    /// The result of an ADAPT-VQE run.
    #[derive(Clone, Debug)]
    pub struct AdaptVqeResult {
        /// The final parameterised ansatz.
        pub circuit: ParameterizedCircuit,
        /// Operators appended to the ansatz, in order.
        pub operators: Vec<PauliWord>,
        /// Optimised angles, one per operator.
        pub parameters: Vec<f64>,
        /// The best energy found.
        pub energy: f64,
        /// Pool gradients at the final point.
        pub gradients: Vec<f64>,
    }

    /// Two-qubit Pauli excitation pool for `qubits` qubits.
    pub fn qubit_excitation_pool(qubits: usize) -> Vec<PauliWord> {
        let mut pool = Vec::new();
        for first in 0..qubits {
            for second in (first + 1)..qubits {
                for left in [Pauli::X, Pauli::Y] {
                    for right in [Pauli::X, Pauli::Y] {
                        let mut factors = vec![Pauli::I; qubits];
                        factors[first] = left;
                        factors[second] = right;
                        pool.push(PauliWord::new(factors));
                    }
                }
            }
        }
        pool
    }

    /// Builds the circuit for `exp(-i theta P/2)` on `qubits` qubits, with the
    /// angle bound to parameter 0. Exposed so the decomposition can be verified
    /// against the matrix exponential.
    pub fn pauli_rotation_circuit(
        word: &PauliWord,
        qubits: usize,
    ) -> Result<ParameterizedCircuit, ChemistryError> {
        if word.qubits() != qubits {
            return Err(ChemistryError::AdaptVqe(format!(
                "Pauli word acts on {} qubits, expected {qubits}",
                word.qubits()
            )));
        }
        let mut circuit = ParameterizedCircuit::new(1);
        append_pauli_rotation(&mut circuit, word, 0)?;
        Ok(circuit)
    }

    /// Appends `exp(-i theta P/2)` for a Pauli word, using one parameter.
    fn append_pauli_rotation(
        circuit: &mut ParameterizedCircuit,
        word: &PauliWord,
        parameter: usize,
    ) -> Result<(), ChemistryError> {
        let qubits = word.qubits();
        let factors = word.factors();
        let target = factors
            .iter()
            .rposition(|pauli| *pauli != Pauli::I)
            .ok_or_else(|| ChemistryError::AdaptVqe("identity pool operator".to_owned()))?;
        if factors.iter().filter(|pauli| **pauli != Pauli::I).count() < 2 {
            return Err(ChemistryError::AdaptVqe(
                "pool operators must act on at least two qubits".to_owned(),
            ));
        }

        let mut basis_changes = Vec::new();
        for (qubit, pauli) in factors.iter().enumerate() {
            match pauli {
                Pauli::I => {}
                Pauli::Z => {}
                Pauli::X => {
                    circuit.push(Gate::H, qubit);
                    basis_changes.push((qubit, Gate::H));
                }
                Pauli::Y => {
                    // (H S^dag) Y (S H) = Z: apply S^dag then H, undo with H then S.
                    circuit.push(Gate::Sdg, qubit);
                    circuit.push(Gate::H, qubit);
                    // Undo in reverse order: H then S.
                    basis_changes.push((qubit, Gate::S));
                    basis_changes.push((qubit, Gate::H));
                }
            }
        }

        let supports: Vec<usize> = factors
            .iter()
            .enumerate()
            .filter(|(_, pauli)| **pauli != Pauli::I && **pauli != Pauli::Z)
            .map(|(qubit, _)| qubit)
            .collect();
        let mut parity = Vec::new();
        for qubit in supports.iter().copied().filter(|qubit| *qubit != target) {
            circuit.cnot(qubit, target);
            parity.push(qubit);
        }
        circuit.rotation(RotationAxis::Z, target, parameter);
        for qubit in parity.into_iter().rev() {
            circuit.cnot(qubit, target);
        }
        for (qubit, gate) in basis_changes.into_iter().rev() {
            circuit.push(gate, qubit);
        }
        let _ = qubits;
        Ok(())
    }

    fn energy(
        circuit: &ParameterizedCircuit,
        parameters: &[f64],
        hamiltonian: &PauliSum,
        qubits: usize,
    ) -> Result<f64, ChemistryError> {
        let state = circuit
            .run(parameters, qubits)
            .map_err(|error| ChemistryError::AdaptVqe(error.to_string()))?;
        let ket = state
            .to_ket()
            .map_err(|error| ChemistryError::AdaptVqe(error.to_string()))?;
        hamiltonian
            .expectation_ket(&ket)
            .map(|value| value.re)
            .map_err(|error| ChemistryError::AdaptVqe(error.to_string()))
    }

    fn with_candidate(
        operators: &[PauliWord],
        parameters: &[f64],
        reference: usize,
        qubits: usize,
        candidate: &PauliWord,
        angle: f64,
    ) -> Result<ParameterizedCircuit, ChemistryError> {
        let mut circuit = ParameterizedCircuit::new(operators.len() + 1);
        for qubit in 0..qubits {
            if (reference >> qubit) & 1 == 1 {
                circuit.push(Gate::X, qubit);
            }
        }
        for (index, word) in operators.iter().enumerate() {
            append_pauli_rotation(&mut circuit, word, index)?;
        }
        let all = [parameters.to_vec(), vec![angle]].concat();
        append_pauli_rotation(&mut circuit, candidate, operators.len())?;
        let _ = all;
        Ok(circuit)
    }

    /// Runs ADAPT-VQE from a Hartree-Fock occupation `reference`.
    ///
    /// The qubit Hamiltonian must already include any frozen-core or embedding
    /// energy shift the caller wants to apply.
    pub fn run_adapt_vqe(
        hamiltonian: &PauliSum,
        reference: usize,
        qubits: usize,
        pool: &[PauliWord],
        options: &AdaptVqeOptions,
    ) -> Result<AdaptVqeResult, ChemistryError> {
        if hamiltonian.qubits() != qubits {
            return Err(ChemistryError::AdaptVqe(format!(
                "Hamiltonian acts on {} qubits, expected {qubits}",
                hamiltonian.qubits()
            )));
        }
        let mut operators: Vec<PauliWord> = Vec::new();
        let mut parameters: Vec<f64> = Vec::new();
        let mut gradients = vec![0.0; pool.len()];

        for _ in 0..options.max_operators {
            for (index, candidate) in pool.iter().enumerate() {
                let mut plus = with_candidate(
                    &operators,
                    &parameters,
                    reference,
                    qubits,
                    candidate,
                    std::f64::consts::FRAC_PI_2,
                )?;
                let mut minus = with_candidate(
                    &operators,
                    &parameters,
                    reference,
                    qubits,
                    candidate,
                    -std::f64::consts::FRAC_PI_2,
                )?;
                let _ = (&mut plus, &mut minus);
                let plus_parameters =
                    [parameters.as_slice(), &[std::f64::consts::FRAC_PI_2]].concat();
                let minus_parameters =
                    [parameters.as_slice(), &[-std::f64::consts::FRAC_PI_2]].concat();
                gradients[index] = 0.5
                    * (energy(&plus, &plus_parameters, hamiltonian, qubits)?
                        - energy(&minus, &minus_parameters, hamiltonian, qubits)?);
            }
            let (index, magnitude) = gradients
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
                .ok_or_else(|| ChemistryError::AdaptVqe("empty operator pool".to_owned()))?;
            if magnitude.abs() < options.gradient_threshold {
                break;
            }
            operators.push(pool[index].clone());
            parameters.push(0.0);

            let mut circuit = ParameterizedCircuit::new(operators.len());
            for qubit in 0..qubits {
                if (reference >> qubit) & 1 == 1 {
                    circuit.push(Gate::X, qubit);
                }
            }
            for (operator_index, word) in operators.iter().enumerate() {
                append_pauli_rotation(&mut circuit, word, operator_index)?;
            }
            let (optimised, energy_value) = gradient_descent_vqe(
                &circuit,
                hamiltonian,
                qubits,
                &parameters,
                options.learning_rate,
                options.optimisation_steps,
            )
            .map_err(|error| ChemistryError::AdaptVqe(error.to_string()))?;
            parameters = optimised;
            let _ = energy_value;
        }

        let mut circuit = ParameterizedCircuit::new(operators.len());
        for qubit in 0..qubits {
            if (reference >> qubit) & 1 == 1 {
                circuit.push(Gate::X, qubit);
            }
        }
        for (index, word) in operators.iter().enumerate() {
            append_pauli_rotation(&mut circuit, word, index)?;
        }
        let final_energy = energy(&circuit, &parameters, hamiltonian, qubits)?;
        Ok(AdaptVqeResult {
            circuit,
            operators,
            parameters,
            energy: final_energy,
            gradients,
        })
    }
}

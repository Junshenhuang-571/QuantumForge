//! Finite-temperature states: exact Gibbs states, free energy, and sampling.
//!
//! The Gibbs state `rho = exp(-beta H) / Z` is the equilibrium state of a
//! system coupled to a bath at temperature `T = 1 / beta`. It is the target of
//! the thermalising noise channels in the crate root, and it is the state a
//! finite-temperature observable is evaluated in.
//!
//! [`gibbs_state`] builds it exactly by diagonalising a `PauliSum`, so it is
//! the reference for small systems. [`sample_gibbs`] draws computational-basis
//! bit strings from it, which is the useful primitive for Boltzmann-style
//! sampling and for checking a thermalising channel against the exact answer.

use crate::analytic::PauliSum;
use crate::analytic::Rational;
use crate::quantum::LinearError;
use crate::{Complex64, DensityMatrix, QuantumError, SmallRng};

/// Exact rational Gibbs weights for a rational Boltzmann base `q = exp(-beta)`:
/// `Z = sum_i q^{E_i}` and `p_i = q^{E_i} / Z`, all exact.
///
/// This is the exact-arithmetic corner of the thermal layer. It is not a
/// replacement for [`gibbs_state`]: for irrational `beta` the base `q` is
/// transcendental, so a general finite-temperature state cannot be represented
/// exactly. What IS exact is the classical Boltzmann weight once `q` is
/// rational, together with the identities `sum_i p_i = 1` and
/// `<E> = sum_i p_i E_i`.
pub fn exact_gibbs_weights(
    energies: &[i128],
    base: &Rational,
) -> Result<(Rational, Vec<Rational>), QuantumError> {
    if energies.is_empty() {
        return Err(QuantumError::InvalidTrainingData);
    }
    let one = || {
        Rational::new(1, 1).map_err(|error| QuantumError::UnsupportedOperation(error.to_string()))
    };
    let mut weights = Vec::with_capacity(energies.len());
    for &energy in energies {
        let exponent = energy.unsigned_abs();
        let mut weight = one()?;
        for _ in 0..exponent {
            weight = weight
                .checked_mul(*base)
                .map_err(|error| QuantumError::UnsupportedOperation(error.to_string()))?;
        }
        if energy < 0 {
            weight = one()?
                .checked_div(weight)
                .map_err(|error| QuantumError::UnsupportedOperation(error.to_string()))?;
        }
        weights.push(weight);
    }
    let zero = Rational::new(0, 1)
        .map_err(|error| QuantumError::UnsupportedOperation(error.to_string()))?;
    let partition = weights.iter().try_fold(zero, |acc, weight| {
        acc.checked_add(*weight)
            .map_err(|error| QuantumError::UnsupportedOperation(error.to_string()))
    })?;
    if partition.is_zero() {
        return Err(QuantumError::InvalidProbability(0.0));
    }
    let probabilities = weights
        .iter()
        .map(|weight| {
            weight
                .checked_div(partition)
                .map_err(|error| QuantumError::UnsupportedOperation(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((partition, probabilities))
}

/// Exact rational expectation `<E> = sum_i p_i E_i` of an exact Gibbs weight set.
pub fn exact_energy_expectation(
    energies: &[i128],
    probabilities: &[Rational],
) -> Result<Rational, QuantumError> {
    if energies.len() != probabilities.len() {
        return Err(QuantumError::InvalidTrainingData);
    }
    let zero = Rational::new(0, 1)
        .map_err(|error| QuantumError::UnsupportedOperation(error.to_string()))?;
    energies
        .iter()
        .zip(probabilities.iter())
        .try_fold(zero, |acc, (energy, probability)| {
            let term = Rational::new(*energy, 1)
                .map_err(|error| QuantumError::UnsupportedOperation(error.to_string()))?
                .checked_mul(*probability)
                .map_err(|error| QuantumError::UnsupportedOperation(error.to_string()))?;
            acc.checked_add(term)
                .map_err(|error| QuantumError::UnsupportedOperation(error.to_string()))
        })
}
use std::collections::BTreeMap;

fn linear_error(error: LinearError) -> QuantumError {
    QuantumError::UnsupportedOperation(error.to_string())
}

/// Validate `beta` and return the eigenvalues and (column) eigenvectors of
/// `hamiltonian`, together with the Boltzmann weight of each level.
fn spectrum_and_weights(
    hamiltonian: &PauliSum,
    beta: f64,
) -> Result<(Vec<f64>, crate::quantum::Operator, Vec<f64>, f64), QuantumError> {
    if !beta.is_finite() || beta < 0.0 {
        return Err(QuantumError::InvalidProbability(beta));
    }
    let sparse = hamiltonian.to_sparse_operator().map_err(linear_error)?;
    let dense = sparse.to_dense().map_err(linear_error)?;
    let spectrum = dense
        .hermitian_eigensystem(1e-10, 2_000)
        .map_err(linear_error)?;
    let eigenvalues = spectrum.eigenvalues().to_vec();
    let mut weights = Vec::with_capacity(eigenvalues.len());
    let mut partition = 0.0f64;
    for &energy in &eigenvalues {
        let weight = if beta == 0.0 {
            1.0
        } else {
            (-beta * energy).exp()
        };
        weights.push(weight);
        partition += weight;
    }
    if !partition.is_finite() || partition <= 0.0 {
        return Err(QuantumError::InvalidProbability(partition));
    }
    Ok((
        eigenvalues,
        spectrum.eigenvectors().clone(),
        weights,
        partition,
    ))
}

/// Exact Gibbs state `exp(-beta H) / Z` as a density matrix.
///
/// `beta = 0` is the infinite-temperature limit, the maximally mixed state.
pub fn gibbs_state(hamiltonian: &PauliSum, beta: f64) -> Result<DensityMatrix, QuantumError> {
    let (_eigenvalues, eigenvectors, weights, partition) = spectrum_and_weights(hamiltonian, beta)?;
    let dimension = weights.len();
    let mut entries = vec![Complex64::ZERO; dimension * dimension];
    for (level, &weight) in weights.iter().enumerate() {
        let probability = weight / partition;
        for row in 0..dimension {
            let left = eigenvectors.entries()[row * dimension + level];
            for col in 0..dimension {
                let right = eigenvectors.entries()[col * dimension + level];
                entries[row * dimension + col] =
                    entries[row * dimension + col] + left * right.conj() * probability;
            }
        }
    }
    Ok(DensityMatrix::from_entries(entries, hamiltonian.qubits()))
}

/// Diagonal of the Gibbs state in the computational basis.
pub fn gibbs_probabilities(hamiltonian: &PauliSum, beta: f64) -> Result<Vec<f64>, QuantumError> {
    let (_eigenvalues, eigenvectors, weights, partition) = spectrum_and_weights(hamiltonian, beta)?;
    let dimension = weights.len();
    let mut probabilities = vec![0.0f64; dimension];
    for (level, &weight) in weights.iter().enumerate() {
        let probability = weight / partition;
        for (row, probability_entry) in probabilities.iter_mut().enumerate() {
            let amplitude = eigenvectors.entries()[row * dimension + level];
            *probability_entry += probability * amplitude.norm_sqr();
        }
    }
    Ok(probabilities)
}

/// Sample computational-basis bit strings from the Gibbs state.
///
/// The returned map is keyed by least-significant-bit-first outcome strings,
/// matching `ShotResult::counts`.
pub fn sample_gibbs(
    hamiltonian: &PauliSum,
    beta: f64,
    shots: usize,
    seed: u64,
) -> Result<BTreeMap<Vec<u8>, usize>, QuantumError> {
    let probabilities = gibbs_probabilities(hamiltonian, beta)?;
    let qubits = hamiltonian.qubits();
    let mut rng = SmallRng::seeded(seed);
    let mut counts: BTreeMap<Vec<u8>, usize> = BTreeMap::new();
    for _ in 0..shots {
        let sample = rng.next_f64();
        let mut cumulative = 0.0f64;
        let mut index = probabilities.len() - 1;
        for (row, &probability) in probabilities.iter().enumerate() {
            cumulative += probability;
            if sample < cumulative {
                index = row;
                break;
            }
        }
        let bits: Vec<u8> = (0..qubits)
            .map(|qubit| u8::from(index & (1usize << qubit) != 0))
            .collect();
        *counts.entry(bits).or_insert(0) += 1;
    }
    Ok(counts)
}

/// Mean energy `<H>_beta = sum_k E_k exp(-beta E_k) / Z`.
pub fn mean_energy(hamiltonian: &PauliSum, beta: f64) -> Result<f64, QuantumError> {
    let (eigenvalues, _vectors, weights, partition) = spectrum_and_weights(hamiltonian, beta)?;
    Ok(eigenvalues
        .iter()
        .zip(weights.iter())
        .map(|(&energy, &weight)| energy * weight)
        .sum::<f64>()
        / partition)
}

/// Helmholtz free energy `F = <E> - T S = -(1/beta) ln Z`.
///
/// Requires `beta > 0`; at `beta = 0` the entropy term dominates and `F` is not
/// finite.
pub fn free_energy(hamiltonian: &PauliSum, beta: f64) -> Result<f64, QuantumError> {
    if !beta.is_finite() || beta <= 0.0 {
        return Err(QuantumError::InvalidProbability(beta));
    }
    let sparse = hamiltonian.to_sparse_operator().map_err(linear_error)?;
    let dense = sparse.to_dense().map_err(linear_error)?;
    let spectrum = dense
        .hermitian_eigensystem(1e-10, 2_000)
        .map_err(linear_error)?;
    let mut partition = 0.0f64;
    for &energy in spectrum.eigenvalues() {
        partition += (-beta * energy).exp();
    }
    if !partition.is_finite() || partition <= 0.0 {
        return Err(QuantumError::InvalidProbability(partition));
    }
    Ok(-partition.ln() / beta)
}

/// Expectation `Tr(rho O)` of a Pauli-sum observable in a density matrix.
pub fn expectation(rho: &DensityMatrix, observable: &PauliSum) -> Result<f64, QuantumError> {
    let sparse = observable.to_sparse_operator().map_err(linear_error)?;
    let dense = sparse.to_dense().map_err(linear_error)?;
    let dimension = rho.dimension();
    let mut trace = Complex64::ZERO;
    for row in 0..dimension {
        for col in 0..dimension {
            trace = trace
                + rho.entries()[row * dimension + col] * dense.entries()[col * dimension + row];
        }
    }
    Ok(trace.re)
}

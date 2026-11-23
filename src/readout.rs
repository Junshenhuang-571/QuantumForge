//! Readout (measurement) error: confusion matrices and mitigation.
//!
//! A single-qubit readout error is a classical channel on the measurement
//! outcome. [`ConfusionMatrix`] stores it as `P(observed | prepared)`, so the
//! columns sum to one, and mitigation applies its inverse to an observed
//! distribution. [`mitigate_distribution`] applies one inverse per qubit to a
//! full computational-basis distribution, which is the inverse of the product
//! channel when the readout errors are independent across qubits.
//!
//! The inverse is a linear map and can produce small negative probabilities;
//! [`ConfusionMatrix::unfold`] is the positivity-preserving iterative Bayesian
//! alternative.

use crate::SmallRng;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

/// Errors from readout mitigation.
#[derive(Clone, Debug, PartialEq)]
pub enum ReadoutError {
    /// An entry outside `[0, 1]`; carries the value.
    InvalidEntry(f64),
    /// A column does not sum to one; carries the column and its sum.
    ColumnSum {
        /// The offending column.
        column: usize,
        /// Its sum.
        sum: f64,
    },
    /// The matrix is singular, so it has no inverse.
    Singular,
    /// A distribution length does not match `2^qubits`.
    DimensionMismatch {
        /// The expected length.
        expected: usize,
        /// The supplied length.
        actual: usize,
    },
}

impl fmt::Display for ReadoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidEntry(value) => {
                write!(f, "confusion-matrix entry must be in [0, 1], got {value}")
            }
            Self::ColumnSum { column, sum } => {
                write!(f, "confusion-matrix column {column} sums to {sum}, not 1")
            }
            Self::Singular => write!(f, "confusion matrix is singular and cannot be inverted"),
            Self::DimensionMismatch { expected, actual } => write!(
                f,
                "distribution length {actual} does not match 2^qubits = {expected}"
            ),
        }
    }
}

impl Error for ReadoutError {}

/// A single-qubit readout confusion matrix.
///
/// `matrix[row][col] = P(observed = row | prepared = col)`, so each column sums
/// to one. The ideal readout is the identity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConfusionMatrix {
    /// Row-major `[[P(0|0), P(0|1)], [P(1|0), P(1|1)]]`.
    matrix: [[f64; 2]; 2],
}

impl ConfusionMatrix {
    /// Build a confusion matrix from `[[P(0|0), P(0|1)], [P(1|0), P(1|1)]]`.
    ///
    /// # Errors
    ///
    /// Returns [`ReadoutError::InvalidEntry`] for an entry outside `[0, 1]` and
    /// [`ReadoutError::ColumnSum`] when a column does not sum to one within
    /// `1e-6`.
    pub fn new(matrix: [[f64; 2]; 2]) -> Result<Self, ReadoutError> {
        for &entry in matrix.iter().flatten() {
            if !entry.is_finite() || !(0.0..=1.0).contains(&entry) {
                return Err(ReadoutError::InvalidEntry(entry));
            }
        }
        for (column, (top, bottom)) in matrix[0].iter().zip(matrix[1].iter()).enumerate() {
            let sum = top + bottom;
            if (sum - 1.0).abs() > 1e-6 {
                return Err(ReadoutError::ColumnSum { column, sum });
            }
        }
        Ok(Self { matrix })
    }

    /// The ideal readout, which never flips an outcome.
    pub fn ideal() -> Self {
        Self {
            matrix: [[1.0, 0.0], [0.0, 1.0]],
        }
    }

    /// The symmetric bit-flip matrix with error rate `epsilon`:
    /// `[[1 - epsilon, epsilon], [epsilon, 1 - epsilon]]`.
    ///
    /// # Errors
    ///
    /// Returns [`ReadoutError::InvalidEntry`] when `epsilon` is outside `[0, 1]`.
    pub fn symmetric(epsilon: f64) -> Result<Self, ReadoutError> {
        Self::new([[1.0 - epsilon, epsilon], [epsilon, 1.0 - epsilon]])
    }

    /// The underlying `[[P(0|0), P(0|1)], [P(1|0), P(1|1)]]` matrix.
    pub fn matrix(&self) -> [[f64; 2]; 2] {
        self.matrix
    }

    /// The inverse matrix, which maps an observed distribution back to the
    /// prepared one.
    ///
    /// # Errors
    ///
    /// Returns [`ReadoutError::Singular`] when the determinant is zero within
    /// `1e-12`.
    pub fn inverse(&self) -> Result<[[f64; 2]; 2], ReadoutError> {
        let [[a, b], [c, d]] = self.matrix;
        let determinant = a * d - b * c;
        if determinant.abs() < 1e-12 {
            return Err(ReadoutError::Singular);
        }
        let inverse_determinant = 1.0 / determinant;
        Ok([
            [d * inverse_determinant, -b * inverse_determinant],
            [-c * inverse_determinant, a * inverse_determinant],
        ])
    }

    /// Apply the forward channel to a prepared two-outcome distribution.
    pub fn apply(&self, prepared: [f64; 2]) -> [f64; 2] {
        [
            self.matrix[0][0] * prepared[0] + self.matrix[0][1] * prepared[1],
            self.matrix[1][0] * prepared[0] + self.matrix[1][1] * prepared[1],
        ]
    }

    /// Apply the inverse channel to an observed two-outcome distribution.
    ///
    /// # Errors
    ///
    /// Returns [`ReadoutError::Singular`] when the matrix is singular.
    pub fn mitigate(&self, observed: [f64; 2]) -> Result<[f64; 2], ReadoutError> {
        let inverse = self.inverse()?;
        Ok([
            inverse[0][0] * observed[0] + inverse[0][1] * observed[1],
            inverse[1][0] * observed[0] + inverse[1][1] * observed[1],
        ])
    }

    /// Iterative Bayesian unfolding (expectation maximisation): a
    /// positivity-preserving alternative to the matrix inverse.
    ///
    /// Each iteration applies the EM update
    /// `p(x) <- p(x) * sum_y M(y|x) p_obs(y) / p_pred(y)`, where `p_pred` is the
    /// forward image of the current estimate, then renormalizes. The fixed
    /// point satisfies `p_obs = M p`, i.e. it is the same estimate the inverse
    /// matrix gives, but every iterate stays non-negative.
    pub fn unfold(&self, observed: [f64; 2], iterations: usize) -> [f64; 2] {
        let total = observed[0] + observed[1];
        if total <= 0.0 {
            return [0.5, 0.5];
        }
        let observed = [observed[0] / total, observed[1] / total];
        let mut estimate = [0.5, 0.5];
        for _ in 0..iterations {
            let predicted = self.apply(estimate);
            let mut next = [0.0f64; 2];
            for prepared in 0..2 {
                let mut sum = 0.0;
                for outcome in 0..2 {
                    if predicted[outcome] > 1e-300 {
                        sum +=
                            self.matrix[outcome][prepared] * observed[outcome] / predicted[outcome];
                    }
                }
                next[prepared] = estimate[prepared] * sum;
            }
            let sum = next[0] + next[1];
            if sum > 0.0 {
                next[0] /= sum;
                next[1] /= sum;
            }
            estimate = next;
        }
        estimate
    }
}

/// Apply one inverse confusion matrix per qubit to a full computational-basis
/// distribution.
///
/// `distribution[index]` is the probability of the basis state with qubit `q`
/// at bit `q` (little-endian). `matrices[q]` is the readout error on qubit `q`.
/// The result is the inverse of the product channel, so it may contain small
/// negative entries.
///
/// # Errors
///
/// Returns [`ReadoutError::DimensionMismatch`] when the distribution length is
/// not `2^qubits`, and [`ReadoutError::Singular`] when a matrix is singular.
pub fn mitigate_distribution(
    distribution: &mut [f64],
    matrices: &[ConfusionMatrix],
) -> Result<(), ReadoutError> {
    let qubits = matrices.len();
    let dimension = 1usize << qubits;
    if distribution.len() != dimension {
        return Err(ReadoutError::DimensionMismatch {
            expected: dimension,
            actual: distribution.len(),
        });
    }
    for (qubit, matrix) in matrices.iter().enumerate() {
        let inverse = matrix.inverse()?;
        apply_single_qubit(distribution, qubit, inverse);
    }
    Ok(())
}

/// Parity expectation of the `watched` classical bits from `counts`, with
/// per-qubit readout error injected and then mitigated.
///
/// This is the correct shot-based VQE pipeline: sample, corrupt, mitigate,
/// estimate. Applying the inverse to counts that never passed through the
/// channel biases them (see the module docs), so the injection is not optional.
pub fn mitigated_parity_expectation(
    counts: &BTreeMap<Vec<u8>, usize>,
    watched: &[usize],
    matrices: &[ConfusionMatrix],
    shots: usize,
    rng: &mut SmallRng,
) -> Result<f64, ReadoutError> {
    if watched.is_empty() {
        return Err(ReadoutError::DimensionMismatch {
            expected: 1,
            actual: 0,
        });
    }
    if shots == 0 {
        return Err(ReadoutError::DimensionMismatch {
            expected: 1,
            actual: 0,
        });
    }
    let noisy = apply_readout_error(counts, matrices, rng);
    let support = watched.len();
    let dimension = 1usize << support;
    let mut distribution = vec![0.0f64; dimension];
    for (bits, count) in &noisy {
        let mut index = 0usize;
        for (position, &classical) in watched.iter().enumerate() {
            if bits.get(classical).copied().unwrap_or(0) != 0 {
                index |= 1usize << position;
            }
        }
        if index < dimension {
            distribution[index] += *count as f64 / shots as f64;
        }
    }
    let selected: Vec<ConfusionMatrix> = watched
        .iter()
        .map(|&qubit| {
            matrices
                .get(qubit)
                .copied()
                .ok_or(ReadoutError::DimensionMismatch {
                    expected: qubit + 1,
                    actual: matrices.len(),
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    mitigate_distribution(&mut distribution, &selected)?;
    let mut expectation = 0.0f64;
    for (index, probability) in distribution.iter().enumerate() {
        let sign = if index.count_ones().is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        expectation += sign * probability;
    }
    Ok(expectation)
}

/// Apply a 2x2 matrix to one qubit of a computational-basis distribution.
fn apply_single_qubit(distribution: &mut [f64], qubit: usize, matrix: [[f64; 2]; 2]) {
    let mask = 1usize << qubit;
    for base in 0..distribution.len() {
        if base & mask == 0 {
            let zero = distribution[base];
            let one = distribution[base | mask];
            distribution[base] = matrix[0][0] * zero + matrix[0][1] * one;
            distribution[base | mask] = matrix[1][0] * zero + matrix[1][1] * one;
        }
    }
}

/// Apply independent single-qubit readout errors to a set of shot counts.
///
/// Each outcome bit is resampled through the corresponding confusion matrix,
/// `P(observed = 1 | prepared = b) = matrix[1][b]`. This models a device with
/// the given readout error; pair it with [`ConfusionMatrix::mitigate`] or
/// [`crate::ShotResult::mitigate_readout`] to correct the counts.
pub fn apply_readout_error(
    counts: &BTreeMap<Vec<u8>, usize>,
    matrices: &[ConfusionMatrix],
    rng: &mut SmallRng,
) -> BTreeMap<Vec<u8>, usize> {
    let mut corrupted: BTreeMap<Vec<u8>, usize> = BTreeMap::new();
    for (bits, &count) in counts {
        for _ in 0..count {
            let mut flipped = bits.clone();
            for (qubit, bit) in flipped.iter_mut().enumerate() {
                let Some(matrix) = matrices.get(qubit) else {
                    break;
                };
                let m = matrix.matrix();
                let probability_one = if *bit == 0 { m[1][0] } else { m[1][1] };
                *bit = u8::from(rng.next_f64() < probability_one);
            }
            *corrupted.entry(flipped).or_insert(0) += 1;
        }
    }
    corrupted
}

/// Build a computational-basis distribution from shot counts.
///
/// `counts` maps least-significant-bit-first outcome strings to occurrence
/// counts; `shots` is the total number of shots. Outcomes outside `0..2^qubits`
/// are ignored.
pub fn distribution_from_counts(
    counts: &BTreeMap<Vec<u8>, usize>,
    qubits: usize,
    shots: usize,
) -> Vec<f64> {
    let dimension = 1usize << qubits;
    let mut distribution = vec![0.0f64; dimension];
    if shots == 0 {
        return distribution;
    }
    for (outcome, &count) in counts {
        if outcome.len() != qubits {
            continue;
        }
        let mut index = 0usize;
        for (position, &bit) in outcome.iter().enumerate() {
            if bit != 0 {
                index |= 1usize << position;
            }
        }
        if index < dimension {
            distribution[index] += count as f64 / shots as f64;
        }
    }
    distribution
}

//! Dimension-aware finite-dimensional Hilbert spaces and linear algebra.

use crate::{Complex64, Gate, SmallRng};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HilbertSpace {
    dimensions: Vec<usize>,
    dimension: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LinearError {
    EmptySpace,
    ZeroSubsystemDimension,
    DimensionOverflow,
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
    SparseEntryOutOfRange {
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
    },
    IncompatibleSpaces,
    BasisIndexOutOfRange {
        index: usize,
        dimension: usize,
    },
    QubitOutOfRange {
        qubit: usize,
        qubits: usize,
    },
    InvalidQubitPair {
        first: usize,
        second: usize,
    },
    InvalidProbability(f64),
    ZeroNorm,
    NotEndomorphism,
    NotHermitian,
    NotUnitary,
    NotQubitSpace,
    InvalidTolerance,
    EigenDecompositionNoConvergence,
    InvalidEigenpairCount {
        requested: usize,
        dimension: usize,
    },
    NonFiniteTime,
    EvolutionOverflow,
}
impl fmt::Display for LinearError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySpace => write!(f, "a Hilbert space must contain at least one subsystem"),
            Self::ZeroSubsystemDimension => write!(f, "subsystem dimensions must be positive"),
            Self::DimensionOverflow => write!(f, "Hilbert-space dimension overflow"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "expected dimension {expected}, got {actual}")
            }
            Self::SparseEntryOutOfRange {
                row,
                col,
                rows,
                cols,
            } => write!(
                f,
                "sparse entry ({row}, {col}) is outside matrix dimensions {rows}x{cols}"
            ),
            Self::IncompatibleSpaces => {
                write!(f, "quantum objects belong to incompatible Hilbert spaces")
            }
            Self::BasisIndexOutOfRange { index, dimension } => {
                write!(f, "basis index {index} is outside dimension {dimension}")
            }
            Self::QubitOutOfRange { qubit, qubits } => {
                write!(f, "qubit index {qubit} is out of range for {qubits} qubits")
            }
            Self::InvalidQubitPair { first, second } => {
                write!(
                    f,
                    "two-qubit operation needs distinct qubits, got {first} and {second}"
                )
            }
            Self::InvalidProbability(probability) => {
                write!(f, "measurement probability is invalid: {probability}")
            }
            Self::ZeroNorm => write!(f, "cannot normalize a zero or non-finite vector"),
            Self::NotEndomorphism => write!(f, "an observable must map a Hilbert space to itself"),
            Self::NotHermitian => write!(
                f,
                "an observable must be Hermitian within the requested tolerance"
            ),
            Self::NotUnitary => {
                write!(f, "operator must be unitary within the requested tolerance")
            }
            Self::NotQubitSpace => {
                write!(f, "operation requires a space composed only of qubits")
            }
            Self::InvalidTolerance => write!(f, "tolerance must be finite and non-negative"),
            Self::EigenDecompositionNoConvergence => {
                write!(
                    f,
                    "Hermitian eigensolver did not converge within the sweep limit"
                )
            }
            Self::InvalidEigenpairCount {
                requested,
                dimension,
            } => write!(
                f,
                "requested {requested} eigenpairs from a {dimension}-dimensional operator"
            ),
            Self::NonFiniteTime => write!(f, "time must be finite"),
            Self::EvolutionOverflow => {
                write!(f, "Hamiltonian phase overflowed floating-point range")
            }
        }
    }
}
impl Error for LinearError {}

impl HilbertSpace {
    pub fn new(dimensions: Vec<usize>) -> Result<Self, LinearError> {
        if dimensions.is_empty() {
            return Err(LinearError::EmptySpace);
        }
        if dimensions.contains(&0) {
            return Err(LinearError::ZeroSubsystemDimension);
        }
        let dimension = dimensions
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or(LinearError::DimensionOverflow)?;
        Ok(Self {
            dimensions,
            dimension,
        })
    }
    pub fn qubits(count: usize) -> Result<Self, LinearError> {
        Self::new(vec![2; count])
    }
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    pub fn subsystem_dimensions(&self) -> &[usize] {
        &self.dimensions
    }
    pub fn tensor(&self, rhs: &Self) -> Result<Self, LinearError> {
        let mut dimensions = self.dimensions.clone();
        dimensions.extend_from_slice(&rhs.dimensions);
        Self::new(dimensions)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Ket {
    space: HilbertSpace,
    amplitudes: Vec<Complex64>,
}
impl Ket {
    pub fn new(space: HilbertSpace, amplitudes: Vec<Complex64>) -> Result<Self, LinearError> {
        if amplitudes.len() != space.dimension {
            return Err(LinearError::DimensionMismatch {
                expected: space.dimension,
                actual: amplitudes.len(),
            });
        }
        Ok(Self { space, amplitudes })
    }
    pub fn basis(space: HilbertSpace, index: usize) -> Result<Self, LinearError> {
        if index >= space.dimension {
            return Err(LinearError::BasisIndexOutOfRange {
                index,
                dimension: space.dimension,
            });
        }
        let mut amplitudes = vec![Complex64::ZERO; space.dimension];
        amplitudes[index] = Complex64::ONE;
        Ok(Self { space, amplitudes })
    }
    pub fn space(&self) -> &HilbertSpace {
        &self.space
    }
    pub fn amplitudes(&self) -> &[Complex64] {
        &self.amplitudes
    }
    pub fn norm_sqr(&self) -> f64 {
        self.amplitudes.iter().map(|a| a.norm_sqr()).sum()
    }
    pub fn normalize(&mut self) -> Result<(), LinearError> {
        let norm = self.norm_sqr().sqrt();
        if norm == 0.0 || !norm.is_finite() {
            return Err(LinearError::ZeroNorm);
        }
        for x in &mut self.amplitudes {
            *x = *x / norm;
        }
        Ok(())
    }
    pub fn bra(&self) -> Bra {
        Bra {
            space: self.space.clone(),
            amplitudes: self.amplitudes.iter().map(|x| x.conj()).collect(),
        }
    }
    pub fn inner(&self, rhs: &Self) -> Result<Complex64, LinearError> {
        self.bra().inner(rhs)
    }
    pub fn outer(&self, bra: &Bra) -> Result<Operator, LinearError> {
        if self.space != bra.space {
            return Err(LinearError::IncompatibleSpaces);
        }
        let mut entries = Vec::with_capacity(self.space.dimension * self.space.dimension);
        for ket_value in &self.amplitudes {
            for bra_value in &bra.amplitudes {
                entries.push(*ket_value * *bra_value);
            }
        }
        Operator::new(self.space.clone(), self.space.clone(), entries)
    }
    pub fn apply(&self, operator: &Operator) -> Result<Self, LinearError> {
        operator.apply(self)
    }
    pub fn apply_map<M: LinearMap>(&self, operator: &M) -> Result<Self, LinearError> {
        operator.apply_ket(self)
    }
    pub fn expectation_map<M: LinearMap>(&self, operator: &M) -> Result<Complex64, LinearError> {
        if self.space != *operator.output_space() {
            return Err(LinearError::IncompatibleSpaces);
        }
        self.inner(&operator.apply_ket(self)?)
    }
    pub fn tensor(&self, rhs: &Self) -> Result<Self, LinearError> {
        let space = self.space.tensor(&rhs.space)?;
        let mut amplitudes = Vec::with_capacity(space.dimension);
        for a in &self.amplitudes {
            for b in &rhs.amplitudes {
                amplitudes.push(*a * *b);
            }
        }
        Self::new(space, amplitudes)
    }
    pub fn expectation(&self, observable: &Observable) -> Result<Complex64, LinearError> {
        observable.expectation_ket(self)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Bra {
    space: HilbertSpace,
    amplitudes: Vec<Complex64>,
}
impl Bra {
    pub fn space(&self) -> &HilbertSpace {
        &self.space
    }
    pub fn amplitudes(&self) -> &[Complex64] {
        &self.amplitudes
    }
    pub fn inner(&self, ket: &Ket) -> Result<Complex64, LinearError> {
        if self.space != ket.space {
            return Err(LinearError::IncompatibleSpaces);
        }
        Ok(self
            .amplitudes
            .iter()
            .zip(&ket.amplitudes)
            .map(|(a, b)| *a * *b)
            .fold(Complex64::ZERO, |x, y| x + y))
    }
    pub fn apply(&self, operator: &Operator) -> Result<Self, LinearError> {
        if self.space != operator.output {
            return Err(LinearError::IncompatibleSpaces);
        }
        let mut out = vec![Complex64::ZERO; operator.input.dimension];
        for (row, value) in self.amplitudes.iter().enumerate() {
            for (col, target) in out.iter_mut().enumerate() {
                *target = *target + *value * operator.entries[row * operator.input.dimension + col];
            }
        }
        Ok(Self {
            space: operator.input.clone(),
            amplitudes: out,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Operator {
    output: HilbertSpace,
    input: HilbertSpace,
    entries: Vec<Complex64>,
}
impl Operator {
    pub fn new(
        output: HilbertSpace,
        input: HilbertSpace,
        entries: Vec<Complex64>,
    ) -> Result<Self, LinearError> {
        let expected = output
            .dimension
            .checked_mul(input.dimension)
            .ok_or(LinearError::DimensionOverflow)?;
        if entries.len() != expected {
            return Err(LinearError::DimensionMismatch {
                expected,
                actual: entries.len(),
            });
        }
        Ok(Self {
            output,
            input,
            entries,
        })
    }
    pub fn identity(space: HilbertSpace) -> Self {
        let n = space.dimension;
        let mut entries = vec![Complex64::ZERO; n * n];
        for i in 0..n {
            entries[i * n + i] = Complex64::ONE;
        }
        Self {
            output: space.clone(),
            input: space,
            entries,
        }
    }
    pub fn pauli_x() -> Self {
        Self::new(
            HilbertSpace::qubits(1).unwrap(),
            HilbertSpace::qubits(1).unwrap(),
            vec![
                Complex64::ZERO,
                Complex64::ONE,
                Complex64::ONE,
                Complex64::ZERO,
            ],
        )
        .unwrap()
    }
    pub fn pauli_y() -> Self {
        Self::new(
            HilbertSpace::qubits(1).unwrap(),
            HilbertSpace::qubits(1).unwrap(),
            vec![
                Complex64::ZERO,
                Complex64::new(0.0, -1.0),
                Complex64::new(0.0, 1.0),
                Complex64::ZERO,
            ],
        )
        .unwrap()
    }
    pub fn pauli_z() -> Self {
        Self::new(
            HilbertSpace::qubits(1).unwrap(),
            HilbertSpace::qubits(1).unwrap(),
            vec![
                Complex64::ONE,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::new(-1.0, 0.0),
            ],
        )
        .unwrap()
    }
    pub fn hadamard() -> Self {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        Self::new(
            HilbertSpace::qubits(1).unwrap(),
            HilbertSpace::qubits(1).unwrap(),
            vec![
                Complex64::new(s, 0.0),
                Complex64::new(s, 0.0),
                Complex64::new(s, 0.0),
                Complex64::new(-s, 0.0),
            ],
        )
        .unwrap()
    }
    pub fn output_space(&self) -> &HilbertSpace {
        &self.output
    }
    pub fn input_space(&self) -> &HilbertSpace {
        &self.input
    }
    pub fn entries(&self) -> &[Complex64] {
        &self.entries
    }
    pub fn apply(&self, ket: &Ket) -> Result<Ket, LinearError> {
        self.apply_ket(ket)
    }
    pub fn adjoint(&self) -> Self {
        let mut entries = vec![Complex64::ZERO; self.entries.len()];
        for row in 0..self.output.dimension {
            for col in 0..self.input.dimension {
                entries[col * self.output.dimension + row] =
                    self.entries[row * self.input.dimension + col].conj();
            }
        }
        Self {
            output: self.input.clone(),
            input: self.output.clone(),
            entries,
        }
    }
    /// Compose self after rhs: `self ∘ rhs`.
    pub fn compose(&self, rhs: &Self) -> Result<Self, LinearError> {
        if self.input != rhs.output {
            return Err(LinearError::IncompatibleSpaces);
        }
        let (m, k, n) = (
            self.output.dimension,
            self.input.dimension,
            rhs.input.dimension,
        );
        let mut entries = vec![Complex64::ZERO; m * n];
        for i in 0..m {
            for j in 0..n {
                for t in 0..k {
                    entries[i * n + j] =
                        entries[i * n + j] + self.entries[i * k + t] * rhs.entries[t * n + j];
                }
            }
        }
        Self::new(self.output.clone(), rhs.input.clone(), entries)
    }
    pub fn trace(&self) -> Result<Complex64, LinearError> {
        if self.output != self.input {
            return Err(LinearError::IncompatibleSpaces);
        }
        let n = self.input.dimension;
        Ok((0..n)
            .map(|i| self.entries[i * n + i])
            .fold(Complex64::ZERO, |x, y| x + y))
    }
    pub fn expectation(&self, ket: &Ket) -> Result<Complex64, LinearError> {
        if self.input != ket.space || self.output != ket.space {
            return Err(LinearError::IncompatibleSpaces);
        }
        ket.bra().inner(&self.apply(ket)?)
    }
    pub fn tensor(&self, rhs: &Self) -> Result<Self, LinearError> {
        let output = self.output.tensor(&rhs.output)?;
        let input = self.input.tensor(&rhs.input)?;
        let (m1, n1, m2, n2) = (
            self.output.dimension,
            self.input.dimension,
            rhs.output.dimension,
            rhs.input.dimension,
        );
        let mut entries = vec![Complex64::ZERO; output.dimension * input.dimension];
        for i1 in 0..m1 {
            for j1 in 0..n1 {
                for i2 in 0..m2 {
                    for j2 in 0..n2 {
                        let row = i1 * m2 + i2;
                        let col = j1 * n2 + j2;
                        entries[row * input.dimension + col] =
                            self.entries[i1 * n1 + j1] * rhs.entries[i2 * n2 + j2];
                    }
                }
            }
        }
        Self::new(output, input, entries)
    }
    pub fn is_hermitian(&self, tolerance: f64) -> bool {
        if self.output != self.input || !tolerance.is_finite() || tolerance < 0.0 {
            return false;
        }
        let n = self.input.dimension;
        (0..n).all(|r| {
            (0..n).all(|c| {
                (self.entries[r * n + c] - self.entries[c * n + r].conj()).abs() <= tolerance
            })
        })
    }

    pub fn is_unitary(&self, tolerance: f64) -> bool {
        if self.output != self.input || !tolerance.is_finite() || tolerance < 0.0 {
            return false;
        }
        let n = self.input.dimension;
        (0..n).all(|row| {
            (0..n).all(|col| {
                let product = (0..n)
                    .map(|k| self.entries[k * n + row].conj() * self.entries[k * n + col])
                    .fold(Complex64::ZERO, |sum, value| sum + value);
                let expected = if row == col {
                    Complex64::ONE
                } else {
                    Complex64::ZERO
                };
                (product - expected).abs() <= tolerance
            })
        })
    }

    /// Diagonalize a dense Hermitian operator using cyclic complex Jacobi
    /// rotations. Eigenvalues are sorted in ascending order; eigenvectors are
    /// stored as columns in the returned unitary operator.
    pub fn hermitian_eigensystem(
        &self,
        tolerance: f64,
        max_sweeps: usize,
    ) -> Result<HermitianEigensystem, LinearError> {
        if self.output != self.input {
            return Err(LinearError::NotEndomorphism);
        }
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(LinearError::InvalidTolerance);
        }
        if !self.is_hermitian(tolerance) {
            return Err(LinearError::NotHermitian);
        }
        let n = self.input.dimension;
        let mut matrix = self.entries.clone();
        let mut vectors = Self::identity(self.input.clone()).entries;

        if n > 1 {
            let mut converged = max_off_diagonal_norm(&matrix, n) <= tolerance;
            for _ in 0..max_sweeps {
                if converged {
                    break;
                }
                for p in 0..n - 1 {
                    for q in p + 1..n {
                        let off_diagonal = matrix[p * n + q];
                        let magnitude = off_diagonal.abs();
                        if magnitude <= tolerance {
                            continue;
                        }
                        let a_pp = matrix[p * n + p].re;
                        let a_qq = matrix[q * n + q].re;
                        let tau = (a_qq - a_pp) / (2.0 * magnitude);
                        let t = if tau >= 0.0 {
                            1.0 / (tau + (1.0 + tau * tau).sqrt())
                        } else {
                            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                        };
                        let cosine = 1.0 / (1.0 + t * t).sqrt();
                        let sine = t * cosine;
                        let phase = off_diagonal / magnitude;
                        let phase_conj = phase.conj();

                        // Right multiply A and V by the local unitary.
                        for row in 0..n {
                            let left = matrix[row * n + p];
                            let right = matrix[row * n + q];
                            matrix[row * n + p] = left * cosine + right * (phase_conj * -sine);
                            matrix[row * n + q] = left * sine + right * (phase_conj * cosine);

                            let left = vectors[row * n + p];
                            let right = vectors[row * n + q];
                            vectors[row * n + p] = left * cosine + right * (phase_conj * -sine);
                            vectors[row * n + q] = left * sine + right * (phase_conj * cosine);
                        }
                        // Left multiply A by the adjoint of that unitary.
                        for col in 0..n {
                            let top = matrix[p * n + col];
                            let bottom = matrix[q * n + col];
                            matrix[p * n + col] = top * cosine + (phase * -sine) * bottom;
                            matrix[q * n + col] = top * sine + (phase * cosine) * bottom;
                        }
                        matrix[p * n + q] = Complex64::ZERO;
                        matrix[q * n + p] = Complex64::ZERO;
                        matrix[p * n + p].im = 0.0;
                        matrix[q * n + q].im = 0.0;
                    }
                }
                if max_off_diagonal_norm(&matrix, n) <= tolerance {
                    converged = true;
                    break;
                }
            }
            if !converged {
                return Err(LinearError::EigenDecompositionNoConvergence);
            }
        }

        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| matrix[a * n + a].re.total_cmp(&matrix[b * n + b].re));
        let values = order
            .iter()
            .map(|&index| matrix[index * n + index].re)
            .collect();
        let mut sorted_vectors = vec![Complex64::ZERO; n * n];
        for (column, &source_column) in order.iter().enumerate() {
            for row in 0..n {
                sorted_vectors[row * n + column] = vectors[row * n + source_column];
            }
        }
        Ok(HermitianEigensystem {
            eigenvalues: values,
            eigenvectors: Self::new(self.output.clone(), self.input.clone(), sorted_vectors)?,
        })
    }

    /// Construct `exp(-i H t)` for a dense Hermitian operator by spectral
    /// decomposition.
    pub fn evolution_operator(
        &self,
        time: f64,
        tolerance: f64,
        max_sweeps: usize,
    ) -> Result<Operator, LinearError> {
        if !time.is_finite() {
            return Err(LinearError::NonFiniteTime);
        }
        let decomposition = self.hermitian_eigensystem(tolerance, max_sweeps)?;
        let n = self.input.dimension;
        let phases: Vec<Complex64> = decomposition
            .eigenvalues
            .iter()
            .map(|&eigenvalue| -> Result<Complex64, LinearError> {
                let angle = -eigenvalue * time;
                if !angle.is_finite() {
                    return Err(LinearError::EvolutionOverflow);
                }
                Ok(Complex64::new(angle.cos(), angle.sin()))
            })
            .collect::<Result<_, _>>()?;
        let mut amplitudes = vec![Complex64::ZERO; n * n];
        for row in 0..n {
            for col in 0..n {
                for (eigenstate, &phase) in phases.iter().enumerate() {
                    amplitudes[row * n + col] = amplitudes[row * n + col]
                        + decomposition.eigenvectors.entries[row * n + eigenstate]
                            * phase
                            * decomposition.eigenvectors.entries[col * n + eigenstate].conj();
                }
            }
        }
        Operator::new(self.output.clone(), self.input.clone(), amplitudes)
    }

    /// Evolve a ket under a time-independent Hermitian Hamiltonian using its
    /// spectral decomposition: `|psi(t)> = exp(-i H t) |psi(0)>`.
    pub fn evolve_ket(
        &self,
        ket: &Ket,
        time: f64,
        tolerance: f64,
        max_sweeps: usize,
    ) -> Result<Ket, LinearError> {
        self.evolution_operator(time, tolerance, max_sweeps)?
            .apply(ket)
    }
}

/// A dimension-aware linear map with a shared ket application interface.
pub trait LinearMap {
    fn output_space(&self) -> &HilbertSpace;
    fn input_space(&self) -> &HilbertSpace;
    fn apply_amplitudes(&self, input: &[Complex64]) -> Result<Vec<Complex64>, LinearError>;

    fn apply_ket(&self, ket: &Ket) -> Result<Ket, LinearError> {
        if ket.space != *self.input_space() {
            return Err(LinearError::IncompatibleSpaces);
        }
        Ket::new(
            self.output_space().clone(),
            self.apply_amplitudes(&ket.amplitudes)?,
        )
    }
}

impl LinearMap for Operator {
    fn output_space(&self) -> &HilbertSpace {
        &self.output
    }
    fn input_space(&self) -> &HilbertSpace {
        &self.input
    }
    fn apply_amplitudes(&self, input: &[Complex64]) -> Result<Vec<Complex64>, LinearError> {
        if input.len() != self.input.dimension {
            return Err(LinearError::DimensionMismatch {
                expected: self.input.dimension,
                actual: input.len(),
            });
        }
        let mut output = vec![Complex64::ZERO; self.output.dimension];
        for (row, value) in output.iter_mut().enumerate() {
            for (col, input_value) in input.iter().enumerate() {
                *value = *value + self.entries[row * self.input.dimension + col] * *input_value;
            }
        }
        Ok(output)
    }
}

/// Compressed sparse row linear operator. Triplets supplied to `new` are
/// sorted, duplicate coordinates are summed, and explicit zeros are omitted.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseOperator {
    output: HilbertSpace,
    input: HilbertSpace,
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
    values: Vec<Complex64>,
}

impl SparseOperator {
    pub fn new(
        output: HilbertSpace,
        input: HilbertSpace,
        entries: Vec<(usize, usize, Complex64)>,
    ) -> Result<Self, LinearError> {
        let rows = output.dimension;
        let cols = input.dimension;
        let mut row_maps = vec![BTreeMap::<usize, Complex64>::new(); rows];
        for (row, col, value) in entries {
            if row >= rows || col >= cols {
                return Err(LinearError::SparseEntryOutOfRange {
                    row,
                    col,
                    rows,
                    cols,
                });
            }
            let accumulated = row_maps[row].entry(col).or_insert(Complex64::ZERO);
            *accumulated = *accumulated + value;
        }
        let mut row_offsets = Vec::with_capacity(rows + 1);
        let mut column_indices = Vec::new();
        let mut values = Vec::new();
        row_offsets.push(0);
        for row in row_maps {
            for (col, value) in row {
                if value != Complex64::ZERO {
                    column_indices.push(col);
                    values.push(value);
                }
            }
            row_offsets.push(values.len());
        }
        Ok(Self {
            output,
            input,
            row_offsets,
            column_indices,
            values,
        })
    }

    pub fn from_dense(operator: &Operator, tolerance: f64) -> Result<Self, LinearError> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(LinearError::InvalidTolerance);
        }
        let cols = operator.input.dimension;
        let entries = operator
            .entries
            .iter()
            .enumerate()
            .filter(|(_, value)| value.abs() > tolerance)
            .map(|(index, &value)| (index / cols, index % cols, value))
            .collect();
        Self::new(operator.output.clone(), operator.input.clone(), entries)
    }

    pub fn output_space(&self) -> &HilbertSpace {
        &self.output
    }
    pub fn input_space(&self) -> &HilbertSpace {
        &self.input
    }
    pub fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }
    pub fn column_indices(&self) -> &[usize] {
        &self.column_indices
    }
    pub fn values(&self) -> &[Complex64] {
        &self.values
    }
    pub fn nonzero_count(&self) -> usize {
        self.values.len()
    }

    pub fn apply(&self, ket: &Ket) -> Result<Ket, LinearError> {
        self.apply_ket(ket)
    }

    pub fn to_dense(&self) -> Result<Operator, LinearError> {
        let mut entries = vec![Complex64::ZERO; self.output.dimension * self.input.dimension];
        for row in 0..self.output.dimension {
            for position in self.row_offsets[row]..self.row_offsets[row + 1] {
                entries[row * self.input.dimension + self.column_indices[position]] =
                    self.values[position];
            }
        }
        Operator::new(self.output.clone(), self.input.clone(), entries)
    }

    pub fn expectation(&self, ket: &Ket) -> Result<Complex64, LinearError> {
        if self.output != self.input || ket.space != self.input {
            return Err(LinearError::IncompatibleSpaces);
        }
        ket.expectation_map(self)
    }

    pub fn is_hermitian(&self, tolerance: f64) -> bool {
        if self.output != self.input || !tolerance.is_finite() || tolerance < 0.0 {
            return false;
        }
        (0..self.output.dimension).all(|row| {
            (self.row_offsets[row]..self.row_offsets[row + 1]).all(|position| {
                let col = self.column_indices[position];
                (self.values[position] - self.value_at(col, row).conj()).abs() <= tolerance
            })
        })
    }

    /// Estimate the lowest eigenpairs of a sparse Hermitian operator with the
    /// Lanczos method. The operator remains in CSR form throughout; memory use
    /// is O(dimension * m + m^2), where m is the Krylov dimension. The tolerance
    /// is applied to the Ritz residual estimate, and failure to meet it is
    /// reported explicitly.
    pub fn lowest_eigenpairs(
        &self,
        count: usize,
        tolerance: f64,
        max_iterations: usize,
    ) -> Result<SparseEigensystem, LinearError> {
        if self.output != self.input {
            return Err(LinearError::NotEndomorphism);
        }
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(LinearError::InvalidTolerance);
        }
        if !self.is_hermitian(tolerance) {
            return Err(LinearError::NotHermitian);
        }
        let dimension = self.input.dimension;
        if count == 0 || count > dimension {
            return Err(LinearError::InvalidEigenpairCount {
                requested: count,
                dimension,
            });
        }
        if max_iterations == 0 {
            return Err(LinearError::EigenDecompositionNoConvergence);
        }

        let krylov_limit = max_iterations.min(dimension);
        let mut rng = SmallRng::seeded(0x51a7_9e37_79b9_7f4a);
        let mut first = (0..dimension)
            .map(|_| Complex64::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5))
            .collect::<Vec<_>>();
        normalize_amplitudes(&mut first)?;
        let mut basis = Vec::<Vec<Complex64>>::with_capacity(krylov_limit);
        let mut diagonal = Vec::<f64>::with_capacity(krylov_limit);
        let mut off_diagonal = Vec::<f64>::with_capacity(krylov_limit.saturating_sub(1));
        let mut current = first;
        let mut previous = vec![Complex64::ZERO; dimension];
        let mut previous_beta = 0.0;

        for iteration in 0..krylov_limit {
            basis.push(current.clone());
            let mut residual = self.apply_amplitudes(&current)?;
            if iteration > 0 {
                for (value, prior) in residual.iter_mut().zip(&previous) {
                    *value = *value - *prior * previous_beta;
                }
            }
            let alpha = inner_product(&current, &residual).re;
            for (value, amplitude) in residual.iter_mut().zip(&current) {
                *value = *value - *amplitude * alpha;
            }

            // Reorthogonalize against older Krylov vectors to limit loss of
            // orthogonality while retaining the tridiagonal Lanczos model.
            for vector in basis.iter().take(iteration) {
                let projection = inner_product(vector, &residual);
                for (value, amplitude) in residual.iter_mut().zip(vector) {
                    *value = *value - *amplitude * projection;
                }
            }

            let beta = vector_norm(&residual);
            diagonal.push(alpha);
            let tridiagonal = tridiagonal_operator(&diagonal, &off_diagonal)?;
            let decomposition = tridiagonal.hermitian_eigensystem(
                tolerance.max(f64::EPSILON),
                krylov_limit.saturating_mul(krylov_limit).max(1),
            )?;
            let requested_converged = diagonal.len() >= count
                && (0..count).all(|column| {
                    let last_component = decomposition.eigenvectors.entries()
                        [(diagonal.len() - 1) * diagonal.len() + column]
                        .abs();
                    beta * last_component <= tolerance || beta <= tolerance
                });
            if requested_converged || beta <= tolerance {
                if diagonal.len() < count {
                    return Err(LinearError::EigenDecompositionNoConvergence);
                }
                let mut eigenvectors = Vec::with_capacity(count);
                for column in 0..count {
                    let mut amplitudes = vec![Complex64::ZERO; dimension];
                    for (basis_index, vector) in basis.iter().enumerate() {
                        let coefficient = decomposition.eigenvectors.entries()
                            [basis_index * diagonal.len() + column]
                            .re;
                        for (amplitude, source) in amplitudes.iter_mut().zip(vector) {
                            *amplitude = *amplitude + *source * coefficient;
                        }
                    }
                    eigenvectors.push(Ket::new(self.input.clone(), amplitudes)?);
                }
                return Ok(SparseEigensystem {
                    eigenvalues: decomposition.eigenvalues()[..count].to_vec(),
                    eigenvectors,
                });
            }

            if iteration + 1 == krylov_limit {
                break;
            }
            off_diagonal.push(beta);
            previous = current;
            current = residual.into_iter().map(|value| value / beta).collect();
            previous_beta = beta;
        }
        Err(LinearError::EigenDecompositionNoConvergence)
    }

    /// Evolve a ket by `exp(-i H time)` using a Lanczos Krylov approximation.
    /// The sparse Hamiltonian is never converted to a dense operator. `tolerance`
    /// is applied to the residual-based evolution estimate; insufficient Krylov
    /// iterations return `EigenDecompositionNoConvergence`.
    pub fn evolve_ket(
        &self,
        ket: &Ket,
        time: f64,
        tolerance: f64,
        max_iterations: usize,
    ) -> Result<Ket, LinearError> {
        if self.output != self.input {
            return Err(LinearError::NotEndomorphism);
        }
        if ket.space != self.input {
            return Err(LinearError::IncompatibleSpaces);
        }
        if !time.is_finite() {
            return Err(LinearError::NonFiniteTime);
        }
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(LinearError::InvalidTolerance);
        }
        if !self.is_hermitian(tolerance) {
            return Err(LinearError::NotHermitian);
        }
        if max_iterations == 0 {
            return Err(LinearError::EigenDecompositionNoConvergence);
        }
        if time == 0.0 {
            return Ok(ket.clone());
        }

        let dimension = self.input.dimension;
        let input_norm = vector_norm(&ket.amplitudes);
        if !input_norm.is_finite() || input_norm <= f64::MIN_POSITIVE {
            return Err(LinearError::ZeroNorm);
        }
        let mut current = ket
            .amplitudes
            .iter()
            .map(|amplitude| *amplitude / input_norm)
            .collect::<Vec<_>>();
        let krylov_limit = max_iterations.min(dimension);
        let mut basis = Vec::<Vec<Complex64>>::with_capacity(krylov_limit);
        let mut diagonal = Vec::<f64>::with_capacity(krylov_limit);
        let mut off_diagonal = Vec::<f64>::with_capacity(krylov_limit.saturating_sub(1));
        let mut previous = vec![Complex64::ZERO; dimension];
        let mut previous_beta = 0.0;

        for iteration in 0..krylov_limit {
            basis.push(current.clone());
            let mut residual = self.apply_amplitudes(&current)?;
            if iteration > 0 {
                for (value, prior) in residual.iter_mut().zip(&previous) {
                    *value = *value - *prior * previous_beta;
                }
            }
            let alpha = inner_product(&current, &residual).re;
            for (value, amplitude) in residual.iter_mut().zip(&current) {
                *value = *value - *amplitude * alpha;
            }
            for vector in basis.iter().take(iteration) {
                let projection = inner_product(vector, &residual);
                for (value, amplitude) in residual.iter_mut().zip(vector) {
                    *value = *value - *amplitude * projection;
                }
            }

            let beta = vector_norm(&residual);
            diagonal.push(alpha);
            let projected = tridiagonal_operator(&diagonal, &off_diagonal)?.hermitian_eigensystem(
                tolerance.max(f64::EPSILON),
                krylov_limit.saturating_mul(krylov_limit).max(1),
            )?;
            let krylov_dimension = diagonal.len();
            let mut projected_state = vec![Complex64::ZERO; krylov_dimension];
            for eigenstate in 0..krylov_dimension {
                let eigenvalue = projected.eigenvalues()[eigenstate];
                let phase = eigenvalue * time;
                if !phase.is_finite() {
                    return Err(LinearError::EvolutionOverflow);
                }
                let factor = Complex64::new(phase.cos(), -phase.sin());
                let initial_component = projected.eigenvectors().entries()[eigenstate].conj();
                for (row, value) in projected_state.iter_mut().enumerate() {
                    let eigenvector_component =
                        projected.eigenvectors().entries()[row * krylov_dimension + eigenstate];
                    *value = *value + eigenvector_component * factor * initial_component;
                }
            }

            let residual_estimate =
                time.abs() * beta * projected_state[krylov_dimension - 1].abs() * input_norm;
            if residual_estimate <= tolerance || beta <= tolerance {
                let mut amplitudes = vec![Complex64::ZERO; dimension];
                for (vector, coefficient) in basis.iter().zip(projected_state) {
                    for (amplitude, source) in amplitudes.iter_mut().zip(vector) {
                        *amplitude = *amplitude + *source * coefficient * input_norm;
                    }
                }
                return Ket::new(self.input.clone(), amplitudes);
            }

            if iteration + 1 == krylov_limit {
                break;
            }
            off_diagonal.push(beta);
            previous = current;
            current = residual.into_iter().map(|value| value / beta).collect();
            previous_beta = beta;
        }
        Err(LinearError::EigenDecompositionNoConvergence)
    }

    fn value_at(&self, row: usize, col: usize) -> Complex64 {
        let range = self.row_offsets[row]..self.row_offsets[row + 1];
        match self.column_indices[range.clone()].binary_search(&col) {
            Ok(index) => self.values[range.start + index],
            Err(_) => Complex64::ZERO,
        }
    }

    pub fn adjoint(&self) -> Self {
        let mut entries = Vec::with_capacity(self.values.len());
        for row in 0..self.output.dimension {
            for position in self.row_offsets[row]..self.row_offsets[row + 1] {
                entries.push((
                    self.column_indices[position],
                    row,
                    self.values[position].conj(),
                ));
            }
        }
        Self::new(self.input.clone(), self.output.clone(), entries)
            .expect("transposed sparse entries remain in range")
    }

    /// Compose this map after `rhs`, preserving sparsity during multiplication.
    pub fn compose(&self, rhs: &Self) -> Result<Self, LinearError> {
        if self.input != rhs.output {
            return Err(LinearError::IncompatibleSpaces);
        }
        let mut entries = Vec::new();
        for row in 0..self.output.dimension {
            let mut accumulated = BTreeMap::<usize, Complex64>::new();
            for left_position in self.row_offsets[row]..self.row_offsets[row + 1] {
                let middle = self.column_indices[left_position];
                let left_value = self.values[left_position];
                for right_position in rhs.row_offsets[middle]..rhs.row_offsets[middle + 1] {
                    let col = rhs.column_indices[right_position];
                    let value = left_value * rhs.values[right_position];
                    let slot = accumulated.entry(col).or_insert(Complex64::ZERO);
                    *slot = *slot + value;
                }
            }
            entries.extend(
                accumulated
                    .into_iter()
                    .filter(|(_, value)| *value != Complex64::ZERO)
                    .map(|(col, value)| (row, col, value)),
            );
        }
        Self::new(self.output.clone(), rhs.input.clone(), entries)
    }
}

/// Evolve a ket under a time-dependent sparse Hermitian Hamiltonian.
///
/// The callback is sampled at midpoint times. Adaptive step doubling compares
/// one midpoint step with two half steps, accepts the two-half-step result, and
/// uses `tolerance` as the accumulated local-error budget. Each propagation
/// step uses the matrix-free sparse Lanczos method. `max_steps` counts both
/// accepted and rejected attempts.
pub fn evolve_ket_time_dependent<F>(
    initial: &Ket,
    start_time: f64,
    end_time: f64,
    tolerance: f64,
    max_steps: usize,
    mut hamiltonian_at: F,
) -> Result<Ket, LinearError>
where
    F: FnMut(f64) -> Result<SparseOperator, LinearError>,
{
    if !start_time.is_finite() || !end_time.is_finite() {
        return Err(LinearError::NonFiniteTime);
    }
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(LinearError::InvalidTolerance);
    }
    if max_steps == 0 {
        return Err(LinearError::EigenDecompositionNoConvergence);
    }
    if start_time == end_time {
        return Ok(initial.clone());
    }

    let total_interval = end_time - start_time;
    if !total_interval.is_finite() {
        return Err(LinearError::NonFiniteTime);
    }
    let total_duration = total_interval.abs();
    let mut state = initial.clone();
    let mut time = start_time;
    let mut step = total_interval;
    let krylov_iterations = initial.space.dimension.clamp(1, 32);

    for _ in 0..max_steps {
        let remaining = end_time - time;
        if remaining == 0.0 {
            return Ok(state);
        }
        if step.abs() > remaining.abs() {
            step = remaining;
        }
        if time + step == time {
            return Err(LinearError::EigenDecompositionNoConvergence);
        }

        let local_budget = tolerance * (step.abs() / total_duration);
        let half_step = step * 0.5;
        let propagated = (|| {
            let one_step_hamiltonian = hamiltonian_at(time + step * 0.5)?;
            let one_step = one_step_hamiltonian.evolve_ket(
                &state,
                step,
                local_budget * 0.1,
                krylov_iterations,
            )?;
            let first_half_hamiltonian = hamiltonian_at(time + half_step * 0.5)?;
            let first_half = first_half_hamiltonian.evolve_ket(
                &state,
                half_step,
                local_budget * 0.05,
                krylov_iterations,
            )?;
            let second_half_hamiltonian = hamiltonian_at(time + half_step + half_step * 0.5)?;
            let two_half_steps = second_half_hamiltonian.evolve_ket(
                &first_half,
                half_step,
                local_budget * 0.05,
                krylov_iterations,
            )?;
            Ok::<_, LinearError>((one_step, two_half_steps))
        })();
        let (one_step, two_half_steps) = match propagated {
            Ok(states) => states,
            Err(LinearError::EigenDecompositionNoConvergence) => {
                step *= 0.5;
                continue;
            }
            Err(error) => return Err(error),
        };

        let error = ket_distance(&one_step, &two_half_steps)? / 3.0;
        if !error.is_finite() {
            return Err(LinearError::EvolutionOverflow);
        }
        if error <= local_budget {
            state = two_half_steps;
            time = if step.abs() >= remaining.abs() {
                end_time
            } else {
                time + step
            };
            let scale = if error == 0.0 {
                2.0
            } else {
                (0.9 * (local_budget / error).cbrt()).clamp(0.5, 2.0)
            };
            step *= scale;
        } else {
            let scale = (0.8 * (local_budget / error).cbrt()).clamp(0.1, 0.8);
            step *= scale;
        }
    }
    if time == end_time {
        Ok(state)
    } else {
        Err(LinearError::EigenDecompositionNoConvergence)
    }
}

fn ket_distance(left: &Ket, right: &Ket) -> Result<f64, LinearError> {
    if left.space != right.space {
        return Err(LinearError::IncompatibleSpaces);
    }
    Ok(left
        .amplitudes
        .iter()
        .zip(&right.amplitudes)
        .map(|(a, b)| (*a - *b).norm_sqr())
        .sum::<f64>()
        .sqrt())
}

impl LinearMap for SparseOperator {
    fn output_space(&self) -> &HilbertSpace {
        &self.output
    }
    fn input_space(&self) -> &HilbertSpace {
        &self.input
    }
    fn apply_amplitudes(&self, input: &[Complex64]) -> Result<Vec<Complex64>, LinearError> {
        if input.len() != self.input.dimension {
            return Err(LinearError::DimensionMismatch {
                expected: self.input.dimension,
                actual: input.len(),
            });
        }
        let mut output = vec![Complex64::ZERO; self.output.dimension];
        for (row, value) in output.iter_mut().enumerate() {
            for position in self.row_offsets[row]..self.row_offsets[row + 1] {
                *value = *value + self.values[position] * input[self.column_indices[position]];
            }
        }
        Ok(output)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SparseObservable {
    operator: SparseOperator,
}
impl SparseObservable {
    pub fn new(operator: SparseOperator, tolerance: f64) -> Result<Self, LinearError> {
        if operator.output != operator.input {
            return Err(LinearError::NotEndomorphism);
        }
        if !operator.is_hermitian(tolerance) {
            return Err(LinearError::NotHermitian);
        }
        Ok(Self { operator })
    }
    pub fn operator(&self) -> &SparseOperator {
        &self.operator
    }
    pub fn expectation_ket(&self, ket: &Ket) -> Result<Complex64, LinearError> {
        self.operator.expectation(ket)
    }
}

/// Sparse pure state represented by its nonzero computational-basis
/// amplitudes. The basis index uses the same little-endian qubit convention as
/// `StateVector` (qubit 0 is the least-significant bit).
#[derive(Clone, Debug, PartialEq)]
pub struct SparseKet {
    space: HilbertSpace,
    amplitudes: BTreeMap<usize, Complex64>,
}

impl SparseKet {
    pub fn new(
        space: HilbertSpace,
        entries: impl IntoIterator<Item = (usize, Complex64)>,
    ) -> Result<Self, LinearError> {
        let mut amplitudes = BTreeMap::new();
        for (index, amplitude) in entries {
            if index >= space.dimension {
                return Err(LinearError::BasisIndexOutOfRange {
                    index,
                    dimension: space.dimension,
                });
            }
            let accumulated = amplitudes.entry(index).or_insert(Complex64::ZERO);
            *accumulated = *accumulated + amplitude;
        }
        amplitudes.retain(|_, value| *value != Complex64::ZERO);
        Ok(Self { space, amplitudes })
    }

    pub fn basis(space: HilbertSpace, index: usize) -> Result<Self, LinearError> {
        Self::new(space, [(index, Complex64::ONE)])
    }

    pub fn from_ket(ket: &Ket) -> Self {
        let amplitudes = ket
            .amplitudes
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, amplitude)| *amplitude != Complex64::ZERO)
            .collect();
        Self {
            space: ket.space.clone(),
            amplitudes,
        }
    }

    pub fn to_ket(&self) -> Result<Ket, LinearError> {
        let mut amplitudes = vec![Complex64::ZERO; self.space.dimension];
        for (&index, &amplitude) in &self.amplitudes {
            amplitudes[index] = amplitude;
        }
        Ket::new(self.space.clone(), amplitudes)
    }

    pub fn space(&self) -> &HilbertSpace {
        &self.space
    }
    pub fn nonzero_count(&self) -> usize {
        self.amplitudes.len()
    }
    pub fn amplitudes(&self) -> &BTreeMap<usize, Complex64> {
        &self.amplitudes
    }
    pub fn amplitude(&self, basis_index: usize) -> Result<Complex64, LinearError> {
        if basis_index >= self.space.dimension {
            return Err(LinearError::BasisIndexOutOfRange {
                index: basis_index,
                dimension: self.space.dimension,
            });
        }
        Ok(self
            .amplitudes
            .get(&basis_index)
            .copied()
            .unwrap_or(Complex64::ZERO))
    }
    pub fn norm_sqr(&self) -> f64 {
        self.amplitudes
            .values()
            .map(|amplitude| amplitude.norm_sqr())
            .sum()
    }
    pub fn normalize(&mut self) -> Result<(), LinearError> {
        let norm = self.norm_sqr().sqrt();
        if norm == 0.0 || !norm.is_finite() {
            return Err(LinearError::ZeroNorm);
        }
        for amplitude in self.amplitudes.values_mut() {
            *amplitude = *amplitude / norm;
        }
        Ok(())
    }
    pub fn inner(&self, rhs: &Self) -> Result<Complex64, LinearError> {
        if self.space != rhs.space {
            return Err(LinearError::IncompatibleSpaces);
        }
        Ok(self
            .amplitudes
            .iter()
            .map(|(index, left)| {
                left.conj()
                    * rhs
                        .amplitudes
                        .get(index)
                        .copied()
                        .unwrap_or(Complex64::ZERO)
            })
            .fold(Complex64::ZERO, |sum, value| sum + value))
    }
    pub fn apply_sparse(&self, operator: &SparseOperator) -> Result<Self, LinearError> {
        if self.space != operator.input {
            return Err(LinearError::IncompatibleSpaces);
        }
        let mut output = BTreeMap::<usize, Complex64>::new();
        for row in 0..operator.output.dimension {
            let mut value = Complex64::ZERO;
            for position in operator.row_offsets[row]..operator.row_offsets[row + 1] {
                if let Some(amplitude) = self.amplitudes.get(&operator.column_indices[position]) {
                    value = value + operator.values[position] * *amplitude;
                }
            }
            if value != Complex64::ZERO {
                output.insert(row, value);
            }
        }
        Ok(Self {
            space: operator.output.clone(),
            amplitudes: output,
        })
    }
    pub fn expectation(&self, observable: &SparseObservable) -> Result<Complex64, LinearError> {
        if observable.operator.input != self.space {
            return Err(LinearError::IncompatibleSpaces);
        }
        self.inner(&self.apply_sparse(&observable.operator)?)
    }
    pub fn tensor(&self, rhs: &Self) -> Result<Self, LinearError> {
        let space = self.space.tensor(&rhs.space)?;
        let mut amplitudes = BTreeMap::new();
        for (&left_index, &left_value) in &self.amplitudes {
            for (&right_index, &right_value) in &rhs.amplitudes {
                amplitudes.insert(
                    left_index * rhs.space.dimension + right_index,
                    left_value * right_value,
                );
            }
        }
        Ok(Self { space, amplitudes })
    }

    pub fn apply_gate(&mut self, gate: Gate, qubit: usize) -> Result<(), LinearError> {
        let qubits = self.qubit_count()?;
        if qubit >= qubits {
            return Err(LinearError::QubitOutOfRange { qubit, qubits });
        }
        let mask = 1usize << qubit;
        let bases: std::collections::BTreeSet<usize> =
            self.amplitudes.keys().map(|&index| index & !mask).collect();
        let (m00, m01, m10, m11) = gate.matrix();
        let mut output = BTreeMap::new();
        for base in bases {
            let one = base | mask;
            let a0 = self
                .amplitudes
                .get(&base)
                .copied()
                .unwrap_or(Complex64::ZERO);
            let a1 = self
                .amplitudes
                .get(&one)
                .copied()
                .unwrap_or(Complex64::ZERO);
            insert_nonzero(&mut output, base, m00 * a0 + m01 * a1);
            insert_nonzero(&mut output, one, m10 * a0 + m11 * a1);
        }
        self.amplitudes = output;
        Ok(())
    }

    pub fn apply_cnot(&mut self, control: usize, target: usize) -> Result<(), LinearError> {
        self.validate_pair(control, target)?;
        let control_mask = 1usize << control;
        let target_mask = 1usize << target;
        self.permute_basis(|index| {
            if index & control_mask != 0 {
                index ^ target_mask
            } else {
                index
            }
        });
        Ok(())
    }
    pub fn apply_cz(&mut self, first: usize, second: usize) -> Result<(), LinearError> {
        self.validate_pair(first, second)?;
        let first_mask = 1usize << first;
        let second_mask = 1usize << second;
        for (&index, amplitude) in &mut self.amplitudes {
            if index & first_mask != 0 && index & second_mask != 0 {
                *amplitude = *amplitude * -1.0;
            }
        }
        Ok(())
    }
    pub fn apply_swap(&mut self, first: usize, second: usize) -> Result<(), LinearError> {
        self.validate_pair(first, second)?;
        let first_mask = 1usize << first;
        let second_mask = 1usize << second;
        self.permute_basis(|index| {
            if (index & first_mask == 0) != (index & second_mask == 0) {
                index ^ first_mask ^ second_mask
            } else {
                index
            }
        });
        Ok(())
    }

    pub fn probability_one(&self, qubit: usize) -> Result<f64, LinearError> {
        let qubits = self.qubit_count()?;
        if qubit >= qubits {
            return Err(LinearError::QubitOutOfRange { qubit, qubits });
        }
        let mask = 1usize << qubit;
        Ok(self
            .amplitudes
            .iter()
            .filter(|(index, _)| **index & mask != 0)
            .map(|(_, amplitude)| amplitude.norm_sqr())
            .sum::<f64>()
            .clamp(0.0, 1.0))
    }
    pub fn expectation_z(&self, qubit: usize) -> Result<f64, LinearError> {
        let qubits = self.qubit_count()?;
        if qubit >= qubits {
            return Err(LinearError::QubitOutOfRange { qubit, qubits });
        }
        let mask = 1usize << qubit;
        Ok(self
            .amplitudes
            .iter()
            .map(|(&index, amplitude)| {
                if index & mask == 0 {
                    amplitude.norm_sqr()
                } else {
                    -amplitude.norm_sqr()
                }
            })
            .sum())
    }
    pub fn measure(&mut self, qubit: usize, rng: &mut SmallRng) -> Result<u8, LinearError> {
        let qubits = self.qubit_count()?;
        if qubit >= qubits {
            return Err(LinearError::QubitOutOfRange { qubit, qubits });
        }
        let probability_one = self.probability_one(qubit)?;
        let outcome = u8::from(rng.next_f64() < probability_one);
        let probability = if outcome == 1 {
            probability_one
        } else {
            1.0 - probability_one
        };
        if probability <= f64::EPSILON {
            return Err(LinearError::InvalidProbability(probability));
        }
        let mask = 1usize << qubit;
        let scale = probability.sqrt();
        self.amplitudes.retain(|index, amplitude| {
            if u8::from(*index & mask != 0) == outcome {
                *amplitude = *amplitude / scale;
                true
            } else {
                false
            }
        });
        Ok(outcome)
    }

    fn qubit_count(&self) -> Result<usize, LinearError> {
        if self
            .space
            .dimensions
            .iter()
            .all(|&dimension| dimension == 2)
        {
            Ok(self.space.dimensions.len())
        } else {
            Err(LinearError::NotQubitSpace)
        }
    }
    fn validate_pair(&self, first: usize, second: usize) -> Result<(), LinearError> {
        let qubits = self.qubit_count()?;
        if first >= qubits {
            return Err(LinearError::QubitOutOfRange {
                qubit: first,
                qubits,
            });
        }
        if second >= qubits {
            return Err(LinearError::QubitOutOfRange {
                qubit: second,
                qubits,
            });
        }
        if first == second {
            return Err(LinearError::InvalidQubitPair { first, second });
        }
        Ok(())
    }
    fn permute_basis(&mut self, permutation: impl Fn(usize) -> usize) {
        self.amplitudes = self
            .amplitudes
            .iter()
            .map(|(&index, &amplitude)| (permutation(index), amplitude))
            .collect();
    }
}

fn insert_nonzero(amplitudes: &mut BTreeMap<usize, Complex64>, index: usize, value: Complex64) {
    if value != Complex64::ZERO {
        amplitudes.insert(index, value);
    }
}

fn max_off_diagonal_norm(matrix: &[Complex64], dimension: usize) -> f64 {
    (0..dimension)
        .flat_map(|row| (0..dimension).map(move |col| (row, col)))
        .filter(|(row, col)| row != col)
        .map(|(row, col)| matrix[row * dimension + col].abs())
        .fold(0.0_f64, f64::max)
}

#[derive(Clone, Debug, PartialEq)]
pub struct HermitianEigensystem {
    eigenvalues: Vec<f64>,
    eigenvectors: Operator,
}
impl HermitianEigensystem {
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }
    pub fn eigenvectors(&self) -> &Operator {
        &self.eigenvectors
    }
}

/// Ritz eigenpairs returned by the matrix-free sparse Hermitian solver.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseEigensystem {
    eigenvalues: Vec<f64>,
    eigenvectors: Vec<Ket>,
}
impl SparseEigensystem {
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }
    pub fn eigenvectors(&self) -> &[Ket] {
        &self.eigenvectors
    }
}

fn inner_product(left: &[Complex64], right: &[Complex64]) -> Complex64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| a.conj() * *b)
        .fold(Complex64::ZERO, |sum, value| sum + value)
}

fn vector_norm(vector: &[Complex64]) -> f64 {
    inner_product(vector, vector).re.max(0.0).sqrt()
}

fn normalize_amplitudes(vector: &mut [Complex64]) -> Result<(), LinearError> {
    let norm = vector_norm(vector);
    if !norm.is_finite() || norm <= f64::MIN_POSITIVE {
        return Err(LinearError::ZeroNorm);
    }
    for amplitude in vector {
        *amplitude = *amplitude / norm;
    }
    Ok(())
}

fn tridiagonal_operator(diagonal: &[f64], off_diagonal: &[f64]) -> Result<Operator, LinearError> {
    let dimension = diagonal.len();
    let space = HilbertSpace::new(vec![dimension])?;
    let mut entries = vec![Complex64::ZERO; dimension * dimension];
    for (index, value) in diagonal.iter().enumerate() {
        entries[index * dimension + index] = Complex64::new(*value, 0.0);
    }
    for (index, value) in off_diagonal.iter().enumerate() {
        entries[index * dimension + index + 1] = Complex64::new(*value, 0.0);
        entries[(index + 1) * dimension + index] = Complex64::new(*value, 0.0);
    }
    Operator::new(space.clone(), space, entries)
}

/// A square Hermitian operator that can be used as a real-valued observable.
#[derive(Clone, Debug, PartialEq)]
pub struct Observable {
    operator: Operator,
}
impl Observable {
    pub fn new(operator: Operator, tolerance: f64) -> Result<Self, LinearError> {
        if operator.output != operator.input {
            return Err(LinearError::NotEndomorphism);
        }
        if !operator.is_hermitian(tolerance) {
            return Err(LinearError::NotHermitian);
        }
        Ok(Self { operator })
    }
    pub fn operator(&self) -> &Operator {
        &self.operator
    }
    pub fn expectation_ket(&self, ket: &Ket) -> Result<Complex64, LinearError> {
        self.operator.expectation(ket)
    }
    pub fn expectation_density(
        &self,
        density: &crate::DensityMatrix,
    ) -> Result<Complex64, LinearError> {
        let expected =
            HilbertSpace::qubits(density.qubits()).map_err(|_| LinearError::DimensionOverflow)?;
        if self.operator.input != expected {
            return Err(LinearError::IncompatibleSpaces);
        }
        let n = expected.dimension();
        let mut value = Complex64::ZERO;
        for r in 0..n {
            for c in 0..n {
                value = value + density.entries()[r * n + c] * self.operator.entries[c * n + r];
            }
        }
        Ok(value)
    }
}

#[cfg(feature = "gpu")]
pub fn sparse_lanczos_gpu(op: &SparseOperator, max_iterations: usize, tolerance: f64) -> Result<f64, LinearError> {
    use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
    use cudarc::nvrtc::compile_ptx;

    let dimension = op.input.dimension;
    if op.output != op.input {
        return Err(LinearError::NotEndomorphism);
    }

    let device = CudaDevice::new(0).map_err(|_| LinearError::EigenDecompositionNoConvergence)?;

    let row_offsets: Vec<i32> = op.row_offsets.iter().map(|&x| x as i32).collect();
    let col_indices: Vec<i32> = op.column_indices.iter().map(|&x| x as i32).collect();
    let values: Vec<f64> = op.values.iter().map(|c| c.re).collect();

    let kernel_src = r#"
extern "C" __global__ void sparse_matvec(
    const double* v, double* w,
    const int* row_offsets, const int* col_indices,
    const double* values, int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    double sum = 0.0;
    for (int j = row_offsets[i]; j < row_offsets[i+1]; j++) {
        sum += values[j] * v[col_indices[j]];
    }
    w[i] = sum;
}
"#;

    let ptx = compile_ptx(kernel_src).map_err(|_| LinearError::EigenDecompositionNoConvergence)?;
    device
        .load_ptx(ptx, "sparse", &["sparse_matvec"])
        .map_err(|_| LinearError::EigenDecompositionNoConvergence)?;
    let func = device
        .get_func("sparse", "sparse_matvec")
        .ok_or(LinearError::EigenDecompositionNoConvergence)?;

    let m = max_iterations.min(dimension).min(100);
    let mut v_curr = vec![0.0f64; dimension];
    let mut seed = 42u64;
    for x in v_curr.iter_mut() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *x = (seed as f64 / u64::MAX as f64) - 0.5;
    }
    let norm: f64 = v_curr.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in v_curr.iter_mut() {
        *x /= norm;
    }

    let mut v_prev = vec![0.0f64; dimension];
    let mut v_all: Vec<Vec<f64>> = Vec::with_capacity(m);
    v_all.push(v_curr.clone());

    let mut alpha = vec![0.0f64; m];
    let mut beta = vec![0.0f64; m];

    let row_gpu = device.htod_copy(row_offsets.clone()).map_err(|_| LinearError::EigenDecompositionNoConvergence)?;
    let col_gpu = device.htod_copy(col_indices.clone()).map_err(|_| LinearError::EigenDecompositionNoConvergence)?;
    let val_gpu = device.htod_copy(values.clone()).map_err(|_| LinearError::EigenDecompositionNoConvergence)?;
    let mut v_gpu = device.htod_copy(v_curr.clone()).map_err(|_| LinearError::EigenDecompositionNoConvergence)?;
    let mut w_gpu = device.alloc_zeros::<f64>(dimension).map_err(|_| LinearError::EigenDecompositionNoConvergence)?;

    for j in 0..m {
        let cfg = LaunchConfig {
            grid_dim: ((dimension as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            func.clone()
                .launch(cfg, (&v_gpu, &mut w_gpu, &row_gpu, &col_gpu, &val_gpu, dimension as i32))
                .map_err(|_| LinearError::EigenDecompositionNoConvergence)?;
        }
        device.synchronize().map_err(|_| LinearError::EigenDecompositionNoConvergence)?;

        let w: Vec<f64> = device.dtoh_sync_copy(&w_gpu).map_err(|_| LinearError::EigenDecompositionNoConvergence)?;

        let mut w = w;
        if j > 0 {
            for i in 0..dimension {
                w[i] -= beta[j - 1] * v_prev[i];
            }
        }
        for vec in v_all.iter().take(j) {
            let proj: f64 = (0..dimension).map(|i| vec[i] * w[i]).sum();
            for i in 0..dimension {
                w[i] -= proj * vec[i];
            }
        }
        let alpha_j: f64 = (0..dimension).map(|i| v_curr[i] * w[i]).sum();
        for i in 0..dimension {
            w[i] -= alpha_j * v_curr[i];
        }
        let beta_j: f64 = (0..dimension).map(|i| w[i] * w[i]).sum::<f64>().sqrt();

        alpha[j] = alpha_j;
        if j + 1 < m {
            beta[j] = beta_j;
        }

        if beta_j < tolerance || j + 1 == m {
            break;
        }

        v_prev = v_curr.clone();
        v_curr = w.iter().map(|x| x / beta_j).collect();
        v_all.push(v_curr.clone());
        v_gpu = device.htod_copy(v_curr.clone()).map_err(|_| LinearError::EigenDecompositionNoConvergence)?;
    }

    // Build tridiagonal
    let actual_m = alpha.iter().position(|&a| a == 0.0).unwrap_or(m).min(m);
    let mut tri = vec![0.0f64; actual_m * actual_m];
    for i in 0..actual_m {
        tri[i * actual_m + i] = alpha[i];
        if i > 0 {
            tri[i * actual_m + (i - 1)] = beta[i - 1];
            tri[(i - 1) * actual_m + i] = beta[i - 1];
        }
    }

    // Solve via faer
    use faer::diag::Diag;
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::linalg::evd::{ComputeEigenvectors, self_adjoint_evd, self_adjoint_evd_scratch};
    use faer::{Mat, Par};
    let par = Par::Seq;
    let params = Default::default();
    let mat = Mat::from_fn(actual_m, actual_m, |i, j| tri[i * actual_m + j]);
    let mut s = Diag::<f64>::zeros(actual_m);
    let scratch = self_adjoint_evd_scratch::<f64>(actual_m, ComputeEigenvectors::No, par, params);
    let mut mem = MemBuffer::new(scratch);
    let stack = MemStack::new(&mut mem);
    let _ = self_adjoint_evd(mat.as_ref(), s.as_mut(), None, par, stack, params);

    let lowest = s[0];

    Ok(lowest)
}

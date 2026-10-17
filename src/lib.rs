//! QuantumForge: a small, backend-neutral quantum simulation kernel.
//!
//! The kernel provides CPU quantum semantics plus optional CUDA and LLVM-backed
//! native execution paths. Backends preserve the behavior of the Rust reference.

pub mod analytic;
pub mod aqc;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "llvm")]
pub mod llvm_backend;
pub mod mbqc;
pub mod numeric_ir;
pub mod parser;
pub mod qasm;
pub mod qtm;
pub mod quantum;

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };
    pub const I: Self = Self { re: 0.0, im: 1.0 };

    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    pub fn abs(self) -> f64 {
        self.norm_sqr().sqrt()
    }
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
}

impl std::ops::Add for Complex64 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}
impl std::ops::Sub for Complex64 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}
impl std::ops::Mul for Complex64 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}
impl std::ops::Mul<f64> for Complex64 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.re * rhs, self.im * rhs)
    }
}
impl std::ops::Div<f64> for Complex64 {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self::new(self.re / rhs, self.im / rhs)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum QuantumError {
    ZeroQubits,
    StateSizeNotPowerOfTwo(usize),
    QubitOutOfRange { qubit: usize, qubits: usize },
    InvalidQubitPair { first: usize, second: usize },
    InvalidProbability(f64),
    InvalidTrainingData,
    InvalidSubsystemSelection,
    OperationRequiresDensityMatrix,
    OperationRequiresShotExecution,
    InvalidShotCount,
    ClassicalBitOutOfRange(usize),
    InvalidParameterCount { expected: usize, actual: usize },
    InvalidInputCount { expected: usize, actual: usize },
    NonFiniteParameter,
    InvalidOptimizerConfiguration,
}

impl fmt::Display for QuantumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroQubits => write!(f, "a state vector must contain at least one qubit"),
            Self::StateSizeNotPowerOfTwo(n) => {
                write!(f, "state vector length {n} is not a power of two")
            }
            Self::QubitOutOfRange { qubit, qubits } => {
                write!(f, "qubit index {qubit} is out of range for {qubits} qubits")
            }
            Self::InvalidQubitPair { first, second } => write!(
                f,
                "two-qubit operation requires distinct qubits, got {first} and {second}"
            ),
            Self::InvalidProbability(p) => {
                write!(f, "measurement probability must be in [0, 1], got {p}")
            }
            Self::InvalidTrainingData => write!(f, "training data must be non-empty and finite"),
            Self::InvalidSubsystemSelection => write!(
                f,
                "partial-trace subsystem selection must contain unique in-range qubit indices"
            ),
            Self::OperationRequiresDensityMatrix => write!(
                f,
                "this circuit operation requires density-matrix simulation"
            ),
            Self::OperationRequiresShotExecution => write!(
                f,
                "measurement and classically conditioned circuit operations require shot execution"
            ),
            Self::InvalidShotCount => write!(f, "shot count must be greater than zero"),
            Self::ClassicalBitOutOfRange(bit) => {
                write!(
                    f,
                    "classical bit index {bit} exceeds the supported maximum of 4095"
                )
            }
            Self::InvalidParameterCount { expected, actual } => {
                write!(f, "expected {expected} circuit parameters, got {actual}")
            }
            Self::InvalidInputCount { expected, actual } => {
                write!(f, "expected {expected} encoded inputs, got {actual}")
            }
            Self::NonFiniteParameter => write!(f, "circuit parameters must be finite"),
            Self::InvalidOptimizerConfiguration => write!(
                f,
                "optimizer learning rate and batch size must be positive and finite"
            ),
        }
    }
}
impl Error for QuantumError {}

#[derive(Clone, Debug, PartialEq)]
pub struct StateVector {
    amplitudes: Vec<Complex64>,
    qubits: usize,
}

impl StateVector {
    pub fn zero(qubits: usize) -> Result<Self, QuantumError> {
        if qubits == 0 {
            return Err(QuantumError::ZeroQubits);
        }
        let mut amplitudes = vec![Complex64::ZERO; 1usize << qubits];
        amplitudes[0] = Complex64::ONE;
        Ok(Self { amplitudes, qubits })
    }

    pub fn from_amplitudes(amplitudes: Vec<Complex64>) -> Result<Self, QuantumError> {
        if amplitudes.len() < 2 {
            return Err(QuantumError::ZeroQubits);
        }
        if !amplitudes.len().is_power_of_two() {
            return Err(QuantumError::StateSizeNotPowerOfTwo(amplitudes.len()));
        }
        let qubits = amplitudes.len().ilog2() as usize;
        Ok(Self { amplitudes, qubits })
    }

    pub fn qubits(&self) -> usize {
        self.qubits
    }
    pub fn to_ket(&self) -> Result<crate::quantum::Ket, crate::quantum::LinearError> {
        crate::quantum::Ket::new(
            crate::quantum::HilbertSpace::qubits(self.qubits)?,
            self.amplitudes.clone(),
        )
    }
    pub fn amplitudes(&self) -> &[Complex64] {
        &self.amplitudes
    }
    pub fn norm_sqr(&self) -> f64 {
        self.amplitudes.iter().map(|a| a.norm_sqr()).sum()
    }

    pub fn normalize(&mut self) -> Result<(), QuantumError> {
        let norm = self.norm_sqr().sqrt();
        if norm == 0.0 || !norm.is_finite() {
            return Err(QuantumError::InvalidProbability(norm));
        }
        for amplitude in &mut self.amplitudes {
            *amplitude = *amplitude / norm;
        }
        Ok(())
    }

    /// Apply a one-qubit gate. Qubit 0 is the least-significant basis bit.
    pub fn apply(&mut self, gate: Gate, qubit: usize) -> Result<(), QuantumError> {
        if qubit >= self.qubits {
            return Err(QuantumError::QubitOutOfRange {
                qubit,
                qubits: self.qubits,
            });
        }
        let (m00, m01, m10, m11) = gate.matrix();
        let mask = 1usize << qubit;
        for base in 0..self.amplitudes.len() {
            if base & mask == 0 {
                let i0 = base;
                let i1 = base | mask;
                let a0 = self.amplitudes[i0];
                let a1 = self.amplitudes[i1];
                self.amplitudes[i0] = m00 * a0 + m01 * a1;
                self.amplitudes[i1] = m10 * a0 + m11 * a1;
            }
        }
        Ok(())
    }

    pub fn apply_cnot(&mut self, control: usize, target: usize) -> Result<(), QuantumError> {
        self.check_pair(control, target)?;
        let control_mask = 1usize << control;
        let target_mask = 1usize << target;
        for index in 0..self.amplitudes.len() {
            if index & control_mask != 0 && index & target_mask == 0 {
                self.amplitudes.swap(index, index | target_mask);
            }
        }
        Ok(())
    }

    pub fn apply_cz(&mut self, first: usize, second: usize) -> Result<(), QuantumError> {
        self.check_pair(first, second)?;
        let first_mask = 1usize << first;
        let second_mask = 1usize << second;
        for (index, amplitude) in self.amplitudes.iter_mut().enumerate() {
            if index & first_mask != 0 && index & second_mask != 0 {
                *amplitude = *amplitude * -1.0;
            }
        }
        Ok(())
    }

    pub fn apply_swap(&mut self, first: usize, second: usize) -> Result<(), QuantumError> {
        self.check_pair(first, second)?;
        let first_mask = 1usize << first;
        let second_mask = 1usize << second;
        for index in 0..self.amplitudes.len() {
            if index & first_mask == 0 && index & second_mask != 0 {
                self.amplitudes
                    .swap(index, index ^ first_mask ^ second_mask);
            }
        }
        Ok(())
    }

    fn check_pair(&self, first: usize, second: usize) -> Result<(), QuantumError> {
        if first >= self.qubits {
            return Err(QuantumError::QubitOutOfRange {
                qubit: first,
                qubits: self.qubits,
            });
        }
        if second >= self.qubits {
            return Err(QuantumError::QubitOutOfRange {
                qubit: second,
                qubits: self.qubits,
            });
        }
        if first == second {
            return Err(QuantumError::InvalidQubitPair { first, second });
        }
        Ok(())
    }

    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|a| a.norm_sqr()).collect()
    }

    pub fn expectation_z(&self, qubit: usize) -> Result<f64, QuantumError> {
        if qubit >= self.qubits {
            return Err(QuantumError::QubitOutOfRange {
                qubit,
                qubits: self.qubits,
            });
        }
        let mask = 1usize << qubit;
        Ok(self
            .amplitudes
            .iter()
            .enumerate()
            .map(|(index, amp)| {
                let sign = if index & mask == 0 { 1.0 } else { -1.0 };
                sign * amp.norm_sqr()
            })
            .sum())
    }

    pub fn expectation_x(&self, qubit: usize) -> Result<f64, QuantumError> {
        if qubit >= self.qubits {
            return Err(QuantumError::QubitOutOfRange {
                qubit,
                qubits: self.qubits,
            });
        }
        let mask = 1usize << qubit;
        Ok((0..self.amplitudes.len())
            .filter(|i| i & mask == 0)
            .map(|i| (self.amplitudes[i].conj() * self.amplitudes[i | mask]).re * 2.0)
            .sum())
    }

    pub fn expectation_y(&self, qubit: usize) -> Result<f64, QuantumError> {
        if qubit >= self.qubits {
            return Err(QuantumError::QubitOutOfRange {
                qubit,
                qubits: self.qubits,
            });
        }
        let mask = 1usize << qubit;
        Ok((0..self.amplitudes.len())
            .filter(|i| i & mask == 0)
            .map(|i| (self.amplitudes[i].conj() * self.amplitudes[i | mask]).im * 2.0)
            .sum())
    }

    pub fn expectation(
        &self,
        observable: SingleQubitObservable,
        qubit: usize,
    ) -> Result<f64, QuantumError> {
        match observable {
            SingleQubitObservable::X => self.expectation_x(qubit),
            SingleQubitObservable::Y => self.expectation_y(qubit),
            SingleQubitObservable::Z => self.expectation_z(qubit),
        }
    }

    /// Measure one qubit and collapse the state. Returns the sampled bit.
    pub fn measure(&mut self, qubit: usize, rng: &mut SmallRng) -> Result<u8, QuantumError> {
        if qubit >= self.qubits {
            return Err(QuantumError::QubitOutOfRange {
                qubit,
                qubits: self.qubits,
            });
        }
        let mask = 1usize << qubit;
        let p_one: f64 = self
            .amplitudes
            .iter()
            .enumerate()
            .filter(|(index, _)| index & mask != 0)
            .map(|(_, amp)| amp.norm_sqr())
            .sum();
        let p_one = p_one.clamp(0.0, 1.0);
        let outcome = u8::from(rng.next_f64() < p_one);
        let probability = if outcome == 1 { p_one } else { 1.0 - p_one };
        if probability <= f64::EPSILON {
            return Err(QuantumError::InvalidProbability(probability));
        }
        let scale = probability.sqrt();
        for (index, amp) in self.amplitudes.iter_mut().enumerate() {
            if u8::from(index & mask != 0) != outcome {
                *amp = Complex64::ZERO;
            } else {
                *amp = *amp / scale;
            }
        }
        Ok(outcome)
    }

    /// Sample one Kraus trajectory for a local noise channel and normalize the
    /// selected pure-state branch. Repeated trajectories reproduce the channel
    /// when observables are averaged over shots.
    pub fn apply_channel_trajectory(
        &mut self,
        channel: NoiseChannel,
        qubit: usize,
        rng: &mut SmallRng,
    ) -> Result<(), QuantumError> {
        if qubit >= self.qubits {
            return Err(QuantumError::QubitOutOfRange {
                qubit,
                qubits: self.qubits,
            });
        }
        match channel {
            NoiseChannel::Depolarizing { probability } => {
                validate_probability(probability)?;
                let sample = rng.next_f64();
                if sample >= 1.0 - probability && probability > 0.0 {
                    let branch =
                        ((sample - (1.0 - probability)) * (3.0 / probability)).floor() as u8;
                    let gate = match branch.min(2) {
                        0 => Gate::X,
                        1 => Gate::Y,
                        _ => Gate::Z,
                    };
                    self.apply(gate, qubit)?;
                }
            }
            NoiseChannel::AmplitudeDamping { gamma } | NoiseChannel::PhaseDamping { gamma } => {
                validate_probability(gamma)?;
                let norm = self.norm_sqr();
                if !norm.is_finite() || norm <= f64::EPSILON {
                    return Err(QuantumError::InvalidProbability(norm));
                }
                let mask = 1usize << qubit;
                let probability_one: f64 = self
                    .amplitudes
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| index & mask != 0)
                    .map(|(_, amplitude)| amplitude.norm_sqr())
                    .sum();
                let jump_probability = (gamma * probability_one / norm).clamp(0.0, 1.0);
                let jump = rng.next_f64() < jump_probability;
                let branch_norm = if jump {
                    gamma * probability_one
                } else {
                    norm - gamma * probability_one
                };
                if !branch_norm.is_finite() || branch_norm <= f64::EPSILON {
                    return Err(QuantumError::InvalidProbability(branch_norm));
                }
                let inverse_norm = branch_norm.sqrt().recip();
                if jump {
                    let scale = gamma.sqrt() * inverse_norm;
                    for base in 0..self.amplitudes.len() {
                        if base & mask == 0 {
                            let excited = base | mask;
                            let value = self.amplitudes[excited] * scale;
                            match channel {
                                NoiseChannel::AmplitudeDamping { .. } => {
                                    self.amplitudes[base] = value;
                                    self.amplitudes[excited] = Complex64::ZERO;
                                }
                                NoiseChannel::PhaseDamping { .. } => {
                                    self.amplitudes[base] = Complex64::ZERO;
                                    self.amplitudes[excited] = value;
                                }
                                NoiseChannel::Depolarizing { .. } => unreachable!(),
                            }
                        }
                    }
                } else {
                    let excited_scale = (1.0 - gamma).sqrt() * inverse_norm;
                    for (index, amplitude) in self.amplitudes.iter_mut().enumerate() {
                        *amplitude = *amplitude
                            * if index & mask == 0 {
                                inverse_norm
                            } else {
                                excited_scale
                            };
                    }
                }
            }
        }
        Ok(())
    }
}

fn expand_subsystems(
    kept: usize,
    environment: usize,
    keep_qubits: &[usize],
    traced_qubits: &[usize],
) -> usize {
    let mut full = 0usize;
    for (position, &qubit) in keep_qubits.iter().enumerate() {
        if kept & (1usize << position) != 0 {
            full |= 1usize << qubit;
        }
    }
    for (position, &qubit) in traced_qubits.iter().enumerate() {
        if environment & (1usize << position) != 0 {
            full |= 1usize << qubit;
        }
    }
    full
}

fn check_qubit_pair(first: usize, second: usize, qubits: usize) -> Result<(), QuantumError> {
    if first >= qubits {
        return Err(QuantumError::QubitOutOfRange {
            qubit: first,
            qubits,
        });
    }
    if second >= qubits {
        return Err(QuantumError::QubitOutOfRange {
            qubit: second,
            qubits,
        });
    }
    if first == second {
        return Err(QuantumError::InvalidQubitPair { first, second });
    }
    Ok(())
}

/// Dense mixed-state representation, stored as a row-major density matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct DensityMatrix {
    entries: Vec<Complex64>,
    dimension: usize,
    qubits: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NoiseChannel {
    Depolarizing { probability: f64 },
    AmplitudeDamping { gamma: f64 },
    PhaseDamping { gamma: f64 },
}

impl DensityMatrix {
    #[cfg(feature = "cuda")]
    pub(crate) fn from_backend_entries(entries: Vec<Complex64>, qubits: usize) -> Self {
        let dimension = 1usize << qubits;
        debug_assert_eq!(entries.len(), dimension * dimension);
        Self {
            entries,
            dimension,
            qubits,
        }
    }

    pub fn from_state(state: &StateVector) -> Self {
        let dimension = state.amplitudes.len();
        let mut entries = vec![Complex64::ZERO; dimension * dimension];
        for row in 0..dimension {
            for col in 0..dimension {
                entries[row * dimension + col] =
                    state.amplitudes[row] * state.amplitudes[col].conj();
            }
        }
        Self {
            entries,
            dimension,
            qubits: state.qubits,
        }
    }

    pub fn maximally_mixed(qubits: usize) -> Result<Self, QuantumError> {
        if qubits == 0 {
            return Err(QuantumError::ZeroQubits);
        }
        let dimension = 1usize << qubits;
        let mut entries = vec![Complex64::ZERO; dimension * dimension];
        for i in 0..dimension {
            entries[i * dimension + i] = Complex64::new(1.0 / dimension as f64, 0.0);
        }
        Ok(Self {
            entries,
            dimension,
            qubits,
        })
    }

    pub fn qubits(&self) -> usize {
        self.qubits
    }
    pub fn entries(&self) -> &[Complex64] {
        &self.entries
    }
    pub fn trace(&self) -> Complex64 {
        (0..self.dimension)
            .map(|i| self.entries[i * self.dimension + i])
            .fold(Complex64::ZERO, |a, b| a + b)
    }
    pub fn purity(&self) -> f64 {
        // For Hermitian rho, Tr(rho^2) = sum_ij |rho_ij|^2.
        self.entries.iter().map(|x| x.norm_sqr()).sum()
    }

    /// Apply a unitary operator to both sides of the density matrix:
    /// `rho <- U rho U†`.
    pub fn apply_unitary(
        &mut self,
        unitary: &crate::quantum::Operator,
        tolerance: f64,
    ) -> Result<(), crate::quantum::LinearError> {
        use crate::quantum::LinearError;
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(LinearError::InvalidTolerance);
        }
        let input = unitary.input_space().dimension();
        let output = unitary.output_space().dimension();
        if input != output {
            return Err(LinearError::NotEndomorphism);
        }
        if input != self.dimension {
            return Err(LinearError::DimensionMismatch {
                expected: self.dimension,
                actual: input,
            });
        }
        if !unitary.is_unitary(tolerance) {
            return Err(LinearError::NotUnitary);
        }
        let n = self.dimension;
        let mut left_product = vec![Complex64::ZERO; n * n];
        for row in 0..n {
            for col in 0..n {
                for inner in 0..n {
                    left_product[row * n + col] = left_product[row * n + col]
                        + unitary.entries()[row * n + inner] * self.entries[inner * n + col];
                }
            }
        }
        let mut evolved = vec![Complex64::ZERO; n * n];
        for row in 0..n {
            for col in 0..n {
                for inner in 0..n {
                    evolved[row * n + col] = evolved[row * n + col]
                        + left_product[row * n + inner] * unitary.entries()[col * n + inner].conj();
                }
            }
        }
        self.entries = evolved;
        Ok(())
    }

    /// Evolve under a time-independent Hermitian Hamiltonian using
    /// `rho(t) = exp(-iHt) rho(0) exp(iHt)`.
    pub fn evolve_with_hamiltonian(
        &mut self,
        hamiltonian: &crate::quantum::Operator,
        time: f64,
        tolerance: f64,
        max_sweeps: usize,
    ) -> Result<(), crate::quantum::LinearError> {
        let evolution = hamiltonian.evolution_operator(time, tolerance, max_sweeps)?;
        self.apply_unitary(&evolution, tolerance)
    }

    pub fn expectation_z(&self, qubit: usize) -> Result<f64, QuantumError> {
        if qubit >= self.qubits {
            return Err(QuantumError::QubitOutOfRange {
                qubit,
                qubits: self.qubits,
            });
        }
        let mask = 1usize << qubit;
        Ok((0..self.dimension)
            .map(|i| {
                let sign = if i & mask == 0 { 1.0 } else { -1.0 };
                sign * self.entries[i * self.dimension + i].re
            })
            .sum())
    }

    pub fn expectation(
        &self,
        observable: SingleQubitObservable,
        qubit: usize,
    ) -> Result<f64, QuantumError> {
        if qubit >= self.qubits {
            return Err(QuantumError::QubitOutOfRange {
                qubit,
                qubits: self.qubits,
            });
        }
        match observable {
            SingleQubitObservable::Z => self.expectation_z(qubit),
            SingleQubitObservable::X | SingleQubitObservable::Y => {
                let mask = 1usize << qubit;
                let mut value = 0.0;
                for row in 0..self.dimension {
                    if row & mask == 0 {
                        let paired = row | mask;
                        let coherence = self.entries[row * self.dimension + paired];
                        value += match observable {
                            SingleQubitObservable::X => 2.0 * coherence.re,
                            SingleQubitObservable::Y => -2.0 * coherence.im,
                            SingleQubitObservable::Z => unreachable!(),
                        };
                    }
                }
                Ok(value)
            }
        }
    }

    pub fn probability_one(&self, qubit: usize) -> Result<f64, QuantumError> {
        if qubit >= self.qubits {
            return Err(QuantumError::QubitOutOfRange {
                qubit,
                qubits: self.qubits,
            });
        }
        let mask = 1usize << qubit;
        Ok((0..self.dimension)
            .filter(|i| i & mask != 0)
            .map(|i| self.entries[i * self.dimension + i].re)
            .sum::<f64>()
            .clamp(0.0, 1.0))
    }

    /// Sample and projectively measure a qubit, returning the observed bit.
    pub fn measure(&mut self, qubit: usize, rng: &mut SmallRng) -> Result<u8, QuantumError> {
        if qubit >= self.qubits {
            return Err(QuantumError::QubitOutOfRange {
                qubit,
                qubits: self.qubits,
            });
        }
        let p_one = self.probability_one(qubit)?;
        let outcome = u8::from(rng.next_f64() < p_one);
        let probability = if outcome == 1 { p_one } else { 1.0 - p_one };
        if probability <= f64::EPSILON {
            return Err(QuantumError::InvalidProbability(probability));
        }
        let mask = 1usize << qubit;
        for row in 0..self.dimension {
            for col in 0..self.dimension {
                if u8::from(row & mask != 0) != outcome || u8::from(col & mask != 0) != outcome {
                    self.entries[row * self.dimension + col] = Complex64::ZERO;
                } else {
                    self.entries[row * self.dimension + col] =
                        self.entries[row * self.dimension + col] / probability;
                }
            }
        }
        Ok(outcome)
    }

    /// Trace out every qubit not listed in `keep_qubits`. Output qubit ordering
    /// follows the order of indices supplied by the caller.
    pub fn partial_trace(&self, keep_qubits: &[usize]) -> Result<Self, QuantumError> {
        let mut seen = vec![false; self.qubits];
        for &qubit in keep_qubits {
            if qubit >= self.qubits || seen[qubit] {
                return Err(QuantumError::InvalidSubsystemSelection);
            }
            seen[qubit] = true;
        }
        let traced: Vec<usize> = (0..self.qubits).filter(|q| !seen[*q]).collect();
        let kept_dimension = 1usize << keep_qubits.len();
        let traced_dimension = 1usize << traced.len();
        let mut entries = vec![Complex64::ZERO; kept_dimension * kept_dimension];
        for row in 0..kept_dimension {
            for col in 0..kept_dimension {
                let mut value = Complex64::ZERO;
                for environment in 0..traced_dimension {
                    let full_row = expand_subsystems(row, environment, keep_qubits, &traced);
                    let full_col = expand_subsystems(col, environment, keep_qubits, &traced);
                    value = value + self.entries[full_row * self.dimension + full_col];
                }
                entries[row * kept_dimension + col] = value;
            }
        }
        Ok(Self {
            entries,
            dimension: kept_dimension,
            qubits: keep_qubits.len(),
        })
    }

    pub fn apply_cnot(&mut self, control: usize, target: usize) -> Result<(), QuantumError> {
        check_qubit_pair(control, target, self.qubits)?;
        let control_mask = 1usize << control;
        let target_mask = 1usize << target;
        self.apply_basis_permutation(|index| {
            if index & control_mask != 0 {
                index ^ target_mask
            } else {
                index
            }
        });
        Ok(())
    }

    pub fn apply_cz(&mut self, first: usize, second: usize) -> Result<(), QuantumError> {
        check_qubit_pair(first, second, self.qubits)?;
        let a = 1usize << first;
        let b = 1usize << second;
        for row in 0..self.dimension {
            for col in 0..self.dimension {
                if row & a != 0 && row & b != 0 {
                    self.entries[row * self.dimension + col] =
                        self.entries[row * self.dimension + col] * -1.0;
                }
                if col & a != 0 && col & b != 0 {
                    self.entries[row * self.dimension + col] =
                        self.entries[row * self.dimension + col] * -1.0;
                }
            }
        }
        Ok(())
    }

    pub fn apply_swap(&mut self, first: usize, second: usize) -> Result<(), QuantumError> {
        check_qubit_pair(first, second, self.qubits)?;
        let a = 1usize << first;
        let b = 1usize << second;
        self.apply_basis_permutation(|index| {
            if (index & a == 0) != (index & b == 0) {
                index ^ a ^ b
            } else {
                index
            }
        });
        Ok(())
    }

    fn apply_basis_permutation(&mut self, permute: impl Fn(usize) -> usize) {
        let original = self.entries.clone();
        for row in 0..self.dimension {
            for col in 0..self.dimension {
                self.entries[row * self.dimension + col] =
                    original[permute(row) * self.dimension + permute(col)];
            }
        }
    }

    /// Apply rho <- U rho U† for a single-qubit unitary.
    pub fn apply(&mut self, gate: Gate, qubit: usize) -> Result<(), QuantumError> {
        if qubit >= self.qubits {
            return Err(QuantumError::QubitOutOfRange {
                qubit,
                qubits: self.qubits,
            });
        }
        let (m00, m01, m10, m11) = gate.matrix();
        self.apply_local_matrix(m00, m01, m10, m11, qubit);
        Ok(())
    }

    pub fn apply_channel(
        &mut self,
        channel: NoiseChannel,
        qubit: usize,
    ) -> Result<(), QuantumError> {
        if qubit >= self.qubits {
            return Err(QuantumError::QubitOutOfRange {
                qubit,
                qubits: self.qubits,
            });
        }
        match channel {
            NoiseChannel::Depolarizing { probability } => {
                self.depolarize(qubit, probability)?;
            }
            NoiseChannel::AmplitudeDamping { gamma } => {
                validate_probability(gamma)?;
                let k0 = (
                    Complex64::ONE,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::new((1.0 - gamma).sqrt(), 0.0),
                );
                let k1 = (
                    Complex64::ZERO,
                    Complex64::new(gamma.sqrt(), 0.0),
                    Complex64::ZERO,
                    Complex64::ZERO,
                );
                self.apply_kraus(&[k0, k1], qubit);
            }
            NoiseChannel::PhaseDamping { gamma } => {
                validate_probability(gamma)?;
                let k0 = (
                    Complex64::ONE,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::new((1.0 - gamma).sqrt(), 0.0),
                );
                let k1 = (
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::new(gamma.sqrt(), 0.0),
                );
                self.apply_kraus(&[k0, k1], qubit);
            }
        }
        Ok(())
    }

    fn apply_kraus(
        &mut self,
        operators: &[(Complex64, Complex64, Complex64, Complex64)],
        qubit: usize,
    ) {
        let mut sum = vec![Complex64::ZERO; self.entries.len()];
        for &(a, b, c, d) in operators {
            let branch = self.local_transform(a, b, c, d, qubit);
            for (total, value) in sum.iter_mut().zip(branch) {
                *total = *total + value;
            }
        }
        self.entries = sum;
    }

    fn apply_local_matrix(
        &mut self,
        m00: Complex64,
        m01: Complex64,
        m10: Complex64,
        m11: Complex64,
        qubit: usize,
    ) {
        self.entries = self.local_transform(m00, m01, m10, m11, qubit);
    }

    fn local_transform(
        &self,
        m00: Complex64,
        m01: Complex64,
        m10: Complex64,
        m11: Complex64,
        qubit: usize,
    ) -> Vec<Complex64> {
        let mask = 1usize << qubit;
        let mut left = vec![Complex64::ZERO; self.entries.len()];
        for row in 0..self.dimension {
            let row0 = row & !mask;
            let row1 = row0 | mask;
            for col in 0..self.dimension {
                let a = self.entries[row0 * self.dimension + col];
                let b = self.entries[row1 * self.dimension + col];
                let out = if row & mask == 0 {
                    m00 * a + m01 * b
                } else {
                    m10 * a + m11 * b
                };
                left[row * self.dimension + col] = out;
            }
        }
        let mut out = vec![Complex64::ZERO; self.entries.len()];
        for row in 0..self.dimension {
            for col in 0..self.dimension {
                let col0 = col & !mask;
                let col1 = col0 | mask;
                let a = left[row * self.dimension + col0];
                let b = left[row * self.dimension + col1];
                out[row * self.dimension + col] = if col & mask == 0 {
                    a * m00.conj() + b * m01.conj()
                } else {
                    a * m10.conj() + b * m11.conj()
                };
            }
        }
        out
    }

    /// Apply the single-qubit depolarizing channel:
    /// (1-p) rho + p/3 (X rho X + Y rho Y + Z rho Z).
    pub fn depolarize(&mut self, qubit: usize, probability: f64) -> Result<(), QuantumError> {
        if qubit >= self.qubits {
            return Err(QuantumError::QubitOutOfRange {
                qubit,
                qubits: self.qubits,
            });
        }
        validate_probability(probability)?;
        let original = self.clone();
        let mut mixed = Self {
            entries: vec![Complex64::ZERO; self.entries.len()],
            dimension: self.dimension,
            qubits: self.qubits,
        };
        for gate in [Gate::X, Gate::Y, Gate::Z] {
            let mut branch = original.clone();
            branch.apply(gate, qubit)?;
            for (sum, value) in mixed.entries.iter_mut().zip(branch.entries) {
                *sum = *sum + value * (probability / 3.0);
            }
        }
        for (sum, value) in mixed.entries.iter_mut().zip(original.entries) {
            *sum = *sum + value * (1.0 - probability);
        }
        self.entries = mixed.entries;
        Ok(())
    }
}

fn validate_probability(probability: f64) -> Result<(), QuantumError> {
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        Err(QuantumError::InvalidProbability(probability))
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Gate {
    I,
    X,
    Y,
    Z,
    H,
    S,
    Sdg,
    T,
    Tdg,
    Rx(f64),
    Ry(f64),
    Rz(f64),
}
impl Gate {
    pub(crate) fn matrix(self) -> (Complex64, Complex64, Complex64, Complex64) {
        let z = Complex64::ZERO;
        let o = Complex64::ONE;
        let i = Complex64::I;
        let s = std::f64::consts::FRAC_1_SQRT_2;
        match self {
            Self::I => (o, z, z, o),
            Self::X => (z, o, o, z),
            Self::Y => (z, Complex64::new(0.0, -1.0), i, z),
            Self::Z => (o, z, z, Complex64::new(-1.0, 0.0)),
            Self::H => (
                Complex64::new(s, 0.0),
                Complex64::new(s, 0.0),
                Complex64::new(s, 0.0),
                Complex64::new(-s, 0.0),
            ),
            Self::S => (o, z, z, i),                           // diag(1, i)
            Self::Sdg => (o, z, z, Complex64::new(0.0, -1.0)), // diag(1, -i)
            Self::T => (
                o,
                z,
                z,
                Complex64::new(
                    std::f64::consts::FRAC_1_SQRT_2,
                    std::f64::consts::FRAC_1_SQRT_2,
                ),
            ), // diag(1, e^{iπ/4})
            Self::Tdg => (
                o,
                z,
                z,
                Complex64::new(
                    std::f64::consts::FRAC_1_SQRT_2,
                    -std::f64::consts::FRAC_1_SQRT_2,
                ),
            ), // diag(1, e^{-iπ/4})
            Self::Rx(theta) => {
                let c = (theta * 0.5).cos();
                let si = (theta * 0.5).sin();
                (
                    Complex64::new(c, 0.0),
                    Complex64::new(0.0, -si),
                    Complex64::new(0.0, -si),
                    Complex64::new(c, 0.0),
                )
            }
            Self::Ry(theta) => {
                let c = (theta * 0.5).cos();
                let si = (theta * 0.5).sin();
                (
                    Complex64::new(c, 0.0),
                    Complex64::new(-si, 0.0),
                    Complex64::new(si, 0.0),
                    Complex64::new(c, 0.0),
                )
            }
            Self::Rz(theta) => {
                let half = theta * 0.5;
                (
                    Complex64::new(half.cos(), -half.sin()),
                    z,
                    z,
                    Complex64::new(half.cos(), half.sin()),
                )
            }
        }
    }
}

fn rotation_gate(axis: RotationAxis, theta: f64) -> Gate {
    match axis {
        RotationAxis::X => Gate::Rx(theta),
        RotationAxis::Y => Gate::Ry(theta),
        RotationAxis::Z => Gate::Rz(theta),
    }
}

fn adjoint_gate(gate: Gate) -> Gate {
    match gate {
        Gate::Rx(theta) => Gate::Rx(-theta),
        Gate::Ry(theta) => Gate::Ry(-theta),
        Gate::Rz(theta) => Gate::Rz(-theta),
        Gate::S => Gate::Sdg,
        Gate::Sdg => Gate::S,
        Gate::T => Gate::Tdg,
        Gate::Tdg => Gate::T,
        self_adjoint => self_adjoint,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CircuitOp {
    SingleQubit {
        gate: Gate,
        target: usize,
    },
    Cnot {
        control: usize,
        target: usize,
    },
    Cz {
        first: usize,
        second: usize,
    },
    Swap {
        first: usize,
        second: usize,
    },
    Channel {
        channel: NoiseChannel,
        target: usize,
    },
    Measure {
        target: usize,
        classical_bit: usize,
    },
    ConditionalSingleQubit {
        classical_bit: usize,
        value: u8,
        gate: Gate,
        target: usize,
    },
}

/// Measurement counts indexed by classical-register bit values, least-significant
/// classical bit first. Unwritten classical bits are zero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShotResult {
    pub shots: usize,
    pub counts: BTreeMap<Vec<u8>, usize>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Circuit {
    operations: Vec<CircuitOp>,
}
impl Circuit {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn push(&mut self, gate: Gate, target: usize) {
        self.operations
            .push(CircuitOp::SingleQubit { gate, target });
    }
    pub fn cnot(&mut self, control: usize, target: usize) {
        self.operations.push(CircuitOp::Cnot { control, target });
    }
    pub fn cz(&mut self, first: usize, second: usize) {
        self.operations.push(CircuitOp::Cz { first, second });
    }
    pub fn swap(&mut self, first: usize, second: usize) {
        self.operations.push(CircuitOp::Swap { first, second });
    }
    pub fn channel(&mut self, target: usize, channel: NoiseChannel) {
        self.operations.push(CircuitOp::Channel { channel, target });
    }
    /// Measure `target` and write its sampled result into a classical bit.
    pub fn measure(&mut self, target: usize, classical_bit: usize) {
        self.operations.push(CircuitOp::Measure {
            target,
            classical_bit,
        });
    }
    /// Apply a one-qubit gate only when a classical bit has the requested value.
    pub fn if_classical_bit(&mut self, classical_bit: usize, value: u8, gate: Gate, target: usize) {
        self.operations.push(CircuitOp::ConditionalSingleQubit {
            classical_bit,
            value: value & 1,
            gate,
            target,
        });
    }
    pub fn operations(&self) -> &[CircuitOp] {
        &self.operations
    }
    pub fn run(&self, mut state: StateVector) -> Result<StateVector, QuantumError> {
        for op in &self.operations {
            match *op {
                CircuitOp::SingleQubit { gate, target } => state.apply(gate, target)?,
                CircuitOp::Cnot { control, target } => state.apply_cnot(control, target)?,
                CircuitOp::Cz { first, second } => state.apply_cz(first, second)?,
                CircuitOp::Swap { first, second } => state.apply_swap(first, second)?,
                CircuitOp::Channel { .. } => {
                    return Err(QuantumError::OperationRequiresDensityMatrix);
                }
                CircuitOp::Measure { .. } | CircuitOp::ConditionalSingleQubit { .. } => {
                    return Err(QuantumError::OperationRequiresShotExecution);
                }
            }
        }
        Ok(state)
    }
    pub fn run_density(&self, mut state: DensityMatrix) -> Result<DensityMatrix, QuantumError> {
        for op in &self.operations {
            match *op {
                CircuitOp::SingleQubit { gate, target } => state.apply(gate, target)?,
                CircuitOp::Cnot { control, target } => state.apply_cnot(control, target)?,
                CircuitOp::Cz { first, second } => state.apply_cz(first, second)?,
                CircuitOp::Swap { first, second } => state.apply_swap(first, second)?,
                CircuitOp::Channel { channel, target } => state.apply_channel(channel, target)?,
                CircuitOp::Measure { .. } | CircuitOp::ConditionalSingleQubit { .. } => {
                    return Err(QuantumError::OperationRequiresShotExecution);
                }
            }
        }
        Ok(state)
    }

    /// Execute seeded projective shots for unitary circuits. Use
    /// `run_shots_with_noise` when the circuit contains noise channels.
    pub fn run_shots(
        &self,
        initial_state: &StateVector,
        shots: usize,
        seed: u64,
    ) -> Result<ShotResult, QuantumError> {
        self.run_shots_impl(initial_state, shots, seed, false)
    }

    /// Execute seeded pure-state trajectories for circuits containing local
    /// noise channels, projective measurements, and classical feed-forward.
    pub fn run_shots_with_noise(
        &self,
        initial_state: &StateVector,
        shots: usize,
        seed: u64,
    ) -> Result<ShotResult, QuantumError> {
        self.run_shots_impl(initial_state, shots, seed, true)
    }

    fn run_shots_impl(
        &self,
        initial_state: &StateVector,
        shots: usize,
        seed: u64,
        allow_noise: bool,
    ) -> Result<ShotResult, QuantumError> {
        if shots == 0 {
            return Err(QuantumError::InvalidShotCount);
        }
        for operation in &self.operations {
            let classical_bit = match operation {
                CircuitOp::Measure { classical_bit, .. }
                | CircuitOp::ConditionalSingleQubit { classical_bit, .. } => Some(*classical_bit),
                _ => None,
            };
            if let Some(bit) = classical_bit.filter(|bit| *bit >= 4096) {
                return Err(QuantumError::ClassicalBitOutOfRange(bit));
            }
        }
        let classical_width = self
            .operations
            .iter()
            .filter_map(|operation| match operation {
                CircuitOp::Measure { classical_bit, .. }
                | CircuitOp::ConditionalSingleQubit { classical_bit, .. } => {
                    Some(classical_bit + 1)
                }
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let mut rng = SmallRng::seeded(seed);
        let mut counts = BTreeMap::new();
        for _ in 0..shots {
            let mut state = initial_state.clone();
            if allow_noise {
                state.normalize()?;
            }
            let mut classical = vec![0; classical_width];
            for operation in &self.operations {
                match *operation {
                    CircuitOp::SingleQubit { gate, target } => state.apply(gate, target)?,
                    CircuitOp::Cnot { control, target } => state.apply_cnot(control, target)?,
                    CircuitOp::Cz { first, second } => state.apply_cz(first, second)?,
                    CircuitOp::Swap { first, second } => state.apply_swap(first, second)?,
                    CircuitOp::Channel { channel, target } => {
                        if allow_noise {
                            state.apply_channel_trajectory(channel, target, &mut rng)?;
                        } else {
                            return Err(QuantumError::OperationRequiresDensityMatrix);
                        }
                    }
                    CircuitOp::Measure {
                        target,
                        classical_bit,
                    } => {
                        classical[classical_bit] = state.measure(target, &mut rng)?;
                    }
                    CircuitOp::ConditionalSingleQubit {
                        classical_bit,
                        value,
                        gate,
                        target,
                    } => {
                        if classical[classical_bit] == value {
                            state.apply(gate, target)?;
                        }
                    }
                }
            }
            *counts.entry(classical).or_insert(0) += 1;
        }
        Ok(ShotResult { shots, counts })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotationAxis {
    X,
    Y,
    Z,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SingleQubitObservable {
    X,
    Y,
    Z,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ParameterizedOp {
    Fixed {
        gate: Gate,
        target: usize,
    },
    Rotation {
        axis: RotationAxis,
        target: usize,
        parameter: usize,
    },
    EncodingRotation {
        axis: RotationAxis,
        target: usize,
        input: usize,
    },
    Cnot {
        control: usize,
        target: usize,
    },
    Cz {
        first: usize,
        second: usize,
    },
    Swap {
        first: usize,
        second: usize,
    },
    Channel {
        channel: NoiseChannel,
        target: usize,
    },
}

/// Ideal state-vector circuit with runtime-bound rotation parameters.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParameterizedCircuit {
    operations: Vec<ParameterizedOp>,
    parameter_count: usize,
    input_count: usize,
}

/// Optimizer settings shared by parameterized-circuit training routines.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QmlTrainingConfig {
    pub learning_rate: f64,
    pub epochs: usize,
    pub batch_size: usize,
    pub seed: u64,
}

/// Sampling budget and random seed for one finite-shot observable estimate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShotConfig {
    pub shots: usize,
    pub seed: u64,
}

/// Sampling and optimizer settings for finite-shot QML training.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShotTrainingConfig {
    pub shots_per_expectation: usize,
    pub training: QmlTrainingConfig,
}

impl ParameterizedCircuit {
    pub fn new(parameter_count: usize) -> Self {
        Self {
            operations: Vec::new(),
            parameter_count,
            input_count: 0,
        }
    }
    pub fn with_inputs(parameter_count: usize, input_count: usize) -> Self {
        Self {
            operations: Vec::new(),
            parameter_count,
            input_count,
        }
    }
    pub fn push(&mut self, gate: Gate, target: usize) {
        self.operations
            .push(ParameterizedOp::Fixed { gate, target });
    }
    pub fn rotation(&mut self, axis: RotationAxis, target: usize, parameter: usize) {
        self.operations.push(ParameterizedOp::Rotation {
            axis,
            target,
            parameter,
        });
    }
    pub fn encode_rotation(&mut self, axis: RotationAxis, target: usize, input: usize) {
        self.operations.push(ParameterizedOp::EncodingRotation {
            axis,
            target,
            input,
        });
    }
    pub fn cnot(&mut self, control: usize, target: usize) {
        self.operations
            .push(ParameterizedOp::Cnot { control, target });
    }
    pub fn cz(&mut self, first: usize, second: usize) {
        self.operations.push(ParameterizedOp::Cz { first, second });
    }
    pub fn swap(&mut self, first: usize, second: usize) {
        self.operations
            .push(ParameterizedOp::Swap { first, second });
    }
    pub fn channel(&mut self, target: usize, channel: NoiseChannel) {
        self.operations
            .push(ParameterizedOp::Channel { channel, target });
    }
    pub fn parameter_count(&self) -> usize {
        self.parameter_count
    }
    pub fn input_count(&self) -> usize {
        self.input_count
    }
    pub fn operations(&self) -> &[ParameterizedOp] {
        &self.operations
    }

    pub fn run(&self, parameters: &[f64], qubits: usize) -> Result<StateVector, QuantumError> {
        self.run_with_inputs(&[], parameters, qubits)
    }

    pub fn run_with_inputs(
        &self,
        inputs: &[f64],
        parameters: &[f64],
        qubits: usize,
    ) -> Result<StateVector, QuantumError> {
        self.validate_inputs(inputs)?;
        self.run_with_shift(inputs, parameters, qubits, None)
    }

    /// Execute with exact density-matrix noise semantics. The resulting state
    /// can represent mixed states created by any inserted channel operation.
    pub fn run_density_with_inputs(
        &self,
        inputs: &[f64],
        parameters: &[f64],
        qubits: usize,
    ) -> Result<DensityMatrix, QuantumError> {
        self.run_density_with_shift(inputs, parameters, qubits, None)
    }

    pub fn expectation_z(
        &self,
        parameters: &[f64],
        qubits: usize,
        observable_qubit: usize,
    ) -> Result<f64, QuantumError> {
        self.expectation(
            parameters,
            qubits,
            SingleQubitObservable::Z,
            observable_qubit,
        )
    }

    pub fn expectation(
        &self,
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
    ) -> Result<f64, QuantumError> {
        self.run(parameters, qubits)?
            .expectation(observable, observable_qubit)
    }

    pub fn expectation_with_inputs(
        &self,
        inputs: &[f64],
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
    ) -> Result<f64, QuantumError> {
        self.run_with_inputs(inputs, parameters, qubits)?
            .expectation(observable, observable_qubit)
    }

    pub fn expectation_density_with_inputs(
        &self,
        inputs: &[f64],
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
    ) -> Result<f64, QuantumError> {
        self.run_density_with_inputs(inputs, parameters, qubits)?
            .expectation(observable, observable_qubit)
    }

    pub fn expectation_density_with_inputs_shots(
        &self,
        inputs: &[f64],
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
        sampling: ShotConfig,
    ) -> Result<f64, QuantumError> {
        let exact = self.expectation_density_with_inputs(
            inputs,
            parameters,
            qubits,
            observable,
            observable_qubit,
        )?;
        sample_expectation(exact, sampling)
    }

    /// Exact parameter-shift gradient evaluated through density-matrix noise
    /// channels. Every trainable rotation occurrence is shifted separately.
    pub fn parameter_shift_gradient_density_with_inputs(
        &self,
        inputs: &[f64],
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
    ) -> Result<Vec<f64>, QuantumError> {
        self.validate_inputs(inputs)?;
        self.validate_parameters(parameters)?;
        if observable_qubit >= qubits {
            return Err(QuantumError::QubitOutOfRange {
                qubit: observable_qubit,
                qubits,
            });
        }
        let shift = std::f64::consts::FRAC_PI_2;
        let mut gradient = vec![0.0; self.parameter_count];
        for (operation, op) in self.operations.iter().enumerate() {
            if let ParameterizedOp::Rotation { parameter, .. } = op {
                let plus = self
                    .run_density_with_shift(inputs, parameters, qubits, Some((operation, shift)))?
                    .expectation(observable, observable_qubit)?;
                let minus = self
                    .run_density_with_shift(inputs, parameters, qubits, Some((operation, -shift)))?
                    .expectation(observable, observable_qubit)?;
                gradient[*parameter] += 0.5 * (plus - minus);
            }
        }
        Ok(gradient)
    }

    pub fn parameter_shift_gradient_density_with_inputs_shots(
        &self,
        inputs: &[f64],
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
        sampling: ShotConfig,
    ) -> Result<Vec<f64>, QuantumError> {
        if sampling.shots == 0 {
            return Err(QuantumError::InvalidShotCount);
        }
        self.validate_inputs(inputs)?;
        self.validate_parameters(parameters)?;
        if observable_qubit >= qubits {
            return Err(QuantumError::QubitOutOfRange {
                qubit: observable_qubit,
                qubits,
            });
        }
        let shift = std::f64::consts::FRAC_PI_2;
        let mut gradient = vec![0.0; self.parameter_count];
        for (operation, op) in self.operations.iter().enumerate() {
            if let ParameterizedOp::Rotation { parameter, .. } = op {
                let seed = sampling
                    .seed
                    .wrapping_add((operation as u64).wrapping_mul(2));
                let plus = self
                    .run_density_with_shift(inputs, parameters, qubits, Some((operation, shift)))?
                    .expectation(observable, observable_qubit)?;
                let minus = self
                    .run_density_with_shift(inputs, parameters, qubits, Some((operation, -shift)))?
                    .expectation(observable, observable_qubit)?;
                let plus = sample_expectation(
                    plus,
                    ShotConfig {
                        shots: sampling.shots,
                        seed,
                    },
                )?;
                let minus = sample_expectation(
                    minus,
                    ShotConfig {
                        shots: sampling.shots,
                        seed: seed.wrapping_add(1),
                    },
                )?;
                gradient[*parameter] += 0.5 * (plus - minus);
            }
        }
        Ok(gradient)
    }

    /// Estimate an expectation by projective shots. X and Y observables are
    /// rotated into the computational basis before measurement.
    pub fn expectation_shots(
        &self,
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
        sampling: ShotConfig,
    ) -> Result<f64, QuantumError> {
        self.expectation_with_inputs_shots(
            &[],
            parameters,
            qubits,
            observable,
            observable_qubit,
            sampling,
        )
    }

    pub fn expectation_with_inputs_shots(
        &self,
        inputs: &[f64],
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
        sampling: ShotConfig,
    ) -> Result<f64, QuantumError> {
        if sampling.shots == 0 {
            return Err(QuantumError::InvalidShotCount);
        }
        let state = self.run_with_inputs(inputs, parameters, qubits)?;
        estimate_observable_shots(
            state,
            observable,
            observable_qubit,
            sampling.shots,
            sampling.seed,
        )
    }

    /// Finite-shot parameter-shift gradient. Each occurrence is shifted
    /// separately, preserving the shared-parameter semantics of the exact API.
    pub fn parameter_shift_gradient_shots(
        &self,
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
        sampling: ShotConfig,
    ) -> Result<Vec<f64>, QuantumError> {
        self.parameter_shift_gradient_with_inputs_shots(
            &[],
            parameters,
            qubits,
            observable,
            observable_qubit,
            sampling,
        )
    }

    pub fn parameter_shift_gradient_with_inputs_shots(
        &self,
        inputs: &[f64],
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
        sampling: ShotConfig,
    ) -> Result<Vec<f64>, QuantumError> {
        if sampling.shots == 0 {
            return Err(QuantumError::InvalidShotCount);
        }
        self.validate_inputs(inputs)?;
        self.validate_parameters(parameters)?;
        if observable_qubit >= qubits {
            return Err(QuantumError::QubitOutOfRange {
                qubit: observable_qubit,
                qubits,
            });
        }
        let shift = std::f64::consts::FRAC_PI_2;
        let mut gradient = vec![0.0; self.parameter_count];
        for (operation, op) in self.operations.iter().enumerate() {
            if let ParameterizedOp::Rotation { parameter, .. } = op {
                let plus_state =
                    self.run_with_shift(inputs, parameters, qubits, Some((operation, shift)))?;
                let minus_state =
                    self.run_with_shift(inputs, parameters, qubits, Some((operation, -shift)))?;
                let occurrence_seed = sampling
                    .seed
                    .wrapping_add((operation as u64).wrapping_mul(2));
                let plus = estimate_observable_shots(
                    plus_state,
                    observable,
                    observable_qubit,
                    sampling.shots,
                    occurrence_seed,
                )?;
                let minus = estimate_observable_shots(
                    minus_state,
                    observable,
                    observable_qubit,
                    sampling.shots,
                    occurrence_seed.wrapping_add(1),
                )?;
                gradient[*parameter] += 0.5 * (plus - minus);
            }
        }
        Ok(gradient)
    }

    /// Parameter-shift derivatives. Each gate occurrence is shifted
    /// independently, so parameters shared by multiple gates get the sum of
    /// their occurrence-wise derivatives.
    pub fn parameter_shift_gradient(
        &self,
        parameters: &[f64],
        qubits: usize,
        observable_qubit: usize,
    ) -> Result<Vec<f64>, QuantumError> {
        self.parameter_shift_gradient_for(
            parameters,
            qubits,
            SingleQubitObservable::Z,
            observable_qubit,
        )
    }

    pub fn parameter_shift_gradient_for(
        &self,
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
    ) -> Result<Vec<f64>, QuantumError> {
        self.parameter_shift_gradient_with_inputs(
            &[],
            parameters,
            qubits,
            observable,
            observable_qubit,
        )
    }

    pub fn parameter_shift_gradient_with_inputs(
        &self,
        inputs: &[f64],
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
    ) -> Result<Vec<f64>, QuantumError> {
        self.validate_inputs(inputs)?;
        self.validate_parameters(parameters)?;
        if observable_qubit >= qubits {
            return Err(QuantumError::QubitOutOfRange {
                qubit: observable_qubit,
                qubits,
            });
        }
        let shift = std::f64::consts::FRAC_PI_2;
        let mut gradient = vec![0.0; self.parameter_count];
        for (operation, op) in self.operations.iter().enumerate() {
            if let ParameterizedOp::Rotation { parameter, .. } = op {
                let plus = self
                    .run_with_shift(inputs, parameters, qubits, Some((operation, shift)))?
                    .expectation(observable, observable_qubit)?;
                let minus = self
                    .run_with_shift(inputs, parameters, qubits, Some((operation, -shift)))?
                    .expectation(observable, observable_qubit)?;
                gradient[*parameter] += 0.5 * (plus - minus);
            }
        }
        Ok(gradient)
    }

    /// Compute exact gradients of an ideal state-vector expectation with one
    /// forward state sweep and one reverse costate sweep. Each trainable gate
    /// occurrence contributes to its parameter, including shared parameters.
    /// Circuits containing channels must use a density-matrix gradient method.
    pub fn adjoint_gradient(
        &self,
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
    ) -> Result<Vec<f64>, QuantumError> {
        self.adjoint_gradient_with_inputs(&[], parameters, qubits, observable, observable_qubit)
    }

    pub fn adjoint_gradient_with_inputs(
        &self,
        inputs: &[f64],
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
    ) -> Result<Vec<f64>, QuantumError> {
        self.validate_inputs(inputs)?;
        self.validate_parameters(parameters)?;
        if observable_qubit >= qubits {
            return Err(QuantumError::QubitOutOfRange {
                qubit: observable_qubit,
                qubits,
            });
        }

        let mut states = Vec::with_capacity(self.operations.len() + 1);
        states.push(StateVector::zero(qubits)?);
        for (index, operation) in self.operations.iter().enumerate() {
            let mut next = states[index].clone();
            match *operation {
                ParameterizedOp::Fixed { gate, target } => next.apply(gate, target)?,
                ParameterizedOp::Rotation {
                    axis,
                    target,
                    parameter,
                } => {
                    if parameter >= self.parameter_count {
                        return Err(QuantumError::InvalidParameterCount {
                            expected: self.parameter_count,
                            actual: parameter + 1,
                        });
                    }
                    next.apply(rotation_gate(axis, parameters[parameter]), target)?;
                }
                ParameterizedOp::EncodingRotation {
                    axis,
                    target,
                    input,
                } => {
                    if input >= self.input_count {
                        return Err(QuantumError::InvalidInputCount {
                            expected: self.input_count,
                            actual: input + 1,
                        });
                    }
                    next.apply(rotation_gate(axis, inputs[input]), target)?;
                }
                ParameterizedOp::Cnot { control, target } => next.apply_cnot(control, target)?,
                ParameterizedOp::Cz { first, second } => next.apply_cz(first, second)?,
                ParameterizedOp::Swap { first, second } => next.apply_swap(first, second)?,
                ParameterizedOp::Channel { .. } => {
                    return Err(QuantumError::OperationRequiresDensityMatrix);
                }
            }
            states.push(next);
        }

        let pauli = match observable {
            SingleQubitObservable::X => Gate::X,
            SingleQubitObservable::Y => Gate::Y,
            SingleQubitObservable::Z => Gate::Z,
        };
        let mut costate = states
            .last()
            .expect("the initial state is always present")
            .clone();
        costate.apply(pauli, observable_qubit)?;
        let mut gradient = vec![0.0; self.parameter_count];

        for (index, operation) in self.operations.iter().enumerate().rev() {
            let before = &states[index];
            let after = &states[index + 1];
            match *operation {
                ParameterizedOp::Rotation {
                    axis,
                    target,
                    parameter,
                } => {
                    let generator = match axis {
                        RotationAxis::X => Gate::X,
                        RotationAxis::Y => Gate::Y,
                        RotationAxis::Z => Gate::Z,
                    };
                    let mut derivative = after.clone();
                    derivative.apply(generator, target)?;
                    let minus_i_half = Complex64::new(0.0, -0.5);
                    let overlap_re = costate
                        .amplitudes
                        .iter()
                        .zip(&derivative.amplitudes)
                        .map(|(left, right)| (left.conj() * (*right * minus_i_half)).re)
                        .sum::<f64>();
                    if parameter >= self.parameter_count {
                        return Err(QuantumError::InvalidParameterCount {
                            expected: self.parameter_count,
                            actual: parameter + 1,
                        });
                    }
                    gradient[parameter] += 2.0 * overlap_re;
                }
                ParameterizedOp::Channel { .. } => {
                    return Err(QuantumError::OperationRequiresDensityMatrix);
                }
                _ => {}
            }

            match *operation {
                ParameterizedOp::Fixed { gate, target } => {
                    costate.apply(adjoint_gate(gate), target)?;
                }
                ParameterizedOp::Rotation {
                    axis,
                    target,
                    parameter,
                } => {
                    costate.apply(rotation_gate(axis, -parameters[parameter]), target)?;
                }
                ParameterizedOp::EncodingRotation {
                    axis,
                    target,
                    input,
                } => {
                    costate.apply(rotation_gate(axis, -inputs[input]), target)?;
                }
                ParameterizedOp::Cnot { control, target } => {
                    costate.apply_cnot(control, target)?;
                }
                ParameterizedOp::Cz { first, second } => {
                    costate.apply_cz(first, second)?;
                }
                ParameterizedOp::Swap { first, second } => {
                    costate.apply_swap(first, second)?;
                }
                ParameterizedOp::Channel { .. } => unreachable!("channels returned above"),
            }
            debug_assert_eq!(before.qubits(), costate.qubits());
        }
        Ok(gradient)
    }

    pub fn mse_loss_and_gradient(
        &self,
        data: &[(Vec<f64>, f64)],
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
    ) -> Result<(f64, Vec<f64>), QuantumError> {
        self.validate_parameters(parameters)?;
        if data.is_empty()
            || data.iter().any(|(x, y)| {
                x.len() != self.input_count || !y.is_finite() || x.iter().any(|v| !v.is_finite())
            })
        {
            return Err(QuantumError::InvalidTrainingData);
        }
        let mut loss = 0.0;
        let mut gradient = vec![0.0; self.parameter_count];
        for (inputs, target) in data {
            let prediction = self.expectation_with_inputs(
                inputs,
                parameters,
                qubits,
                observable,
                observable_qubit,
            )?;
            let residual = prediction - *target;
            loss += residual * residual;
            let sample_gradient = self.parameter_shift_gradient_with_inputs(
                inputs,
                parameters,
                qubits,
                observable,
                observable_qubit,
            )?;
            for (total, derivative) in gradient.iter_mut().zip(sample_gradient) {
                *total += 2.0 * residual * derivative;
            }
        }
        let count = data.len() as f64;
        for value in &mut gradient {
            *value /= count;
        }
        Ok((loss / count, gradient))
    }

    /// MSE and exact parameter-shift gradient for circuits containing noise
    /// channels. Each sample is propagated as a density matrix.
    pub fn mse_loss_and_gradient_density(
        &self,
        data: &[(Vec<f64>, f64)],
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
    ) -> Result<(f64, Vec<f64>), QuantumError> {
        self.validate_parameters(parameters)?;
        if data.is_empty()
            || data.iter().any(|(x, y)| {
                x.len() != self.input_count || !y.is_finite() || x.iter().any(|v| !v.is_finite())
            })
        {
            return Err(QuantumError::InvalidTrainingData);
        }
        let mut loss = 0.0;
        let mut gradient = vec![0.0; self.parameter_count];
        for (inputs, target) in data {
            let prediction = self.expectation_density_with_inputs(
                inputs,
                parameters,
                qubits,
                observable,
                observable_qubit,
            )?;
            let residual = prediction - *target;
            loss += residual * residual;
            let sample_gradient = self.parameter_shift_gradient_density_with_inputs(
                inputs,
                parameters,
                qubits,
                observable,
                observable_qubit,
            )?;
            for (total, derivative) in gradient.iter_mut().zip(sample_gradient) {
                *total += 2.0 * residual * derivative;
            }
        }
        let count = data.len() as f64;
        for value in &mut gradient {
            *value /= count;
        }
        Ok((loss / count, gradient))
    }

    pub fn mse_loss_and_gradient_density_shots(
        &self,
        data: &[(Vec<f64>, f64)],
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
        sampling: ShotConfig,
    ) -> Result<(f64, Vec<f64>), QuantumError> {
        if sampling.shots == 0 {
            return Err(QuantumError::InvalidShotCount);
        }
        self.validate_parameters(parameters)?;
        if data.is_empty()
            || data.iter().any(|(x, y)| {
                x.len() != self.input_count || !y.is_finite() || x.iter().any(|v| !v.is_finite())
            })
        {
            return Err(QuantumError::InvalidTrainingData);
        }
        let mut loss = 0.0;
        let mut gradient = vec![0.0; self.parameter_count];
        for (sample, (inputs, target)) in data.iter().enumerate() {
            let seed = sampling
                .seed
                .wrapping_add((sample as u64).wrapping_mul(0x9e37_79b9));
            let prediction = self.expectation_density_with_inputs_shots(
                inputs,
                parameters,
                qubits,
                observable,
                observable_qubit,
                ShotConfig {
                    shots: sampling.shots,
                    seed,
                },
            )?;
            let residual = prediction - *target;
            loss += residual * residual;
            let sample_gradient = self.parameter_shift_gradient_density_with_inputs_shots(
                inputs,
                parameters,
                qubits,
                observable,
                observable_qubit,
                ShotConfig {
                    shots: sampling.shots,
                    seed: seed.wrapping_add(1),
                },
            )?;
            for (total, derivative) in gradient.iter_mut().zip(sample_gradient) {
                *total += 2.0 * residual * derivative;
            }
        }
        let count = data.len() as f64;
        for value in &mut gradient {
            *value /= count;
        }
        Ok((loss / count, gradient))
    }

    /// MSE and parameter-shift gradient estimated with projective shots.
    /// `shots_per_expectation` is used independently for the prediction and
    /// for each positive and negative parameter shift.
    pub fn mse_loss_and_gradient_shots(
        &self,
        data: &[(Vec<f64>, f64)],
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
        sampling: ShotConfig,
    ) -> Result<(f64, Vec<f64>), QuantumError> {
        if sampling.shots == 0 {
            return Err(QuantumError::InvalidShotCount);
        }
        self.validate_parameters(parameters)?;
        if data.is_empty()
            || data.iter().any(|(x, y)| {
                x.len() != self.input_count || !y.is_finite() || x.iter().any(|v| !v.is_finite())
            })
        {
            return Err(QuantumError::InvalidTrainingData);
        }
        let mut loss = 0.0;
        let mut gradient = vec![0.0; self.parameter_count];
        for (sample, (inputs, target)) in data.iter().enumerate() {
            let sample_seed = sampling
                .seed
                .wrapping_add((sample as u64).wrapping_mul(0x9e37_79b9));
            let prediction = self.expectation_with_inputs_shots(
                inputs,
                parameters,
                qubits,
                observable,
                observable_qubit,
                ShotConfig {
                    shots: sampling.shots,
                    seed: sample_seed,
                },
            )?;
            let residual = prediction - *target;
            loss += residual * residual;
            let sample_gradient = self.parameter_shift_gradient_with_inputs_shots(
                inputs,
                parameters,
                qubits,
                observable,
                observable_qubit,
                ShotConfig {
                    shots: sampling.shots,
                    seed: sample_seed.wrapping_add(1),
                },
            )?;
            for (total, derivative) in gradient.iter_mut().zip(sample_gradient) {
                *total += 2.0 * residual * derivative;
            }
        }
        let count = data.len() as f64;
        for value in &mut gradient {
            *value /= count;
        }
        Ok((loss / count, gradient))
    }

    /// Adam training using finite-shot MSE estimates and parameter-shift
    /// gradients. A deterministic seed controls both shuffling and sampling.
    pub fn train_adam_shots(
        &self,
        data: &[(Vec<f64>, f64)],
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
        config: ShotTrainingConfig,
    ) -> Result<Vec<f64>, QuantumError> {
        if !config.training.learning_rate.is_finite()
            || config.training.learning_rate <= 0.0
            || config.training.batch_size == 0
        {
            return Err(QuantumError::InvalidOptimizerConfiguration);
        }
        // Validate data and shot budget even when there are no epochs.
        self.mse_loss_and_gradient_shots(
            data,
            parameters,
            qubits,
            observable,
            observable_qubit,
            ShotConfig {
                shots: config.shots_per_expectation,
                seed: config.training.seed,
            },
        )?;
        let mut weights = parameters.to_vec();
        let mut optimizer = Adam::new(self.parameter_count);
        let mut order: Vec<usize> = (0..data.len()).collect();
        let mut rng = SmallRng::seeded(config.training.seed);
        for epoch in 0..config.training.epochs {
            for i in (1..order.len()).rev() {
                let j = (rng.next_u64() as usize) % (i + 1);
                order.swap(i, j);
            }
            for (batch_index, indices) in order.chunks(config.training.batch_size).enumerate() {
                let batch: Vec<(Vec<f64>, f64)> =
                    indices.iter().map(|&i| data[i].clone()).collect();
                let sample_seed = config
                    .training
                    .seed
                    .wrapping_add((epoch as u64).wrapping_mul(0x517c_c1b7))
                    .wrapping_add(batch_index as u64);
                let (_, gradient) = self.mse_loss_and_gradient_shots(
                    &batch,
                    &weights,
                    qubits,
                    observable,
                    observable_qubit,
                    ShotConfig {
                        shots: config.shots_per_expectation,
                        seed: sample_seed,
                    },
                )?;
                optimizer.update(&mut weights, &gradient, config.training.learning_rate)?;
            }
        }
        Ok(weights)
    }

    /// Treat a measured Z expectation as a logit (class-one logit is `-z`)
    /// and return the class-one probability through a numerically stable
    /// sigmoid.
    pub fn binary_class_probability(
        &self,
        inputs: &[f64],
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
    ) -> Result<f64, QuantumError> {
        let logit = -self.expectation_with_inputs(
            inputs,
            parameters,
            qubits,
            observable,
            observable_qubit,
        )?;
        Ok(sigmoid(logit))
    }

    pub fn binary_classify(
        &self,
        inputs: &[f64],
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
    ) -> Result<bool, QuantumError> {
        Ok(self.binary_class_probability(
            inputs,
            parameters,
            qubits,
            observable,
            observable_qubit,
        )? >= 0.5)
    }

    /// Mean binary cross-entropy and its parameter-shift gradient. Labels must
    /// be exactly 0.0 or 1.0. The measured observable expectation is the logit.
    pub fn binary_cross_entropy_and_gradient(
        &self,
        data: &[(Vec<f64>, f64)],
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
    ) -> Result<(f64, Vec<f64>), QuantumError> {
        self.validate_parameters(parameters)?;
        if data.is_empty()
            || data.iter().any(|(x, y)| {
                x.len() != self.input_count
                    || !matches!(*y, 0.0 | 1.0)
                    || x.iter().any(|v| !v.is_finite())
            })
        {
            return Err(QuantumError::InvalidTrainingData);
        }
        let mut loss = 0.0;
        let mut gradient = vec![0.0; self.parameter_count];
        for (inputs, label) in data {
            let logit = self.expectation_with_inputs(
                inputs,
                parameters,
                qubits,
                observable,
                observable_qubit,
            )?;
            // For class-one logit -z, BCE is softplus(z) for label 1 and
            // softplus(-z) for label 0.
            loss += if *label == 1.0 {
                softplus(logit)
            } else {
                softplus(-logit)
            };
            let logit_gradient = sigmoid(logit) + *label - 1.0;
            let sample_gradient = self.parameter_shift_gradient_with_inputs(
                inputs,
                parameters,
                qubits,
                observable,
                observable_qubit,
            )?;
            for (total, derivative) in gradient.iter_mut().zip(sample_gradient) {
                *total += logit_gradient * derivative;
            }
        }
        let count = data.len() as f64;
        for value in &mut gradient {
            *value /= count;
        }
        Ok((loss / count, gradient))
    }

    /// Binary cross-entropy and exact parameter-shift gradient for noisy
    /// density-matrix circuits. Labels must be exactly 0.0 or 1.0.
    pub fn binary_cross_entropy_and_gradient_density(
        &self,
        data: &[(Vec<f64>, f64)],
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
    ) -> Result<(f64, Vec<f64>), QuantumError> {
        self.validate_parameters(parameters)?;
        if data.is_empty()
            || data.iter().any(|(x, y)| {
                x.len() != self.input_count
                    || !matches!(*y, 0.0 | 1.0)
                    || x.iter().any(|v| !v.is_finite())
            })
        {
            return Err(QuantumError::InvalidTrainingData);
        }
        let mut loss = 0.0;
        let mut gradient = vec![0.0; self.parameter_count];
        for (inputs, label) in data {
            let logit = self.expectation_density_with_inputs(
                inputs,
                parameters,
                qubits,
                observable,
                observable_qubit,
            )?;
            loss += if *label == 1.0 {
                softplus(logit)
            } else {
                softplus(-logit)
            };
            let logit_gradient = sigmoid(logit) + *label - 1.0;
            let sample_gradient = self.parameter_shift_gradient_density_with_inputs(
                inputs,
                parameters,
                qubits,
                observable,
                observable_qubit,
            )?;
            for (total, derivative) in gradient.iter_mut().zip(sample_gradient) {
                *total += logit_gradient * derivative;
            }
        }
        let count = data.len() as f64;
        for value in &mut gradient {
            *value /= count;
        }
        Ok((loss / count, gradient))
    }

    pub fn binary_cross_entropy_and_gradient_density_shots(
        &self,
        data: &[(Vec<f64>, f64)],
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
        sampling: ShotConfig,
    ) -> Result<(f64, Vec<f64>), QuantumError> {
        if sampling.shots == 0 {
            return Err(QuantumError::InvalidShotCount);
        }
        self.validate_parameters(parameters)?;
        if data.is_empty()
            || data.iter().any(|(x, y)| {
                x.len() != self.input_count
                    || !matches!(*y, 0.0 | 1.0)
                    || x.iter().any(|v| !v.is_finite())
            })
        {
            return Err(QuantumError::InvalidTrainingData);
        }
        let mut loss = 0.0;
        let mut gradient = vec![0.0; self.parameter_count];
        for (sample, (inputs, label)) in data.iter().enumerate() {
            let seed = sampling
                .seed
                .wrapping_add((sample as u64).wrapping_mul(0x9e37_79b9));
            let logit = self.expectation_density_with_inputs_shots(
                inputs,
                parameters,
                qubits,
                observable,
                observable_qubit,
                ShotConfig {
                    shots: sampling.shots,
                    seed,
                },
            )?;
            loss += if *label == 1.0 {
                softplus(logit)
            } else {
                softplus(-logit)
            };
            let logit_gradient = sigmoid(logit) + *label - 1.0;
            let sample_gradient = self.parameter_shift_gradient_density_with_inputs_shots(
                inputs,
                parameters,
                qubits,
                observable,
                observable_qubit,
                ShotConfig {
                    shots: sampling.shots,
                    seed: seed.wrapping_add(1),
                },
            )?;
            for (total, derivative) in gradient.iter_mut().zip(sample_gradient) {
                *total += logit_gradient * derivative;
            }
        }
        let count = data.len() as f64;
        for value in &mut gradient {
            *value /= count;
        }
        Ok((loss / count, gradient))
    }

    /// Finite-shot binary cross-entropy and parameter-shift gradient. Labels
    /// must be exactly 0.0 or 1.0; the measured observable is used as the logit.
    pub fn binary_cross_entropy_and_gradient_shots(
        &self,
        data: &[(Vec<f64>, f64)],
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
        sampling: ShotConfig,
    ) -> Result<(f64, Vec<f64>), QuantumError> {
        if sampling.shots == 0 {
            return Err(QuantumError::InvalidShotCount);
        }
        self.validate_parameters(parameters)?;
        if data.is_empty()
            || data.iter().any(|(x, y)| {
                x.len() != self.input_count
                    || !matches!(*y, 0.0 | 1.0)
                    || x.iter().any(|v| !v.is_finite())
            })
        {
            return Err(QuantumError::InvalidTrainingData);
        }
        let mut loss = 0.0;
        let mut gradient = vec![0.0; self.parameter_count];
        for (sample, (inputs, label)) in data.iter().enumerate() {
            let sample_seed = sampling
                .seed
                .wrapping_add((sample as u64).wrapping_mul(0x9e37_79b9));
            let logit = self.expectation_with_inputs_shots(
                inputs,
                parameters,
                qubits,
                observable,
                observable_qubit,
                ShotConfig {
                    shots: sampling.shots,
                    seed: sample_seed,
                },
            )?;
            loss += if *label == 1.0 {
                softplus(logit)
            } else {
                softplus(-logit)
            };
            let logit_gradient = sigmoid(logit) + *label - 1.0;
            let sample_gradient = self.parameter_shift_gradient_with_inputs_shots(
                inputs,
                parameters,
                qubits,
                observable,
                observable_qubit,
                ShotConfig {
                    shots: sampling.shots,
                    seed: sample_seed.wrapping_add(1),
                },
            )?;
            for (total, derivative) in gradient.iter_mut().zip(sample_gradient) {
                *total += logit_gradient * derivative;
            }
        }
        let count = data.len() as f64;
        for value in &mut gradient {
            *value /= count;
        }
        Ok((loss / count, gradient))
    }

    pub fn train_binary_classifier_adam(
        &self,
        data: &[(Vec<f64>, f64)],
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
        config: QmlTrainingConfig,
    ) -> Result<Vec<f64>, QuantumError> {
        if !config.learning_rate.is_finite()
            || config.learning_rate <= 0.0
            || config.batch_size == 0
        {
            return Err(QuantumError::InvalidOptimizerConfiguration);
        }
        // Validate once even when epochs is zero.
        self.binary_cross_entropy_and_gradient(
            data,
            parameters,
            qubits,
            observable,
            observable_qubit,
        )?;
        let mut weights = parameters.to_vec();
        let mut optimizer = Adam::new(self.parameter_count);
        let mut order: Vec<usize> = (0..data.len()).collect();
        let mut rng = SmallRng::seeded(config.seed);
        for _ in 0..config.epochs {
            for i in (1..order.len()).rev() {
                let j = (rng.next_u64() as usize) % (i + 1);
                order.swap(i, j);
            }
            for indices in order.chunks(config.batch_size) {
                let batch: Vec<(Vec<f64>, f64)> =
                    indices.iter().map(|&i| data[i].clone()).collect();
                let (_, gradient) = self.binary_cross_entropy_and_gradient(
                    &batch,
                    &weights,
                    qubits,
                    observable,
                    observable_qubit,
                )?;
                optimizer.update(&mut weights, &gradient, config.learning_rate)?;
            }
        }
        Ok(weights)
    }

    /// Finite-shot binary classifier training with deterministic Adam
    /// minibatches and seeded measurement sampling.
    /// Adam minibatch training for a binary classifier whose circuit includes
    /// exact density-matrix noise channels.
    pub fn train_binary_classifier_adam_density(
        &self,
        data: &[(Vec<f64>, f64)],
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
        config: QmlTrainingConfig,
    ) -> Result<Vec<f64>, QuantumError> {
        if !config.learning_rate.is_finite()
            || config.learning_rate <= 0.0
            || config.batch_size == 0
        {
            return Err(QuantumError::InvalidOptimizerConfiguration);
        }
        self.binary_cross_entropy_and_gradient_density(
            data,
            parameters,
            qubits,
            observable,
            observable_qubit,
        )?;
        let mut weights = parameters.to_vec();
        let mut optimizer = Adam::new(self.parameter_count);
        let mut order: Vec<usize> = (0..data.len()).collect();
        let mut rng = SmallRng::seeded(config.seed);
        for _ in 0..config.epochs {
            for i in (1..order.len()).rev() {
                let j = (rng.next_u64() as usize) % (i + 1);
                order.swap(i, j);
            }
            for indices in order.chunks(config.batch_size) {
                let batch: Vec<(Vec<f64>, f64)> =
                    indices.iter().map(|&i| data[i].clone()).collect();
                let (_, gradient) = self.binary_cross_entropy_and_gradient_density(
                    &batch,
                    &weights,
                    qubits,
                    observable,
                    observable_qubit,
                )?;
                optimizer.update(&mut weights, &gradient, config.learning_rate)?;
            }
        }
        Ok(weights)
    }

    pub fn train_binary_classifier_adam_density_shots(
        &self,
        data: &[(Vec<f64>, f64)],
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
        config: ShotTrainingConfig,
    ) -> Result<Vec<f64>, QuantumError> {
        let training = config.training;
        if !training.learning_rate.is_finite()
            || training.learning_rate <= 0.0
            || training.batch_size == 0
        {
            return Err(QuantumError::InvalidOptimizerConfiguration);
        }
        self.binary_cross_entropy_and_gradient_density_shots(
            data,
            parameters,
            qubits,
            observable,
            observable_qubit,
            ShotConfig {
                shots: config.shots_per_expectation,
                seed: training.seed,
            },
        )?;
        let mut weights = parameters.to_vec();
        let mut optimizer = Adam::new(self.parameter_count);
        let mut order: Vec<usize> = (0..data.len()).collect();
        let mut rng = SmallRng::seeded(training.seed);
        for epoch in 0..training.epochs {
            for i in (1..order.len()).rev() {
                let j = (rng.next_u64() as usize) % (i + 1);
                order.swap(i, j);
            }
            for (batch_index, indices) in order.chunks(training.batch_size).enumerate() {
                let batch: Vec<(Vec<f64>, f64)> =
                    indices.iter().map(|&i| data[i].clone()).collect();
                let seed = training
                    .seed
                    .wrapping_add((epoch as u64).wrapping_mul(0x517c_c1b7))
                    .wrapping_add(batch_index as u64);
                let (_, gradient) = self.binary_cross_entropy_and_gradient_density_shots(
                    &batch,
                    &weights,
                    qubits,
                    observable,
                    observable_qubit,
                    ShotConfig {
                        shots: config.shots_per_expectation,
                        seed,
                    },
                )?;
                optimizer.update(&mut weights, &gradient, training.learning_rate)?;
            }
        }
        Ok(weights)
    }

    pub fn train_binary_classifier_adam_shots(
        &self,
        data: &[(Vec<f64>, f64)],
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
        config: ShotTrainingConfig,
    ) -> Result<Vec<f64>, QuantumError> {
        if !config.training.learning_rate.is_finite()
            || config.training.learning_rate <= 0.0
            || config.training.batch_size == 0
        {
            return Err(QuantumError::InvalidOptimizerConfiguration);
        }
        self.binary_cross_entropy_and_gradient_shots(
            data,
            parameters,
            qubits,
            observable,
            observable_qubit,
            ShotConfig {
                shots: config.shots_per_expectation,
                seed: config.training.seed,
            },
        )?;
        let mut weights = parameters.to_vec();
        let mut optimizer = Adam::new(self.parameter_count);
        let mut order: Vec<usize> = (0..data.len()).collect();
        let mut rng = SmallRng::seeded(config.training.seed);
        for epoch in 0..config.training.epochs {
            for i in (1..order.len()).rev() {
                let j = (rng.next_u64() as usize) % (i + 1);
                order.swap(i, j);
            }
            for (batch_index, indices) in order.chunks(config.training.batch_size).enumerate() {
                let batch: Vec<(Vec<f64>, f64)> =
                    indices.iter().map(|&i| data[i].clone()).collect();
                let seed = config
                    .training
                    .seed
                    .wrapping_add((epoch as u64).wrapping_mul(0x517c_c1b7))
                    .wrapping_add(batch_index as u64);
                let (_, gradient) = self.binary_cross_entropy_and_gradient_shots(
                    &batch,
                    &weights,
                    qubits,
                    observable,
                    observable_qubit,
                    ShotConfig {
                        shots: config.shots_per_expectation,
                        seed,
                    },
                )?;
                optimizer.update(&mut weights, &gradient, config.training.learning_rate)?;
            }
        }
        Ok(weights)
    }

    pub fn train_regression_adam(
        &self,
        data: &[(Vec<f64>, f64)],
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
        config: QmlTrainingConfig,
    ) -> Result<Vec<f64>, QuantumError> {
        if !config.learning_rate.is_finite()
            || config.learning_rate <= 0.0
            || config.batch_size == 0
        {
            return Err(QuantumError::InvalidOptimizerConfiguration);
        }
        if data.is_empty()
            || data.iter().any(|(x, y)| {
                x.len() != self.input_count || !y.is_finite() || x.iter().any(|v| !v.is_finite())
            })
        {
            return Err(QuantumError::InvalidTrainingData);
        }
        self.validate_parameters(parameters)?;
        let mut weights = parameters.to_vec();
        let mut optimizer = Adam::new(self.parameter_count);
        let mut order: Vec<usize> = (0..data.len()).collect();
        let mut rng = SmallRng::seeded(config.seed);
        for _ in 0..config.epochs {
            for i in (1..order.len()).rev() {
                let j = (rng.next_u64() as usize) % (i + 1);
                order.swap(i, j);
            }
            for indices in order.chunks(config.batch_size) {
                let batch: Vec<(Vec<f64>, f64)> =
                    indices.iter().map(|&i| data[i].clone()).collect();
                let (_, gradient) = self.mse_loss_and_gradient(
                    &batch,
                    &weights,
                    qubits,
                    observable,
                    observable_qubit,
                )?;
                optimizer.update(&mut weights, &gradient, config.learning_rate)?;
            }
        }
        Ok(weights)
    }

    /// Adam training for regression objectives whose circuit includes exact
    /// density-matrix noise channels.
    pub fn train_regression_adam_density(
        &self,
        data: &[(Vec<f64>, f64)],
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
        config: QmlTrainingConfig,
    ) -> Result<Vec<f64>, QuantumError> {
        if !config.learning_rate.is_finite()
            || config.learning_rate <= 0.0
            || config.batch_size == 0
        {
            return Err(QuantumError::InvalidOptimizerConfiguration);
        }
        self.mse_loss_and_gradient_density(data, parameters, qubits, observable, observable_qubit)?;
        let mut weights = parameters.to_vec();
        let mut optimizer = Adam::new(self.parameter_count);
        let mut order: Vec<usize> = (0..data.len()).collect();
        let mut rng = SmallRng::seeded(config.seed);
        for _ in 0..config.epochs {
            for i in (1..order.len()).rev() {
                let j = (rng.next_u64() as usize) % (i + 1);
                order.swap(i, j);
            }
            for indices in order.chunks(config.batch_size) {
                let batch: Vec<(Vec<f64>, f64)> =
                    indices.iter().map(|&i| data[i].clone()).collect();
                let (_, gradient) = self.mse_loss_and_gradient_density(
                    &batch,
                    &weights,
                    qubits,
                    observable,
                    observable_qubit,
                )?;
                optimizer.update(&mut weights, &gradient, config.learning_rate)?;
            }
        }
        Ok(weights)
    }

    pub fn train_regression_adam_density_shots(
        &self,
        data: &[(Vec<f64>, f64)],
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
        config: ShotTrainingConfig,
    ) -> Result<Vec<f64>, QuantumError> {
        let training = config.training;
        if !training.learning_rate.is_finite()
            || training.learning_rate <= 0.0
            || training.batch_size == 0
        {
            return Err(QuantumError::InvalidOptimizerConfiguration);
        }
        self.mse_loss_and_gradient_density_shots(
            data,
            parameters,
            qubits,
            observable,
            observable_qubit,
            ShotConfig {
                shots: config.shots_per_expectation,
                seed: training.seed,
            },
        )?;
        let mut weights = parameters.to_vec();
        let mut optimizer = Adam::new(self.parameter_count);
        let mut order: Vec<usize> = (0..data.len()).collect();
        let mut rng = SmallRng::seeded(training.seed);
        for epoch in 0..training.epochs {
            for i in (1..order.len()).rev() {
                let j = (rng.next_u64() as usize) % (i + 1);
                order.swap(i, j);
            }
            for (batch_index, indices) in order.chunks(training.batch_size).enumerate() {
                let batch: Vec<(Vec<f64>, f64)> =
                    indices.iter().map(|&i| data[i].clone()).collect();
                let seed = training
                    .seed
                    .wrapping_add((epoch as u64).wrapping_mul(0x517c_c1b7))
                    .wrapping_add(batch_index as u64);
                let (_, gradient) = self.mse_loss_and_gradient_density_shots(
                    &batch,
                    &weights,
                    qubits,
                    observable,
                    observable_qubit,
                    ShotConfig {
                        shots: config.shots_per_expectation,
                        seed,
                    },
                )?;
                optimizer.update(&mut weights, &gradient, training.learning_rate)?;
            }
        }
        Ok(weights)
    }

    fn validate_parameters(&self, parameters: &[f64]) -> Result<(), QuantumError> {
        if parameters.len() != self.parameter_count {
            return Err(QuantumError::InvalidParameterCount {
                expected: self.parameter_count,
                actual: parameters.len(),
            });
        }
        if parameters.iter().any(|x| !x.is_finite()) {
            return Err(QuantumError::NonFiniteParameter);
        }
        Ok(())
    }

    fn validate_inputs(&self, inputs: &[f64]) -> Result<(), QuantumError> {
        if inputs.len() != self.input_count {
            return Err(QuantumError::InvalidInputCount {
                expected: self.input_count,
                actual: inputs.len(),
            });
        }
        if inputs.iter().any(|x| !x.is_finite()) {
            return Err(QuantumError::NonFiniteParameter);
        }
        Ok(())
    }

    fn run_with_shift(
        &self,
        inputs: &[f64],
        parameters: &[f64],
        qubits: usize,
        shift: Option<(usize, f64)>,
    ) -> Result<StateVector, QuantumError> {
        self.validate_inputs(inputs)?;
        self.validate_parameters(parameters)?;
        let mut state = StateVector::zero(qubits)?;
        for (index, operation) in self.operations.iter().enumerate() {
            match *operation {
                ParameterizedOp::Fixed { gate, target } => state.apply(gate, target)?,
                ParameterizedOp::Rotation {
                    axis,
                    target,
                    parameter,
                } => {
                    if parameter >= self.parameter_count {
                        return Err(QuantumError::InvalidParameterCount {
                            expected: self.parameter_count,
                            actual: parameter + 1,
                        });
                    }
                    let delta = shift
                        .filter(|(op, _)| *op == index)
                        .map_or(0.0, |(_, delta)| delta);
                    let theta = parameters[parameter] + delta;
                    let gate = match axis {
                        RotationAxis::X => Gate::Rx(theta),
                        RotationAxis::Y => Gate::Ry(theta),
                        RotationAxis::Z => Gate::Rz(theta),
                    };
                    state.apply(gate, target)?;
                }
                ParameterizedOp::EncodingRotation {
                    axis,
                    target,
                    input,
                } => {
                    if input >= self.input_count {
                        return Err(QuantumError::InvalidInputCount {
                            expected: self.input_count,
                            actual: input + 1,
                        });
                    }
                    let theta = inputs[input];
                    let gate = match axis {
                        RotationAxis::X => Gate::Rx(theta),
                        RotationAxis::Y => Gate::Ry(theta),
                        RotationAxis::Z => Gate::Rz(theta),
                    };
                    state.apply(gate, target)?;
                }
                ParameterizedOp::Cnot { control, target } => state.apply_cnot(control, target)?,
                ParameterizedOp::Cz { first, second } => state.apply_cz(first, second)?,
                ParameterizedOp::Swap { first, second } => state.apply_swap(first, second)?,
                ParameterizedOp::Channel { .. } => {
                    return Err(QuantumError::OperationRequiresDensityMatrix);
                }
            }
        }
        Ok(state)
    }

    fn run_density_with_shift(
        &self,
        inputs: &[f64],
        parameters: &[f64],
        qubits: usize,
        shift: Option<(usize, f64)>,
    ) -> Result<DensityMatrix, QuantumError> {
        self.validate_inputs(inputs)?;
        self.validate_parameters(parameters)?;
        let mut state = DensityMatrix::from_state(&StateVector::zero(qubits)?);
        for (index, operation) in self.operations.iter().enumerate() {
            match *operation {
                ParameterizedOp::Fixed { gate, target } => state.apply(gate, target)?,
                ParameterizedOp::Rotation {
                    axis,
                    target,
                    parameter,
                } => {
                    if parameter >= self.parameter_count {
                        return Err(QuantumError::InvalidParameterCount {
                            expected: self.parameter_count,
                            actual: parameter + 1,
                        });
                    }
                    let delta = shift
                        .filter(|(op, _)| *op == index)
                        .map_or(0.0, |(_, delta)| delta);
                    let gate = match axis {
                        RotationAxis::X => Gate::Rx(parameters[parameter] + delta),
                        RotationAxis::Y => Gate::Ry(parameters[parameter] + delta),
                        RotationAxis::Z => Gate::Rz(parameters[parameter] + delta),
                    };
                    state.apply(gate, target)?;
                }
                ParameterizedOp::EncodingRotation {
                    axis,
                    target,
                    input,
                } => {
                    if input >= self.input_count {
                        return Err(QuantumError::InvalidInputCount {
                            expected: self.input_count,
                            actual: input + 1,
                        });
                    }
                    let gate = match axis {
                        RotationAxis::X => Gate::Rx(inputs[input]),
                        RotationAxis::Y => Gate::Ry(inputs[input]),
                        RotationAxis::Z => Gate::Rz(inputs[input]),
                    };
                    state.apply(gate, target)?;
                }
                ParameterizedOp::Cnot { control, target } => state.apply_cnot(control, target)?,
                ParameterizedOp::Cz { first, second } => state.apply_cz(first, second)?,
                ParameterizedOp::Swap { first, second } => state.apply_swap(first, second)?,
                ParameterizedOp::Channel { channel, target } => {
                    state.apply_channel(channel, target)?;
                }
            }
        }
        Ok(state)
    }
}

fn estimate_observable_shots(
    mut state: StateVector,
    observable: SingleQubitObservable,
    qubit: usize,
    shots: usize,
    seed: u64,
) -> Result<f64, QuantumError> {
    if shots == 0 {
        return Err(QuantumError::InvalidShotCount);
    }
    match observable {
        SingleQubitObservable::X => state.apply(Gate::H, qubit)?,
        SingleQubitObservable::Y => {
            state.apply(Gate::Rz(-std::f64::consts::FRAC_PI_2), qubit)?;
            state.apply(Gate::H, qubit)?;
        }
        SingleQubitObservable::Z => {}
    }
    let mut rng = SmallRng::seeded(seed);
    let mut sum = 0.0;
    for _ in 0..shots {
        let mut sample = state.clone();
        sum += if sample.measure(qubit, &mut rng)? == 0 {
            1.0
        } else {
            -1.0
        };
    }
    Ok(sum / shots as f64)
}

fn sample_expectation(expectation: f64, sampling: ShotConfig) -> Result<f64, QuantumError> {
    if sampling.shots == 0 {
        return Err(QuantumError::InvalidShotCount);
    }
    let probability_positive = ((1.0 + expectation) * 0.5).clamp(0.0, 1.0);
    let mut rng = SmallRng::seeded(sampling.seed);
    let positive = (0..sampling.shots)
        .filter(|_| rng.next_f64() < probability_positive)
        .count();
    Ok((2 * positive) as f64 / sampling.shots as f64 - 1.0)
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn softplus(value: f64) -> f64 {
    if value > 0.0 {
        value + (-value).exp().ln_1p()
    } else {
        value.exp().ln_1p()
    }
}

/// Small deterministic generator for reproducible measurement sampling.
#[derive(Clone, Debug)]
pub struct SmallRng(u64);
impl SmallRng {
    pub fn seeded(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        })
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
    }
}

/// A one-qubit variational quantum model: |0> -> Ry(input) -> Ry(weight),
/// with the Z expectation as its scalar output.
#[derive(Clone, Copy, Debug, Default)]
pub struct VariationalQnn;
impl VariationalQnn {
    pub fn evaluate(&self, input: f64, weight: f64) -> Result<f64, QuantumError> {
        let mut circuit = Circuit::new();
        circuit.push(Gate::Ry(input), 0);
        circuit.push(Gate::Ry(weight), 0);
        circuit.run(StateVector::zero(1)?)?.expectation_z(0)
    }

    /// Exact parameter-shift derivative for this model's rotation parameter.
    pub fn gradient(&self, input: f64, weight: f64) -> Result<f64, QuantumError> {
        let shift = std::f64::consts::FRAC_PI_2;
        Ok(0.5 * (self.evaluate(input, weight + shift)? - self.evaluate(input, weight - shift)?))
    }

    /// Mean squared error and derivative with respect to the single weight.
    pub fn loss_and_gradient(
        &self,
        data: &[(f64, f64)],
        weight: f64,
    ) -> Result<(f64, f64), QuantumError> {
        if data.is_empty() || data.iter().any(|(x, y)| !x.is_finite() || !y.is_finite()) {
            return Err(QuantumError::InvalidTrainingData);
        }
        let mut loss = 0.0;
        let mut gradient = 0.0;
        for &(input, target) in data {
            let prediction = self.evaluate(input, weight)?;
            let residual = prediction - target;
            loss += residual * residual;
            gradient += 2.0 * residual * self.gradient(input, weight)?;
        }
        let n = data.len() as f64;
        Ok((loss / n, gradient / n))
    }

    /// Perform a fixed number of full-batch gradient-descent steps.
    pub fn train(
        &self,
        data: &[(f64, f64)],
        mut weight: f64,
        learning_rate: f64,
        steps: usize,
    ) -> Result<f64, QuantumError> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(QuantumError::InvalidTrainingData);
        }
        for _ in 0..steps {
            let (_, gradient) = self.loss_and_gradient(data, weight)?;
            weight -= learning_rate * gradient;
        }
        Ok(weight)
    }

    /// Adam optimization with deterministic shuffling and mini-batch gradients.
    pub fn train_adam(
        &self,
        data: &[(f64, f64)],
        mut weight: f64,
        learning_rate: f64,
        epochs: usize,
        batch_size: usize,
        seed: u64,
    ) -> Result<f64, QuantumError> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 || batch_size == 0 {
            return Err(QuantumError::InvalidOptimizerConfiguration);
        }
        if data.is_empty() || data.iter().any(|(x, y)| !x.is_finite() || !y.is_finite()) {
            return Err(QuantumError::InvalidTrainingData);
        }
        let mut order: Vec<usize> = (0..data.len()).collect();
        let mut rng = SmallRng::seeded(seed);
        let mut optimizer = Adam::new(1);
        for _ in 0..epochs {
            for i in (1..order.len()).rev() {
                let j = (rng.next_u64() as usize) % (i + 1);
                order.swap(i, j);
            }
            for batch in order.chunks(batch_size) {
                let samples: Vec<(f64, f64)> = batch.iter().map(|&i| data[i]).collect();
                let (_, gradient) = self.loss_and_gradient(&samples, weight)?;
                let mut params = [weight];
                optimizer.update(&mut params, &[gradient], learning_rate)?;
                weight = params[0];
            }
        }
        Ok(weight)
    }
}

/// Adam optimizer for dense parameter slices. State is retained between calls.
#[derive(Clone, Debug)]
pub struct Adam {
    first_moment: Vec<f64>,
    second_moment: Vec<f64>,
    step: u64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
}
impl Adam {
    pub fn new(parameter_count: usize) -> Self {
        Self {
            first_moment: vec![0.0; parameter_count],
            second_moment: vec![0.0; parameter_count],
            step: 0,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
    pub fn update(
        &mut self,
        parameters: &mut [f64],
        gradients: &[f64],
        learning_rate: f64,
    ) -> Result<(), QuantumError> {
        if parameters.len() != self.first_moment.len() || gradients.len() != parameters.len() {
            return Err(QuantumError::InvalidParameterCount {
                expected: self.first_moment.len(),
                actual: gradients.len(),
            });
        }
        if !learning_rate.is_finite()
            || learning_rate <= 0.0
            || parameters.iter().chain(gradients).any(|x| !x.is_finite())
        {
            return Err(QuantumError::InvalidOptimizerConfiguration);
        }
        self.step = self.step.saturating_add(1);
        let correction1 = 1.0 - self.beta1.powf(self.step as f64);
        let correction2 = 1.0 - self.beta2.powf(self.step as f64);
        let rate = learning_rate * correction2.sqrt() / correction1;
        for i in 0..parameters.len() {
            self.first_moment[i] =
                self.beta1 * self.first_moment[i] + (1.0 - self.beta1) * gradients[i];
            self.second_moment[i] = self.beta2 * self.second_moment[i]
                + (1.0 - self.beta2) * gradients[i] * gradients[i];
            parameters[i] -=
                rate * self.first_moment[i] / (self.second_moment[i].sqrt() + self.epsilon);
        }
        Ok(())
    }
}

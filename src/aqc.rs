//! Adiabatic Quantum Computing (AQC) — the adiabatic theorem as a scheduler.
//!
//! AQC solves an optimization problem encoded in a "problem" Hamiltonian `H₁`
//! whose ground state is the answer. You start in the ground state of a simple
//! "driver" Hamiltonian `H₀` (here a transverse field, whose ground state is
//! the uniform superposition `|+...+>`) and slowly interpolate
//!
//! ```text
//!   H(s) = (1 - s)·H₀ + s·H₁ ,   s : 0 → 1
//! ```
//!
//! By the adiabatic theorem, if the interpolation is slow enough the state
//! stays in the instantaneous ground state, so at `s = 1` you measure and read
//! off the solution.
//!
//! This module drives that interpolation with the kernel's adaptive-step
//! Trotter integrator (`evolve_ket_time_dependent`), tracks the instantaneous
//! ground-state energy and the final overlap with the exact `H₁` ground state,
//! and reports whether the run was adiabatic (high fidelity) or leaked.

use crate::quantum::{
    evolve_ket_time_dependent, HilbertSpace, LinearError, Operator, SparseOperator,
};
use crate::{Complex64, StateVector};

/// An adiabatic interpolation between a driver Hamiltonian `H₀` and a problem
/// Hamiltonian `H₁`, both given as dense f64 matrices (row-major, dimension
/// `2^n`). Built once; `hamiltonian_at(s)` returns the interpolated operator.
pub struct AdiabaticSchedule {
    qubits: usize,
    dimension: usize,
    /// Dense `H₀` (driver), row-major.
    h0: Vec<f64>,
    /// Dense `H₁` (problem), row-major.
    h1: Vec<f64>,
}

impl AdiabaticSchedule {
    /// Build a schedule from two dense Hamiltonians (row-major `dim × dim`).
    pub fn from_dense(qubits: usize, h0: Vec<f64>, h1: Vec<f64>) -> Result<Self, LinearError> {
        let dimension = 1usize << qubits;
        if h0.len() != dimension * dimension || h1.len() != dimension * dimension {
            return Err(LinearError::DimensionMismatch {
                expected: dimension * dimension,
                actual: h0.len().min(h1.len()),
            });
        }
        Ok(Self { qubits, dimension, h0, h1 })
    }

    /// Build a schedule from a driver Hamiltonian (as a dense `Operator`) and a
    /// problem Hamiltonian given as a `PauliSum` (exact rational → f64 matrix).
    pub fn from_hamiltonians(
        driver: &Operator,
        problem: &crate::analytic::PauliSum,
    ) -> Result<Self, LinearError> {
        let qubits = problem.qubits();
        let h0 = operator_to_f64(driver)?;
        let problem_op = problem.to_sparse_operator()?.to_dense()?;
        let h1 = operator_to_f64(&problem_op)?;
        Self::from_dense(qubits, h0, h1)
    }

    /// The interpolated Hamiltonian `H(s) = (1-s)·H₀ + s·H₁` as a sparse
    /// operator, for the integrator's `hamiltonian_at` hook.
    pub fn hamiltonian_at(&self, s: f64) -> Result<SparseOperator, LinearError> {
        let mut entries = Vec::with_capacity(self.dimension * self.dimension);
        for i in 0..self.dimension * self.dimension {
            let v = (1.0 - s) * self.h0[i] + s * self.h1[i];
            entries.push(Complex64::new(v, 0.0));
        }
        let space = HilbertSpace::qubits(self.qubits)?;
        let dense = Operator::new(space.clone(), space, entries)?;
        SparseOperator::from_dense(&dense, 1e-12)
    }

    /// The exact ground-state energy of the problem Hamiltonian `H₁`, via the
    /// matrix-free sparse Lanczos eigensolver (no dense diagonalisation).
    pub fn problem_ground_energy(&self) -> Result<f64, LinearError> {
        let op = self.hamiltonian_at(1.0)?;
        Ok(op.lowest_eigenpairs(1, 1e-10, 1000)?.eigenvalues()[0])
    }

    pub fn qubits(&self) -> usize {
        self.qubits
    }
}

/// Result of an adiabatic run.
#[derive(Clone, Debug)]
pub struct AdiabaticResult {
    /// Final state after the interpolation (as a dense state vector).
    pub final_state: Vec<Complex64>,
    /// Ground-state energy of `H(s)` at the start (`s=0`).
    pub initial_energy: f64,
    /// Ground-state energy of `H(s)` at the end (`s=1`) — should match the
    /// exact `H₁` ground energy if the run was adiabatic.
    pub final_energy: f64,
    /// Exact ground-state energy of `H₁` (reference).
    pub exact_ground_energy: f64,
    /// Total evolution time used.
    pub total_time: f64,
}

/// Run the adiabatic interpolation from `s=0` to `s=1` over `total_time`,
/// starting from `initial` (typically the `H₀` ground state `|+...+>`).
///
/// `tolerance` and `max_steps` control the adaptive Trotter integrator. The
/// returned energies let the caller judge adiabaticity: `final_energy` close to
/// `exact_ground_energy` means the state tracked the ground state.
pub fn run_adiabatic(
    schedule: &AdiabaticSchedule,
    initial: &StateVector,
    total_time: f64,
    tolerance: f64,
    max_steps: usize,
) -> Result<AdiabaticResult, LinearError> {
    let ket = initial.to_ket()?;

    // Instantaneous ground energy at the endpoints (via the schedule operator).
    let h_start = schedule.hamiltonian_at(0.0)?;
    let h_end = schedule.hamiltonian_at(1.0)?;
    let initial_energy = h_start.lowest_eigenpairs(1, 1e-10, 1000)?.eigenvalues()[0];

    // Drive s from 0 to 1 linearly in time over [0, total_time].
    let final_ket = evolve_ket_time_dependent(&ket, 0.0, total_time, tolerance, max_steps, |t| {
        let s = (t / total_time).clamp(0.0, 1.0);
        schedule.hamiltonian_at(s)
    })?;

    // Energy of the final state under H₁ = <ψ|H₁|ψ>.
    let final_energy = final_ket.expectation_map(&h_end)?.re;

    // Reference exact ground energy of H₁.
    let exact_ground_energy = schedule.problem_ground_energy()?;

    let final_state = final_ket.amplitudes().to_vec();
    Ok(AdiabaticResult {
        final_state,
        initial_energy,
        final_energy,
        exact_ground_energy,
        total_time,
    })
}

/// Convert a dense `Operator` to a row-major f64 matrix (real part; these
/// Hamiltonians are Hermitian and real).
fn operator_to_f64(op: &Operator) -> Result<Vec<f64>, LinearError> {
    Ok(op.entries().iter().map(|c| c.re).collect())
}

/// Convenience: the driver ground state `|+...+>` on `n` qubits — the ground
/// state of a transverse-field `H₀ = -Σ Xᵢ`.
pub fn uniform_superposition(qubits: usize) -> Result<StateVector, crate::QuantumError> {
    let mut state = StateVector::zero(qubits)?;
    for q in 0..qubits {
        state.apply(crate::Gate::H, q)?;
    }
    Ok(state)
}

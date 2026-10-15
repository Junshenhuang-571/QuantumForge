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
    HilbertSpace, LinearError, Operator, SparseOperator, evolve_ket_time_dependent,
};
use crate::{Complex64, StateVector};

/// Minimum per-step tolerance the adiabatic driver will relax to before giving
/// up. Below this the Krylov evolution cannot converge within any sane step
/// budget, so retrying is futile — we report an actionable error instead.
const TOLERANCE_FLOOR: f64 = 1e-4;

/// Errors from an adiabatic run. Wraps the linear-algebra errors and adds an
/// actionable non-convergence case (the raw `EigenDecompositionNoConvergence`
/// from the integrator carries no guidance for the caller).
#[derive(Clone, Debug)]
pub enum AdiabaticError {
    /// A wrapped linear-algebra / eigensolver error.
    Linear(LinearError),
    /// The integrator could not converge even at the relaxed tolerance floor.
    NonConvergent {
        total_time: f64,
        tolerance: f64,
        max_steps: usize,
    },
}

impl std::fmt::Display for AdiabaticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AdiabaticError::Linear(e) => write!(f, "{e}"),
            AdiabaticError::NonConvergent {
                total_time,
                tolerance,
                max_steps,
            } => write!(
                f,
                "adiabatic evolution did not converge: total_time={total_time}, tolerance={tolerance} (floor {TOLERANCE_FLOOR}), max_steps={max_steps}. \
                 Try a shorter total_time, a looser tolerance, or a larger max_steps."
            ),
        }
    }
}
impl std::error::Error for AdiabaticError {}
impl From<LinearError> for AdiabaticError {
    fn from(e: LinearError) -> Self {
        AdiabaticError::Linear(e)
    }
}

/// The annealing path `s(t) ∈ [0,1]` from driver (`s=0`) to problem (`s=1`)
/// over normalized time `τ = t / total_time ∈ [0,1]`. Different paths trade
/// off minimum-gap crossing speed against runtime — the linear path is the
/// textbook default; `FastStart` spends more time near the small-gap region.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum AnnealPath {
    /// `s = τ` — the standard linear interpolation.
    #[default]
    Linear,
    /// `s = τ / (τ + a·(1-τ))` with `a < 1` — a "fast-start" (Roland–Cerf)
    /// schedule that lingers near `s≈0` and accelerates through the gap.
    /// `a` must be in `(0,1)`; smaller `a` = faster start.
    FastStart(f64),
    /// `s = τ^p` with `p > 1` — a polynomial path that is slow at the start.
    Polynomial(f64),
}

impl AnnealPath {
    /// Evaluate the path at normalized time `τ ∈ [0,1]`, clamped to `[0,1]`.
    pub fn at(&self, tau: f64) -> f64 {
        let tau = tau.clamp(0.0, 1.0);
        match self {
            AnnealPath::Linear => tau,
            AnnealPath::FastStart(a) => {
                let a = a.clamp(1e-6, 1.0 - 1e-6);
                tau / (tau + a * (1.0 - tau))
            }
            AnnealPath::Polynomial(p) => tau.powf(*p),
        }
    }

    /// Short label for display.
    pub fn label(&self) -> String {
        match self {
            AnnealPath::Linear => "linear".to_owned(),
            AnnealPath::FastStart(a) => format!("fast-start(a={a})"),
            AnnealPath::Polynomial(p) => format!("poly(p={p})"),
        }
    }
}

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
    /// The annealing path `s(t)`.
    path: AnnealPath,
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
        Ok(Self {
            qubits,
            dimension,
            h0,
            h1,
            path: AnnealPath::Linear,
        })
    }

    /// Set the annealing path (builder). Defaults to [`AnnealPath::Linear`].
    pub fn with_path(mut self, path: AnnealPath) -> Self {
        self.path = path;
        self
    }

    /// The annealing path in use.
    pub fn path(&self) -> AnnealPath {
        self.path
    }

    /// Evaluate the annealing path at normalized time `τ = t / total_time`.
    pub fn s_at(&self, tau: f64) -> f64 {
        self.path.at(tau)
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
) -> Result<AdiabaticResult, AdiabaticError> {
    let ket = initial.to_ket()?;

    // Instantaneous ground energy at the endpoints (via the schedule operator).
    let h_start = schedule.hamiltonian_at(0.0)?;
    let h_end = schedule.hamiltonian_at(1.0)?;
    let initial_energy = h_start.lowest_eigenpairs(1, 1e-10, 1000)?.eigenvalues()[0];

    // Drive s from 0 to 1 along the schedule's annealing path over
    // [0, total_time]. The adaptive integrator halves its step on convergence
    // failure but never floors it, so a tolerance tighter than the step budget
    // allows burns through `max_steps` and returns EigenDecompositionNoConvergence.
    // Guard against that by retrying once with a relaxed tolerance (floor =
    // 1e-4) if the first pass fails to converge.
    let path = schedule.path;
    let drive = |tol: f64| {
        evolve_ket_time_dependent(&ket, 0.0, total_time, tol, max_steps, |t| {
            let s = path.at(t / total_time);
            schedule.hamiltonian_at(s)
        })
    };
    let final_ket = match drive(tolerance) {
        Ok(k) => k,
        Err(LinearError::EigenDecompositionNoConvergence) => {
            let relaxed = tolerance.max(TOLERANCE_FLOOR);
            drive(relaxed).map_err(|_| AdiabaticError::NonConvergent {
                total_time,
                tolerance: relaxed,
                max_steps,
            })?
        }
        Err(e) => return Err(e.into()),
    };

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

/// Build the canonical transverse-field driver `H₀ = -Σᵢ Xᵢ` as a dense
/// row-major `dim × dim` operator on `qubits` qubits. Its ground state is the
/// uniform superposition `|+...+>` (see [`uniform_superposition`]), the standard
/// AQC starting point. Lets callers derive the driver from the problem's size
/// instead of hand-constructing it.
pub fn transverse_field_driver(qubits: usize) -> Result<Vec<f64>, LinearError> {
    let dim = 1usize << qubits;
    let mut entries = vec![0.0; dim * dim];
    for q in 0..qubits {
        let mask = 1usize << q;
        for i in 0..dim {
            let j = i ^ mask; // X flips bit q
            entries[i * dim + j] -= 1.0;
        }
    }
    Ok(entries)
}

/// Kronecker product of two 2×2 matrices (row-major), returning a 4×4 matrix.
/// Shared helper for building small adiabatic Hamiltonians (driver / problem)
/// in the REPL, tests, and examples.
pub fn kron2(a: &[f64; 4], b: &[f64; 4]) -> Vec<f64> {
    let mut out = vec![0.0; 16];
    for r in 0..2 {
        for c in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    out[(r * 2 + k) * 4 + (c * 2 + l)] = a[r * 2 + c] * b[k * 2 + l];
                }
            }
        }
    }
    out
}

/// Pauli matrices as 2×2 row-major arrays.
pub const PAULI_I: [f64; 4] = [1.0, 0.0, 0.0, 1.0];
pub const PAULI_X: [f64; 4] = [0.0, 1.0, 1.0, 0.0];
pub const PAULI_Z: [f64; 4] = [1.0, 0.0, 0.0, -1.0];

/// The canonical 2-qubit demo schedule: transverse-field driver
/// `H₀ = -(X⊗I + I⊗X)` (ground state `|++>`) interpolated to the Ising problem
/// `H₁ = Z⊗Z` (ground states `|00>`,`|11>`). Shared by the REPL `anneal` demo
/// and its tests so they cannot drift apart.
pub fn demo_schedule() -> Result<AdiabaticSchedule, LinearError> {
    let xi = kron2(&PAULI_X, &PAULI_I);
    let ix = kron2(&PAULI_I, &PAULI_X);
    let h0: Vec<f64> = (0..16).map(|k| -(xi[k] + ix[k])).collect();
    let zz = kron2(&PAULI_Z, &PAULI_Z);
    AdiabaticSchedule::from_dense(2, h0, zz)
}

//! Preflight estimates for dense quantum-state representations.
//!
//! The estimates cover the complex-number payload only. They exclude allocator
//! metadata, temporary workspaces, thread-local copies, and simulator overhead,
//! so they are lower bounds rather than guarantees that a run will fit.

use crate::Complex64;
use serde::Serialize;
use std::mem::size_of;

/// A lower-bound estimate for dense state-vector and density-matrix storage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DenseMemoryEstimate {
    /// Number of qubits requested.
    pub qubits: usize,
    /// Bytes for `2^n` complex amplitudes, if representable in `u128`.
    pub state_vector_bytes: Option<u128>,
    /// Bytes for `4^n` complex density-matrix entries, if representable.
    pub density_matrix_bytes: Option<u128>,
    /// Caller supplied memory budget. `None` means no budget was supplied.
    pub memory_budget_bytes: Option<u128>,
    /// Whether the state-vector payload fits the supplied budget.
    pub state_vector_fits_budget: Option<bool>,
    /// Whether the density-matrix payload fits the supplied budget.
    pub density_matrix_fits_budget: Option<bool>,
    /// Reminder that these values exclude working memory and process overhead.
    pub estimate_scope: &'static str,
}

/// Errors produced by dense-memory preflight estimation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimulationPlanError {
    /// A quantum state must contain at least one qubit.
    ZeroQubits,
}

impl std::fmt::Display for SimulationPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroQubits => write!(f, "qubit count must be greater than zero"),
        }
    }
}

impl std::error::Error for SimulationPlanError {}

/// Estimate the minimum dense state-vector and density-matrix payload sizes.
///
/// The returned `None` byte count means the size exceeds the `u128` estimate
/// range. A supplied budget is compared to the payload only; applications
/// should leave additional headroom for simulator workspaces and the rest of
/// the process. This function does not choose a simulator or predict whether
/// tensor-network, stabilizer, or trajectory simulation is feasible.
pub fn estimate_dense_memory(
    qubits: usize,
    memory_budget_bytes: Option<u128>,
) -> Result<DenseMemoryEstimate, SimulationPlanError> {
    if qubits == 0 {
        return Err(SimulationPlanError::ZeroQubits);
    }

    let dimension = u32::try_from(qubits)
        .ok()
        .and_then(|shift| 1u128.checked_shl(shift));
    let bytes_per_complex = size_of::<Complex64>() as u128;
    let state_vector_bytes = dimension.and_then(|n| n.checked_mul(bytes_per_complex));
    let density_matrix_bytes = dimension
        .and_then(|n| n.checked_mul(n))
        .and_then(|entries| entries.checked_mul(bytes_per_complex));

    Ok(DenseMemoryEstimate {
        qubits,
        state_vector_fits_budget: memory_budget_bytes
            .zip(state_vector_bytes)
            .map(|(budget, needed)| needed <= budget),
        density_matrix_fits_budget: memory_budget_bytes
            .zip(density_matrix_bytes)
            .map(|(budget, needed)| needed <= budget),
        state_vector_bytes,
        density_matrix_bytes,
        memory_budget_bytes,
        estimate_scope: "complex payload lower bound; excludes workspaces and process overhead",
    })
}

#[cfg(test)]
mod tests {
    use super::{SimulationPlanError, estimate_dense_memory};

    #[test]
    fn estimates_dense_payloads_and_budget_fit() {
        let estimate = estimate_dense_memory(10, Some(1_000_000)).unwrap();
        assert_eq!(estimate.state_vector_bytes, Some(16_384));
        assert_eq!(estimate.density_matrix_bytes, Some(16_777_216));
        assert_eq!(estimate.state_vector_fits_budget, Some(true));
        assert_eq!(estimate.density_matrix_fits_budget, Some(false));
    }

    #[test]
    fn estimates_remain_useful_when_density_size_overflows() {
        let estimate = estimate_dense_memory(64, Some(u128::MAX)).unwrap();
        assert_eq!(estimate.state_vector_bytes, Some(1u128 << 68));
        assert_eq!(estimate.density_matrix_bytes, None);
        assert_eq!(estimate.state_vector_fits_budget, Some(true));
        assert_eq!(estimate.density_matrix_fits_budget, None);
    }

    #[test]
    fn rejects_zero_qubits() {
        assert_eq!(
            estimate_dense_memory(0, None),
            Err(SimulationPlanError::ZeroQubits)
        );
    }

    #[test]
    fn reports_when_both_estimates_exceed_the_supported_integer_range() {
        let estimate = estimate_dense_memory(128, None).unwrap();
        assert_eq!(estimate.state_vector_bytes, None);
        assert_eq!(estimate.density_matrix_bytes, None);
        assert_eq!(estimate.state_vector_fits_budget, None);
        assert_eq!(estimate.density_matrix_fits_budget, None);
    }

    #[test]
    fn large_byte_counts_serialize_without_losing_the_estimate() {
        let estimate = estimate_dense_memory(64, None).unwrap();
        let json = serde_json::to_string(&estimate).unwrap();
        assert!(json.contains("295147905179352825856"));
    }
}

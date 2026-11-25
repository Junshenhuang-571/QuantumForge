use quantumforge::simulation_plan::estimate_dense_memory;
use quantumforge::{DensityMatrix, QuantumError, StateVector};

/// The preflight estimate and the budgeted constructors must agree: a payload
/// the estimate says fits has to be allocatable, and one it says does not fit
/// has to be rejected.
#[test]
fn estimate_agrees_with_the_budgeted_constructors() {
    let qubits = 3;
    // 2^3 amplitudes of Complex64 (16 bytes each) = 128 bytes.
    let state_bytes = 1usize << qubits;
    let state_bytes = state_bytes * std::mem::size_of::<quantumforge::Complex64>();
    // 4^3 density-matrix entries of Complex64 = 1024 bytes.
    let density_bytes = (1usize << qubits).pow(2) * std::mem::size_of::<quantumforge::Complex64>();

    let at_limit = estimate_dense_memory(qubits, Some(state_bytes as u128)).unwrap();
    assert_eq!(at_limit.state_vector_bytes, Some(state_bytes as u128));
    assert_eq!(at_limit.state_vector_fits_budget, Some(true));
    assert_eq!(at_limit.density_matrix_bytes, Some(density_bytes as u128));
    assert_eq!(at_limit.density_matrix_fits_budget, Some(false));

    // Exactly at the boundary still allocates; one byte below is rejected.
    assert!(StateVector::zero_with_memory_budget(qubits, state_bytes as u128).is_ok());
    assert!(StateVector::zero_with_memory_budget(qubits, state_bytes as u128 - 1).is_err());
    assert!(
        DensityMatrix::maximally_mixed_with_memory_budget(qubits, density_bytes as u128).is_ok()
    );
    assert!(
        DensityMatrix::maximally_mixed_with_memory_budget(qubits, density_bytes as u128 - 1)
            .is_err()
    );

    // With no budget supplied the fit flags stay unset rather than guessing.
    let unbounded = estimate_dense_memory(qubits, None).unwrap();
    assert_eq!(unbounded.memory_budget_bytes, None);
    assert_eq!(unbounded.state_vector_fits_budget, None);
    assert_eq!(unbounded.density_matrix_fits_budget, None);
}

#[test]
fn dense_allocations_obey_payload_budgets() {
    let state = StateVector::zero_with_memory_budget(3, 128).unwrap();
    assert_eq!(state.amplitudes().len(), 8);
    assert!(matches!(
        StateVector::zero_with_memory_budget(3, 127),
        Err(QuantumError::MemoryBudgetExceeded {
            required_bytes: 128,
            budget_bytes: 127
        })
    ));

    let mixed = DensityMatrix::maximally_mixed_with_memory_budget(3, 1024).unwrap();
    assert_eq!(mixed.entries().len(), 64);
    assert!((mixed.trace().re - 1.0).abs() < 1e-12);
    assert!(matches!(
        DensityMatrix::maximally_mixed_with_memory_budget(3, 1023),
        Err(QuantumError::MemoryBudgetExceeded {
            required_bytes: 1024,
            budget_bytes: 1023
        })
    ));
}

#[test]
fn legacy_dense_constructors_return_errors_for_unrepresentable_sizes() {
    assert!(matches!(
        StateVector::zero(usize::MAX),
        Err(QuantumError::SimulationSizeOverflow)
    ));
    assert!(matches!(
        DensityMatrix::maximally_mixed(usize::MAX),
        Err(QuantumError::SimulationSizeOverflow)
    ));
}

use quantumforge::quantum::{
    HilbertSpace, Ket, LinearError, Observable, Operator, SparseOperator, evolve_ket_time_dependent,
};
use quantumforge::{Complex64, DensityMatrix, Gate, StateVector};

fn near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} differs from {expected}"
    );
}

fn pauli_y() -> Operator {
    let space = HilbertSpace::qubits(1).unwrap();
    Operator::new(
        space.clone(),
        space,
        vec![
            Complex64::ZERO,
            Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 1.0),
            Complex64::ZERO,
        ],
    )
    .unwrap()
}

#[test]
fn complex_hermitian_eigensystem_has_expected_spectrum_and_vectors() {
    let hamiltonian = pauli_y();
    let result = hamiltonian.hermitian_eigensystem(1e-13, 32).unwrap();
    near(result.eigenvalues()[0], -1.0, 1e-12);
    near(result.eigenvalues()[1], 1.0, 1e-12);

    for (column, &eigenvalue) in result.eigenvalues().iter().enumerate() {
        let mut vector = vec![Complex64::ZERO; 2];
        for (row, amplitude) in vector.iter_mut().enumerate() {
            *amplitude = result.eigenvectors().entries()[row * 2 + column];
        }
        let ket = Ket::new(HilbertSpace::qubits(1).unwrap(), vector.clone()).unwrap();
        let residual = hamiltonian
            .apply(&ket)
            .unwrap()
            .amplitudes()
            .iter()
            .zip(vector)
            .map(|(actual, value)| (*actual - value * eigenvalue).abs())
            .fold(0.0_f64, f64::max);
        assert!(residual < 1e-12, "eigenvector residual {residual}");
    }
}

#[test]
fn jacobi_eigensolver_resolves_a_dense_complex_hamiltonian() {
    let space = HilbertSpace::new(vec![3]).unwrap();
    let a01 = Complex64::new(0.7, 1.2);
    let a02 = Complex64::new(-0.4, 0.3);
    let a12 = Complex64::new(0.9, -0.8);
    let hamiltonian = Operator::new(
        space.clone(),
        space.clone(),
        vec![
            Complex64::new(1.0, 0.0),
            a01,
            a02,
            a01.conj(),
            Complex64::new(-0.5, 0.0),
            a12,
            a02.conj(),
            a12.conj(),
            Complex64::new(2.0, 0.0),
        ],
    )
    .unwrap();
    let result = hamiltonian.hermitian_eigensystem(1e-12, 64).unwrap();

    for (column, &eigenvalue) in result.eigenvalues().iter().enumerate() {
        let mut vector = vec![Complex64::ZERO; 3];
        for (row, amplitude) in vector.iter_mut().enumerate() {
            *amplitude = result.eigenvectors().entries()[row * 3 + column];
        }
        let ket = Ket::new(space.clone(), vector.clone()).unwrap();
        let residual = hamiltonian
            .apply(&ket)
            .unwrap()
            .amplitudes()
            .iter()
            .zip(vector)
            .map(|(actual, value)| (*actual - value * eigenvalue).abs())
            .fold(0.0_f64, f64::max);
        assert!(residual < 1e-10, "eigenvector residual {residual}");
    }
}

#[test]
fn sparse_lanczos_time_evolution_matches_dense_spectral_reference() {
    let space = HilbertSpace::new(vec![3]).unwrap();
    let a01 = Complex64::new(0.7, 0.2);
    let a02 = Complex64::new(-0.4, 0.3);
    let a12 = Complex64::new(0.9, -0.1);
    let sparse = SparseOperator::new(
        space.clone(),
        space.clone(),
        vec![
            (0, 0, Complex64::new(1.0, 0.0)),
            (1, 1, Complex64::new(-0.5, 0.0)),
            (2, 2, Complex64::new(2.0, 0.0)),
            (0, 1, a01),
            (1, 0, a01.conj()),
            (0, 2, a02),
            (2, 0, a02.conj()),
            (1, 2, a12),
            (2, 1, a12.conj()),
        ],
    )
    .unwrap();
    let dense = sparse.to_dense().unwrap();
    let initial = Ket::new(
        space,
        vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(0.5, -0.75),
            Complex64::new(-0.25, 0.1),
        ],
    )
    .unwrap();
    let time = 0.73;
    let evolved_sparse = sparse.evolve_ket(&initial, time, 1e-11, 3).unwrap();
    let evolved_dense = dense
        .evolution_operator(time, 1e-12, 64)
        .unwrap()
        .apply(&initial)
        .unwrap();
    for (actual, expected) in evolved_sparse
        .amplitudes()
        .iter()
        .zip(evolved_dense.amplitudes())
    {
        assert!((*actual - *expected).abs() < 1e-10);
    }
    near(evolved_sparse.norm_sqr(), initial.norm_sqr(), 1e-10);

    assert_eq!(
        sparse.evolve_ket(&initial, time, 1e-14, 1).unwrap_err(),
        LinearError::EigenDecompositionNoConvergence
    );
}

#[test]
fn adaptive_sparse_evolution_tracks_a_time_dependent_hamiltonian() {
    let space = HilbertSpace::qubits(1).unwrap();
    let amplitude = std::f64::consts::FRAC_1_SQRT_2;
    let initial = Ket::new(
        space.clone(),
        vec![
            Complex64::new(amplitude, 0.0),
            Complex64::new(amplitude, 0.0),
        ],
    )
    .unwrap();
    let schedule_space = space.clone();
    let end_time = 1.3;
    let evolved = evolve_ket_time_dependent(&initial, 0.0, end_time, 1e-6, 10_000, move |time| {
        SparseOperator::new(
            schedule_space.clone(),
            schedule_space.clone(),
            vec![
                (0, 0, Complex64::new(time, 0.0)),
                (1, 1, Complex64::new(-time, 0.0)),
            ],
        )
    })
    .unwrap();

    let integrated_phase = end_time * end_time * 0.5;
    let expected_zero = Complex64::new(
        amplitude * integrated_phase.cos(),
        -amplitude * integrated_phase.sin(),
    );
    let expected_one = expected_zero.conj();
    assert!((evolved.amplitudes()[0] - expected_zero).abs() < 2e-6);
    assert!((evolved.amplitudes()[1] - expected_one).abs() < 2e-6);
    near(evolved.norm_sqr(), 1.0, 1e-12);

    // The max-steps guard must trigger non-convergence when a single step is
    // insufficient. H(t) = t*Z is solved exactly by one midpoint step, so the
    // guard check uses H(t) = t*Z + X, where a single midpoint step is not exact
    // and the adaptive error test rejects the lone step.
    assert_eq!(
        evolve_ket_time_dependent(&initial, 0.0, end_time, 1e-12, 1, move |time| {
            let schedule_space = HilbertSpace::qubits(1).unwrap();
            SparseOperator::new(
                schedule_space.clone(),
                schedule_space,
                vec![
                    (0, 0, Complex64::new(time, 0.0)),
                    (1, 1, Complex64::new(-time, 0.0)),
                    (0, 1, Complex64::ONE),
                    (1, 0, Complex64::ONE),
                ],
            )
        })
        .unwrap_err(),
        LinearError::EigenDecompositionNoConvergence
    );
}

#[test]
fn spectral_time_evolution_matches_pauli_y_analytic_solution() {
    let hamiltonian = pauli_y();
    let initial = Ket::basis(HilbertSpace::qubits(1).unwrap(), 0).unwrap();
    let time = std::f64::consts::FRAC_PI_4;
    let evolved = hamiltonian.evolve_ket(&initial, time, 1e-13, 32).unwrap();
    let expected = std::f64::consts::FRAC_1_SQRT_2;
    near(evolved.amplitudes()[0].re, expected, 1e-12);
    near(evolved.amplitudes()[0].im, 0.0, 1e-12);
    near(evolved.amplitudes()[1].re, expected, 1e-12);
    near(evolved.amplitudes()[1].im, 0.0, 1e-12);
    near(evolved.norm_sqr(), 1.0, 1e-12);
}

#[test]
fn density_matrix_evolution_preserves_trace_and_applies_unitary_phase() {
    let space = HilbertSpace::qubits(1).unwrap();
    let z = Operator::new(
        space.clone(),
        space,
        vec![
            Complex64::ONE,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::new(-1.0, 0.0),
        ],
    )
    .unwrap();
    let mut plus = StateVector::zero(1).unwrap();
    plus.apply(Gate::H, 0).unwrap();
    let mut density = DensityMatrix::from_state(&plus);
    density
        .evolve_with_hamiltonian(&z, std::f64::consts::FRAC_PI_4, 1e-12, 16)
        .unwrap();
    near(density.trace().re, 1.0, 1e-12);
    near(density.trace().im, 0.0, 1e-12);
    near(density.purity(), 1.0, 1e-12);
    near(density.entries()[1].re, 0.0, 1e-12);
    near(density.entries()[1].im, -0.5, 1e-12);
    near(density.entries()[2].re, 0.0, 1e-12);
    near(density.entries()[2].im, 0.5, 1e-12);
}

#[test]
fn density_matrix_rejects_non_unitary_updates() {
    let space = HilbertSpace::qubits(1).unwrap();
    let non_unitary = Operator::new(
        space.clone(),
        space,
        vec![
            Complex64::ONE,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
        ],
    )
    .unwrap();
    let mut density = DensityMatrix::maximally_mixed(1).unwrap();
    assert_eq!(
        density.apply_unitary(&non_unitary, 1e-12),
        Err(LinearError::NotUnitary)
    );
}

#[test]
fn eigensolver_reports_invalid_inputs_and_nonconvergence() {
    let hamiltonian = pauli_y();
    assert_eq!(
        hamiltonian.hermitian_eigensystem(f64::NAN, 10).unwrap_err(),
        LinearError::InvalidTolerance
    );
    assert_eq!(
        hamiltonian.hermitian_eigensystem(1e-12, 0).unwrap_err(),
        LinearError::EigenDecompositionNoConvergence
    );
    let space = HilbertSpace::qubits(1).unwrap();
    let diagonal = Operator::new(
        space.clone(),
        space.clone(),
        vec![
            Complex64::new(-2.0, 0.0),
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::new(3.0, 0.0),
        ],
    )
    .unwrap();
    assert_eq!(
        diagonal
            .hermitian_eigensystem(1e-12, 0)
            .unwrap()
            .eigenvalues(),
        &[-2.0, 3.0]
    );
    let not_hermitian = Operator::new(
        space.clone(),
        space,
        vec![
            Complex64::ZERO,
            Complex64::ONE,
            Complex64::ZERO,
            Complex64::ZERO,
        ],
    )
    .unwrap();
    assert_eq!(
        not_hermitian.hermitian_eigensystem(1e-12, 10).unwrap_err(),
        LinearError::NotHermitian
    );
    let observable = Observable::new(hamiltonian, 1e-12).unwrap();
    assert_eq!(
        observable.operator().evolve_ket(
            &Ket::basis(HilbertSpace::qubits(1).unwrap(), 0).unwrap(),
            f64::INFINITY,
            1e-12,
            10
        ),
        Err(LinearError::NonFiniteTime)
    );
    let huge = Operator::new(
        HilbertSpace::qubits(1).unwrap(),
        HilbertSpace::qubits(1).unwrap(),
        vec![
            Complex64::new(1e308, 0.0),
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::new(1e308, 0.0),
        ],
    )
    .unwrap();
    assert_eq!(
        huge.evolution_operator(2.0, 1e-12, 8),
        Err(LinearError::EvolutionOverflow)
    );
}

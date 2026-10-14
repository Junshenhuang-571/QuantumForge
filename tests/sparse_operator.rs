use quantumforge::quantum::{
    HilbertSpace, Ket, LinearError, LinearMap, Operator, SparseKet, SparseObservable,
    SparseOperator,
};
use quantumforge::{Complex64, Gate, SmallRng, StateVector};

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn close(a: Complex64, b: Complex64, tolerance: f64) {
    assert!((a - b).abs() <= tolerance, "{a:?} != {b:?}");
}

#[test]
fn sparse_and_dense_maps_share_ket_semantics() {
    let output = HilbertSpace::new(vec![2]).unwrap();
    let input = HilbertSpace::new(vec![3]).unwrap();
    let dense = Operator::new(
        output.clone(),
        input.clone(),
        vec![
            Complex64::ZERO,
            c(2.0, 1.0),
            Complex64::ZERO,
            c(0.0, -1.0),
            Complex64::ZERO,
            c(3.0, 0.0),
        ],
    )
    .unwrap();
    let sparse = SparseOperator::new(
        output,
        input.clone(),
        vec![
            (0, 1, c(1.0, 0.5)),
            (0, 1, c(1.0, 0.5)),
            (1, 0, c(0.0, -1.0)),
            (1, 2, c(3.0, 0.0)),
            (1, 1, Complex64::ONE),
            (1, 1, c(-1.0, 0.0)),
        ],
    )
    .unwrap();
    assert_eq!(sparse.nonzero_count(), 3);
    assert_eq!(sparse.row_offsets(), &[0, 1, 3]);
    let ket = Ket::new(input, vec![c(0.5, 0.2), c(-1.0, 0.3), c(0.1, -0.4)]).unwrap();
    let dense_result = dense.apply(&ket).unwrap();
    let sparse_result = ket.apply_map(&sparse).unwrap();
    for (a, b) in dense_result
        .amplitudes()
        .iter()
        .zip(sparse_result.amplitudes())
    {
        close(*a, *b, 1e-13);
    }
    assert_eq!(sparse.to_dense().unwrap(), dense);
}

#[test]
fn sparse_adjoint_and_composition_match_dense_algebra() {
    let space = HilbertSpace::new(vec![3]).unwrap();
    let a = Operator::new(
        space.clone(),
        space.clone(),
        vec![
            c(1.0, 0.0),
            c(0.0, 2.0),
            Complex64::ZERO,
            c(0.0, -2.0),
            c(3.0, 0.0),
            c(0.5, 0.25),
            Complex64::ZERO,
            c(0.5, -0.25),
            c(-1.0, 0.0),
        ],
    )
    .unwrap();
    let b = Operator::new(
        space.clone(),
        space.clone(),
        vec![
            Complex64::ZERO,
            Complex64::ONE,
            Complex64::ZERO,
            Complex64::ONE,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            c(2.0, 0.0),
        ],
    )
    .unwrap();
    let sparse_a = SparseOperator::from_dense(&a, 0.0).unwrap();
    let sparse_b = SparseOperator::from_dense(&b, 0.0).unwrap();

    let dense_adjoint = a.adjoint();
    let sparse_adjoint = sparse_a.adjoint();
    let input = Ket::new(space.clone(), vec![c(0.3, 0.2), c(-0.5, 0.1), c(0.8, -0.3)]).unwrap();
    let dense_adjoint_result = dense_adjoint.apply(&input).unwrap();
    let sparse_adjoint_result = input.apply_map(&sparse_adjoint).unwrap();
    for (x, y) in dense_adjoint_result
        .amplitudes()
        .iter()
        .zip(sparse_adjoint_result.amplitudes())
    {
        close(*x, *y, 1e-13);
    }

    let dense_product = a.compose(&b).unwrap();
    let sparse_product = sparse_a.compose(&sparse_b).unwrap();
    let dense_product_result = dense_product.apply(&input).unwrap();
    let sparse_product_result = input.apply_map(&sparse_product).unwrap();
    for (x, y) in dense_product_result
        .amplitudes()
        .iter()
        .zip(sparse_product_result.amplitudes())
    {
        close(*x, *y, 1e-13);
    }
}

#[test]
fn sparse_constructor_checks_bounds_and_tolerance() {
    let space = HilbertSpace::new(vec![2]).unwrap();
    assert_eq!(
        SparseOperator::new(space.clone(), space.clone(), vec![(2, 0, Complex64::ONE)])
            .unwrap_err(),
        LinearError::SparseEntryOutOfRange {
            row: 2,
            col: 0,
            rows: 2,
            cols: 2
        }
    );
    let dense = Operator::identity(space.clone());
    assert_eq!(
        SparseOperator::from_dense(&dense, f64::NAN).unwrap_err(),
        LinearError::InvalidTolerance
    );
    assert_eq!(
        Ket::basis(space, 0)
            .unwrap()
            .expectation_map(&Operator::identity(HilbertSpace::new(vec![3]).unwrap()))
            .unwrap_err(),
        LinearError::IncompatibleSpaces
    );
}

#[test]
fn sparse_observables_validate_hermiticity_and_evaluate_expectations() {
    let space = HilbertSpace::qubits(1).unwrap();
    let z = SparseOperator::new(
        space.clone(),
        space.clone(),
        vec![(0, 0, Complex64::ONE), (1, 1, c(-1.0, 0.0))],
    )
    .unwrap();
    let observable = SparseObservable::new(z, 1e-12).unwrap();
    let zero = Ket::basis(space, 0).unwrap();
    close(
        observable.expectation_ket(&zero).unwrap(),
        Complex64::ONE,
        1e-12,
    );

    let non_hermitian = SparseOperator::new(
        HilbertSpace::qubits(1).unwrap(),
        HilbertSpace::qubits(1).unwrap(),
        vec![(0, 1, Complex64::ONE)],
    )
    .unwrap();
    assert_eq!(
        SparseObservable::new(non_hermitian, 1e-12).unwrap_err(),
        LinearError::NotHermitian
    );
}

#[test]
fn sparse_lanczos_solver_returns_lowest_ritz_pairs_without_densifying_operator() {
    let space = HilbertSpace::new(vec![3]).unwrap();
    let hamiltonian = SparseOperator::new(
        space.clone(),
        space,
        vec![
            (0, 0, c(-2.0, 0.0)),
            (1, 1, c(0.5, 0.0)),
            (2, 2, c(3.0, 0.0)),
            (0, 1, c(0.2, 0.1)),
            (1, 0, c(0.2, -0.1)),
        ],
    )
    .unwrap();
    assert_eq!(hamiltonian.nonzero_count(), 5);

    let result = hamiltonian.lowest_eigenpairs(2, 1e-10, 3).unwrap();
    assert_eq!(result.eigenvalues().len(), 2);
    assert_eq!(result.eigenvectors().len(), 2);
    assert!(result.eigenvalues()[0] < result.eigenvalues()[1]);
    for (eigenvalue, eigenvector) in result.eigenvalues().iter().zip(result.eigenvectors()) {
        let image = hamiltonian
            .apply_amplitudes(eigenvector.amplitudes())
            .unwrap();
        let residual = image
            .iter()
            .zip(eigenvector.amplitudes())
            .map(|(actual, amplitude)| (*actual - *amplitude * *eigenvalue).norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(residual < 1e-9, "eigenvector residual {residual}");
    }

    assert_eq!(
        hamiltonian.lowest_eigenpairs(4, 1e-10, 4).unwrap_err(),
        LinearError::InvalidEigenpairCount {
            requested: 4,
            dimension: 3
        }
    );
    assert_eq!(
        hamiltonian.lowest_eigenpairs(1, 1e-10, 0).unwrap_err(),
        LinearError::EigenDecompositionNoConvergence
    );

    let degenerate_identity = SparseOperator::new(
        HilbertSpace::new(vec![3]).unwrap(),
        HilbertSpace::new(vec![3]).unwrap(),
        vec![
            (0, 0, Complex64::ONE),
            (1, 1, Complex64::ONE),
            (2, 2, Complex64::ONE),
        ],
    )
    .unwrap();
    assert_eq!(
        degenerate_identity
            .lowest_eigenpairs(2, 1e-10, 3)
            .unwrap_err(),
        LinearError::EigenDecompositionNoConvergence
    );
}

#[test]
fn sparse_ket_preserves_gate_circuit_semantics_and_measurement() {
    let space = HilbertSpace::qubits(4).unwrap();
    let mut sparse = SparseKet::basis(space, 0).unwrap();
    let mut dense = StateVector::zero(4).unwrap();
    sparse.apply_gate(Gate::H, 0).unwrap();
    dense.apply(Gate::H, 0).unwrap();
    sparse.apply_cnot(0, 3).unwrap();
    dense.apply_cnot(0, 3).unwrap();
    sparse.apply_gate(Gate::Ry(-0.42), 2).unwrap();
    dense.apply(Gate::Ry(-0.42), 2).unwrap();
    sparse.apply_cz(2, 3).unwrap();
    dense.apply_cz(2, 3).unwrap();
    sparse.apply_swap(1, 3).unwrap();
    dense.apply_swap(1, 3).unwrap();
    assert!(sparse.nonzero_count() <= 4);
    for (a, b) in sparse
        .to_ket()
        .unwrap()
        .amplitudes()
        .iter()
        .zip(dense.amplitudes())
    {
        close(*a, *b, 1e-13);
    }
    for qubit in 0..4 {
        assert!(
            (sparse.expectation_z(qubit).unwrap() - dense.expectation_z(qubit).unwrap()).abs()
                < 1e-13
        );
    }

    let mut sparse_rng = SmallRng::seeded(0x1234);
    let mut dense_rng = SmallRng::seeded(0x1234);
    assert_eq!(
        sparse.measure(2, &mut sparse_rng).unwrap(),
        dense.measure(2, &mut dense_rng).unwrap()
    );
    assert!((sparse.norm_sqr() - 1.0).abs() < 1e-13);
    for (a, b) in sparse
        .to_ket()
        .unwrap()
        .amplitudes()
        .iter()
        .zip(dense.amplitudes())
    {
        close(*a, *b, 1e-13);
    }
}

#[test]
fn sparse_ket_applies_sparse_maps_and_tensors_without_densifying() {
    let qubit = HilbertSpace::qubits(1).unwrap();
    let bit_flip = SparseOperator::new(
        qubit.clone(),
        qubit.clone(),
        vec![(0, 1, Complex64::ONE), (1, 0, Complex64::ONE)],
    )
    .unwrap();
    let zero = SparseKet::basis(qubit.clone(), 0).unwrap();
    let one = zero.apply_sparse(&bit_flip).unwrap();
    assert_eq!(one.amplitude(1).unwrap(), Complex64::ONE);
    assert_eq!(one.amplitude(0).unwrap(), Complex64::ZERO);

    let two_qubit = one.tensor(&one).unwrap();
    assert_eq!(two_qubit.space().dimension(), 4);
    assert_eq!(two_qubit.nonzero_count(), 1);
    assert_eq!(two_qubit.amplitude(3).unwrap(), Complex64::ONE);
}

#[test]
fn sparse_ket_observables_match_dense_semantics_for_unnormalized_vectors() {
    let space = HilbertSpace::qubits(1).unwrap();
    let sparse = SparseKet::new(space, [(0, c(2.0, 0.0))]).unwrap();
    assert_eq!(sparse.probability_one(0).unwrap(), 0.0);
    assert_eq!(sparse.expectation_z(0).unwrap(), 4.0);
}

#[test]
fn sparse_ket_matches_dense_for_the_full_single_qubit_gate_set() {
    let space = HilbertSpace::qubits(1).unwrap();
    let mut sparse = SparseKet::basis(space, 1).unwrap();
    let mut dense = StateVector::zero(1).unwrap();
    dense.apply(Gate::X, 0).unwrap();
    let gates = [
        Gate::I,
        Gate::X,
        Gate::Y,
        Gate::Z,
        Gate::H,
        Gate::Rx(0.37),
        Gate::Ry(-1.13),
        Gate::Rz(0.82),
    ];
    for gate in gates {
        sparse.apply_gate(gate, 0).unwrap();
        dense.apply(gate, 0).unwrap();
        for (a, b) in sparse
            .to_ket()
            .unwrap()
            .amplitudes()
            .iter()
            .zip(dense.amplitudes())
        {
            close(*a, *b, 1e-13);
        }
    }
    assert_eq!(
        sparse.apply_gate(Gate::H, 1),
        Err(LinearError::QubitOutOfRange {
            qubit: 1,
            qubits: 1
        })
    );
    assert_eq!(
        sparse.apply_cnot(0, 0),
        Err(LinearError::InvalidQubitPair {
            first: 0,
            second: 0
        })
    );
}

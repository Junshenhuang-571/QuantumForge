//! Tests for SubsCoRe: the VQE kernel, the GP surrogate, shot allocation and
//! the full adaptive loop.

use quantumforge::analytic::{ExactComplex, Pauli, PauliSum, PauliWord, Rational};
use quantumforge::subscore::{
    GaussianProcess, ReadoutStrategy, SearchOptions, ShotHamiltonian, SubsCoReOptions, axis_points,
    choose_shots_bound, choose_shots_center, run_subscore, shots_for_variance, vqe_kernel,
};
use quantumforge::{ParameterizedCircuit, RotationAxis};

fn real(value: i128) -> ExactComplex {
    ExactComplex::real(Rational::new(value, 1).unwrap())
}

/// A 2-layer hardware-efficient ansatz on `qubits` qubits with `2 * qubits`
/// parameters per layer.
fn ansatz(layers: usize, qubits: usize) -> (ParameterizedCircuit, Vec<f64>) {
    let mut circuit = ParameterizedCircuit::new(layers * 2 * qubits);
    let mut params = Vec::new();
    for _ in 0..layers {
        for qubit in 0..qubits {
            circuit.rotation(RotationAxis::Y, qubit, params.len());
            params.push(0.3);
        }
        for qubit in 0..qubits {
            circuit.rotation(RotationAxis::X, qubit, params.len());
            params.push(-0.2);
        }
        for qubit in 0..qubits - 1 {
            circuit.cnot(qubit, qubit + 1);
        }
    }
    (circuit, params)
}

/// Transverse-field Ising Hamiltonian `-sum ZZ - sum X`.
fn tfim(qubits: usize) -> PauliSum {
    let mut sum = PauliSum::zero(qubits);
    for qubit in 0..qubits - 1 {
        let mut word = vec![Pauli::I; qubits];
        word[qubit] = Pauli::Z;
        word[qubit + 1] = Pauli::Z;
        let _ = sum.add_term(real(-1), PauliWord::new(word));
    }
    for qubit in 0..qubits {
        let mut word = vec![Pauli::I; qubits];
        word[qubit] = Pauli::X;
        let _ = sum.add_term(real(-1), PauliWord::new(word));
    }
    sum
}

#[test]
fn kernel_is_symmetric_and_has_unit_diagonal() {
    let counts = vec![1, 2];
    let x = [0.3, 1.1];
    let y = [2.0, 0.4];
    let kxx = vqe_kernel(&x, &x, 2.0, 3.0, &counts);
    assert!(
        (kxx - 3.0).abs() < 1e-12,
        "k(x,x) must equal sigma0^2, got {kxx}"
    );
    let kxy = vqe_kernel(&x, &y, 2.0, 3.0, &counts);
    let kyx = vqe_kernel(&y, &x, 2.0, 3.0, &counts);
    assert!((kxy - kyx).abs() < 1e-12, "kernel must be symmetric");

    // A parameter with V_d = 0 contributes nothing: the kernel then depends
    // only on the other coordinate.
    let a = vqe_kernel(&[0.3, 1.1], &[2.0, 0.4], 2.0, 3.0, &[0, 2]);
    let b = vqe_kernel(&[1.7, 1.1], &[0.9, 0.4], 2.0, 3.0, &[0, 2]);
    assert!(
        (a - b).abs() < 1e-12,
        "an unused parameter must not affect the kernel"
    );
}

#[test]
fn gp_interpolates_and_shrinks_variance() {
    let counts = vec![1];
    let mut gp = GaussianProcess::new(2.0, 1.0, counts);
    assert_eq!(gp.len(), 0);
    assert!((gp.posterior_variance(&[0.7]) - 1.0).abs() < 1e-12);

    // A deterministic function of the kernel's feature space.
    gp.push(vec![0.0], 0.5, 1e-6);
    gp.push(vec![1.0], -0.25, 1e-6);
    gp.push(vec![2.0], 0.5, 1e-6);

    assert_eq!(gp.len(), 3);
    for (x, expected) in [(0.0, 0.5), (1.0, -0.25), (2.0, 0.5)] {
        let (mean, variance) = gp.posterior_mean_and_variance(&[x]);
        assert!(
            (mean - expected).abs() < 1e-4,
            "posterior mean at {x} should interpolate: {mean} vs {expected}"
        );
        assert!(
            variance < 1e-5,
            "variance at a trained point should collapse"
        );
    }
    // With V_d = 1 the three observations determine the whole 1D trigonometric
    // polynomial, so the posterior collapses across the entire subspace — that
    // is exactly the confident region covering the axis.
    assert!(
        gp.posterior_variance(&[0.5]) < 1e-4,
        "three equidistant observations should identify the subspace"
    );
}

/// Naive Gaussian elimination, written independently of the crate's GP code so
/// the posterior it produces is a genuine cross-check.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for column in 0..n {
        for row in (column + 1)..n {
            let factor = a[row][column] / a[column][column];
            let pivot_row: Vec<f64> = a[column].clone();
            for (c, pivot) in pivot_row.into_iter().enumerate().skip(column) {
                a[row][c] -= factor * pivot;
            }
            b[row] -= factor * b[column];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let tail: f64 = a[row]
            .iter()
            .zip(&x)
            .skip(row + 1)
            .map(|(entry, solved)| entry * solved)
            .sum();
        x[row] = (b[row] - tail) / a[row][row];
    }
    x
}

#[test]
fn gp_matches_a_direct_dense_computation() {
    let counts = vec![1, 1];
    let mut gp = GaussianProcess::new(2.0, 1.0, counts.clone());
    let points = [vec![0.2, 0.7], vec![1.3, 2.1], vec![2.6, 0.4]];
    let values = [0.4, -0.9, 1.2];
    let variances = [0.01, 0.02, 0.03];
    for ((point, value), variance) in points.iter().zip(values).zip(variances) {
        gp.push(point.clone(), value, variance);
    }

    let test = [0.9, 1.7];
    let mut kernel = vec![vec![0.0; 3]; 3];
    for (i, point) in points.iter().enumerate() {
        for (j, other) in points.iter().enumerate() {
            let mut entry = vqe_kernel(point, other, 2.0, 1.0, &counts);
            if i == j {
                entry += variances[i]; // heteroscedastic noise on the diagonal
            }
            kernel[i][j] = entry;
        }
    }

    let alpha = solve(kernel.clone(), values.to_vec());
    let kernel_test: Vec<f64> = points
        .iter()
        .map(|point| vqe_kernel(&test, point, 2.0, 1.0, &counts))
        .collect();
    let dense_mean: f64 = alpha.iter().zip(&kernel_test).map(|(a, k)| a * k).sum();

    let projected = solve(kernel, kernel_test.clone());
    let reduction: f64 = kernel_test.iter().zip(&projected).map(|(a, b)| a * b).sum();
    let dense_variance = 1.0 - reduction;

    let (mean, variance) = gp.posterior_mean_and_variance(&test);
    assert!(
        (mean - dense_mean).abs() < 1e-9,
        "mean {mean} vs dense {dense_mean}"
    );
    assert!(
        (variance - dense_variance).abs() < 1e-9,
        "variance {variance} vs dense {dense_variance}"
    );
}

#[test]
fn axis_points_are_equidistant_and_include_the_best() {
    let counts = vec![1, 3];
    let best = [0.5, 1.0];
    let points = axis_points(&best, 0, &counts);
    assert_eq!(points.len(), 3, "1 + 2 V_d with V_d = 1");
    assert_eq!(points[0], best, "the first point is the current best");
    for (index, point) in points.iter().enumerate() {
        assert_eq!(point[1], best[1], "other coordinates must not move");
        let expected =
            (best[0] + std::f64::consts::TAU * index as f64 / 3.0) % std::f64::consts::TAU;
        assert!((point[0] - expected).abs() < 1e-12, "point {index}");
    }
    assert_eq!(axis_points(&best, 1, &counts).len(), 7, "1 + 2*3");
}

#[test]
fn shot_budget_arithmetic() {
    assert_eq!(shots_for_variance(1.0, 0.25), 4);
    assert_eq!(shots_for_variance(1.0, 1.0), 1);
    assert_eq!(shots_for_variance(1.0, 0.0), 1, "degenerate variance");
    assert_eq!(choose_shots_bound(0.01, 1.0, 3), vec![100, 100, 100]);
}

#[test]
fn center_allocation_spends_fewer_shots_at_the_center() {
    let counts = vec![1];
    let mut gp = GaussianProcess::new(2.0, 1.0, counts.clone());
    // Accurately sample the current best point so the GP already knows it.
    gp.push(vec![0.0], 1.0, 1e-8);
    let base = [0.0];
    let points = axis_points(&base, 0, &counts);
    let shots = choose_shots_center(
        &gp,
        &base,
        &points,
        0,
        1.0 / 16.0,
        1.0,
        &SearchOptions {
            shot_grid: 16,
            scan_grid: 36,
        },
    );
    assert_eq!(shots.len(), 3);
    assert!(
        shots[0] <= shots[1] && shots[0] <= shots[2],
        "the well-known center should not need more shots than the shifts: {shots:?}"
    );
    let bound = choose_shots_bound(1.0 / 16.0, 1.0, 3);
    let center_total: usize = shots.iter().sum();
    assert!(
        center_total <= bound.iter().sum::<usize>(),
        "SubsCoRe-Center should not exceed the Bound allocation"
    );
}

#[test]
fn shot_objective_tracks_the_exact_energy() {
    let qubits = 4;
    let (circuit, params) = ansatz(2, qubits);
    let objective = ShotHamiltonian::new(circuit, tfim(qubits), qubits);
    let counts = objective.parameter_counts();
    assert_eq!(counts.len(), 2 * 2 * qubits);
    assert!(
        counts.iter().all(|&c| c == 1),
        "each parameter drives one gate"
    );

    let exact = objective.exact_energy(&params).unwrap();
    // A large shot count should land close to the exact value.
    let estimate = objective.observe(&params, 20_000, 7);
    assert!(
        (estimate - exact).abs() < 0.1,
        "shot estimate {estimate} should approach the exact energy {exact}"
    );
}

#[test]
fn subscore_lowers_the_energy_within_budget() {
    // Shot noise makes this a budget-limited problem: the paper uses ~10^6
    // shots, and a few thousand shots buy only a handful of subspace steps.
    let qubits = 3;
    let (circuit, params) = ansatz(1, qubits);
    let objective = ShotHamiltonian::new(circuit, tfim(qubits), qubits);
    let counts = objective.parameter_counts();
    let start = objective.exact_energy(&params).unwrap();

    let options = SubsCoReOptions {
        total_shot_budget: 20_000,
        initial_shots: 128,
        search: SearchOptions {
            shot_grid: 24,
            scan_grid: 72,
        },
        strategy: ReadoutStrategy::Center,
        seed: 3,
        ..SubsCoReOptions::default()
    };
    let outcome = run_subscore(
        |point, shots, seed| objective.observe(point, shots, seed),
        &params,
        &counts,
        &options,
    );

    assert!(outcome.shots_used <= options.total_shot_budget);
    assert!(outcome.iterations > 0);
    assert_eq!(outcome.shots_per_iteration.len(), outcome.iterations);
    let final_energy = objective.exact_energy(&outcome.best).unwrap();
    assert!(
        final_energy < start - 0.5,
        "SubsCoRe should lower the exact energy: {final_energy} vs start {start}"
    );
    // The GP mean tracks the true energy at the reported minimizer.
    assert!(
        (outcome.best_value - final_energy).abs() < 0.3,
        "GP mean {} should track the exact energy {final_energy}",
        outcome.best_value
    );
}

#[test]
fn bound_strategy_runs() {
    let qubits = 3;
    let (circuit, params) = ansatz(1, qubits);
    let objective = ShotHamiltonian::new(circuit, tfim(qubits), qubits);
    let counts = objective.parameter_counts();
    let options = SubsCoReOptions {
        total_shot_budget: 1_500,
        search: SearchOptions {
            shot_grid: 8,
            scan_grid: 24,
        },
        strategy: ReadoutStrategy::Bound,
        ..SubsCoReOptions::default()
    };
    let outcome = run_subscore(
        |point, shots, seed| objective.observe(point, shots, seed),
        &params,
        &counts,
        &options,
    );
    assert!(outcome.shots_used <= options.total_shot_budget);
    assert!(outcome.iterations > 0);
}

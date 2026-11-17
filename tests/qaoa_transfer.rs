//! Tests for the parameter-transfer + targeted-single-layer QAOA pipeline.

use quantumforge::mps_qaoa::MaxCutProblem;
use quantumforge::qaoa_transfer::{
    QaoaParameters, TargetedLayerOptions, TransferOptions, adagrad, maxcut_expectation,
    nelder_mead, optimize_targeted_layer, run_parameter_transfer, select_target_layer_majority,
    targeted_layer_objective, tqa_initialization,
};
use quantumforge::{Gate, StateVector};

fn donor() -> MaxCutProblem {
    MaxCutProblem::random_3_regular(8, 7)
}

fn acceptor() -> MaxCutProblem {
    MaxCutProblem::random_3_regular(10, 11)
}

#[test]
fn tqa_schedule_matches_equation_one() {
    let params = tqa_initialization(4, 0.75);
    for layer in 0..4 {
        let fraction = layer as f64 / 4.0;
        assert!((params.gammas[layer] - fraction * 0.75).abs() < 1e-12);
        assert!((params.betas[layer] - (1.0 - fraction) * 0.75).abs() < 1e-12);
    }
    // Cost angles increase, mixer angles decrease.
    assert!(params.gammas[0] < params.gammas[3]);
    assert!(params.betas[0] > params.betas[3]);
}

#[test]
fn flat_roundtrip_and_transfer() {
    let params = QaoaParameters {
        gammas: vec![0.1, 0.2, 0.3],
        betas: vec![0.9, 0.8, 0.7],
    };
    let flat = params.as_flat();
    assert_eq!(flat, vec![0.1, 0.2, 0.3, 0.9, 0.8, 0.7]);
    assert_eq!(QaoaParameters::from_flat(&flat), params);

    let grown = params.transferred(5);
    assert_eq!(grown.depth(), 5);
    assert_eq!(grown.gammas[3], 0.3, "growth repeats the last layer");
    assert_eq!(grown.betas[4], 0.7);

    let shrunk = params.transferred(2);
    assert_eq!(shrunk.gammas, vec![0.1, 0.2]);
    assert_eq!(shrunk.betas, vec![0.9, 0.8]);
}

#[test]
fn maxcut_expectation_matches_brute_force() {
    let problem = acceptor();
    // All-zero state: every edge is uncut, so the expectation is zero.
    let zero = StateVector::zero(10).unwrap();
    assert!(maxcut_expectation(&zero, &problem.edges).abs() < 1e-12);

    // Equal superposition over every basis state: P(differ) = 1/2 per edge.
    let mut circuit = quantumforge::Circuit::new();
    for qubit in 0..10 {
        circuit.push(Gate::H, qubit);
    }
    let uniform = circuit.run(zero).unwrap();
    let expected = 0.5 * problem.total_weight();
    assert!(
        (maxcut_expectation(&uniform, &problem.edges) - expected).abs() < 1e-9,
        "uniform superposition should cut half the total weight"
    );
}

#[test]
fn nelder_mead_minimizes_a_quadratic() {
    let quadratic = |point: &[f64]| (point[0] - 1.5).powi(2) + (point[1] + 2.0).powi(2);
    let (best, value) = nelder_mead(&quadratic, &[0.0, 0.0], 1e-12, 500);
    // Convergence is on the simplex spread, so the residual is small rather than
    // exactly zero.
    assert!(
        value < 1e-10,
        "simplex should reach the minimum, got {value}"
    );
    assert!((best[0] - 1.5).abs() < 1e-4);
    assert!((best[1] + 2.0).abs() < 1e-4);
}

#[test]
fn adagrad_descends_a_quadratic() {
    let quadratic = |point: &[f64]| point[0].powi(2) + point[1].powi(2);
    let best = adagrad(&quadratic, &[3.0, -4.0], 0.5, 1e-8, 200);
    assert!(best[0].abs() < 1e-3, "x did not converge: {}", best[0]);
    assert!(best[1].abs() < 1e-3, "y did not converge: {}", best[1]);
}

#[test]
fn targeted_objective_penalizes_large_angles() {
    let problem = acceptor();
    let params = QaoaParameters {
        gammas: vec![0.3, 0.1],
        betas: vec![0.4, 0.2],
    };
    // Same angles, two strengths: the difference is exactly the ridge penalty on
    // layer 0.
    let unpenalized = targeted_layer_objective(&problem, &params, 0, 0.0).unwrap();
    let penalized = targeted_layer_objective(&problem, &params, 0, 1e-4).unwrap();
    let penalty = 1e-4 * (0.3_f64.powi(2) + 0.4_f64.powi(2));
    assert!(
        (penalized - unpenalized - penalty).abs() < 1e-12,
        "objective must increase by exactly the ridge penalty: {penalized} - {unpenalized} vs {penalty}"
    );
}

#[test]
fn targeted_layer_improves_on_transferred_angles() {
    let acceptor = acceptor();
    let options = TransferOptions {
        depth: 3,
        donor_iterations: 15,
        targeted: TargetedLayerOptions {
            max_iterations: 120,
            ..TargetedLayerOptions::default()
        },
        ..TransferOptions::default()
    };

    let outcome = run_parameter_transfer(&donor(), &acceptor, &options).unwrap();
    assert!(
        outcome.energy
            >= outcome
                .layer_energies
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
                - 1e-9,
        "the reported energy must match the best layer tried"
    );
    assert_eq!(outcome.target_layer, {
        let mut best = 0;
        for (layer, &energy) in outcome.layer_energies.iter().enumerate() {
            if energy > outcome.layer_energies[best] {
                best = layer;
            }
        }
        best
    });
    assert!(
        outcome.transferred_ratio > 0.0 && outcome.approximation_ratio > 0.0,
        "ratios must be positive"
    );
    // The targeted step must not make the cut worse than the transferred angles.
    assert!(
        outcome.approximation_ratio + 1e-9 >= outcome.transferred_ratio,
        "targeted optimization degraded the cut: r_s={} r_n={}",
        outcome.approximation_ratio,
        outcome.transferred_ratio
    );
}

#[test]
fn optimizing_a_single_layer_moves_only_that_layer() {
    let problem = acceptor();
    let params = tqa_initialization(4, 0.75);
    let options = TargetedLayerOptions {
        lambda: 1e-4,
        tolerance: 1e-10,
        max_iterations: 150,
    };
    let (optimized, _) = optimize_targeted_layer(&problem, &params, 2, &options).unwrap();
    for layer in 0..4 {
        if layer == 2 {
            continue;
        }
        assert_eq!(
            optimized.gammas[layer], params.gammas[layer],
            "layer {layer} must be untouched"
        );
        assert_eq!(optimized.betas[layer], params.betas[layer]);
    }
}

#[test]
fn majority_vote_counts_every_instance() {
    let problems = vec![
        MaxCutProblem::random_3_regular(8, 1),
        MaxCutProblem::random_3_regular(8, 2),
        MaxCutProblem::random_3_regular(8, 3),
    ];
    let params = tqa_initialization(3, 0.75);
    let options = TargetedLayerOptions {
        lambda: 1e-4,
        tolerance: 1e-8,
        max_iterations: 80,
    };
    let (layer, votes) = select_target_layer_majority(&problems, &params, &options).unwrap();
    assert_eq!(votes.len(), 3);
    assert_eq!(votes.iter().sum::<usize>(), problems.len());
    assert!(layer < 3);
    assert_eq!(*votes.iter().max().unwrap(), votes[layer]);
}

#[test]
fn out_of_range_layer_is_an_error() {
    let problem = acceptor();
    let params = QaoaParameters::zeros(2);
    let options = TargetedLayerOptions::default();
    let error = optimize_targeted_layer(&problem, &params, 5, &options).unwrap_err();
    assert!(format!("{error}").contains("out of range"));
    assert!(targeted_layer_objective(&problem, &params, 5, 1e-4).is_err());
}

#[test]
fn donor_must_not_exceed_acceptor() {
    let options = TransferOptions::default();
    let error = run_parameter_transfer(&acceptor(), &donor(), &options).unwrap_err();
    assert!(format!("{error}").contains("donor"));
}

#[test]
fn empty_problem_is_rejected() {
    let empty = MaxCutProblem::new(4, Vec::new());
    let options = TransferOptions::default();
    let error = run_parameter_transfer(&empty, &empty, &options).unwrap_err();
    assert_eq!(format!("{error}"), "the problem has no edges");
}

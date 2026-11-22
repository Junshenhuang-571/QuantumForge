//! Readout confusion matrices and mitigation.

use quantumforge::ShotResult;
use quantumforge::readout::{
    ConfusionMatrix, ReadoutError, distribution_from_counts, mitigate_distribution,
};
use std::collections::BTreeMap;

#[test]
fn symmetric_noise_mitigation_recovers_the_distribution() {
    let matrix = ConfusionMatrix::symmetric(0.1).unwrap();
    let prepared = [0.8, 0.2];
    let observed = matrix.apply(prepared);
    // Forward: (0.8*0.9 + 0.2*0.1, 0.8*0.1 + 0.2*0.9) = (0.74, 0.26).
    assert!((observed[0] - 0.74).abs() < 1e-12);
    let mitigated = matrix.mitigate(observed).unwrap();
    assert!((mitigated[0] - prepared[0]).abs() < 1e-12);
    assert!((mitigated[1] - prepared[1]).abs() < 1e-12);
}

#[test]
fn unfold_recovers_the_distribution_and_stays_positive() {
    let matrix = ConfusionMatrix::symmetric(0.1).unwrap();
    let prepared = [0.8, 0.2];
    let observed = matrix.apply(prepared);
    let unfolded = matrix.unfold(observed, 50);
    assert!((unfolded[0] - prepared[0]).abs() < 1e-6);
    assert!(unfolded[0] > 0.0 && unfolded[1] > 0.0);
}

#[test]
fn full_distribution_mitigation_two_qubits() {
    let first = ConfusionMatrix::symmetric(0.1).unwrap();
    let second = ConfusionMatrix::new([[0.95, 0.02], [0.05, 0.98]]).unwrap();
    // Prepared product distribution: p(0) = 0.7, p(1) = 0.3 on qubit 0 and
    // p(0) = 0.4, p(1) = 0.6 on qubit 1.
    let mut distribution = vec![0.0f64; 4];
    let marginals = [[0.7, 0.3], [0.4, 0.6]];
    for (index, entry) in distribution.iter_mut().enumerate() {
        let bit0 = index & 1;
        let bit1 = (index >> 1) & 1;
        *entry = marginals[0][bit0] * marginals[1][bit1];
    }
    let prepared = distribution.clone();
    // Forward: apply each single-qubit channel.
    for (qubit, matrix) in [first, second].iter().enumerate() {
        let mask = 1usize << qubit;
        for base in 0..4 {
            if base & mask == 0 {
                let zero = distribution[base];
                let one = distribution[base | mask];
                distribution[base] = matrix.matrix()[0][0] * zero + matrix.matrix()[0][1] * one;
                distribution[base | mask] =
                    matrix.matrix()[1][0] * zero + matrix.matrix()[1][1] * one;
            }
        }
    }
    let observed = distribution.clone();
    mitigate_distribution(&mut distribution, &[first, second]).unwrap();
    for (recovered, expected) in distribution.iter().zip(prepared.iter()) {
        assert!((recovered - expected).abs() < 1e-12);
    }
    assert!((observed[0] - prepared[0]).abs() > 1e-6);
}

#[test]
fn shot_result_mitigate_readout() {
    let mut counts = BTreeMap::new();
    counts.insert(vec![0u8], 740);
    counts.insert(vec![1u8], 260);
    let result = ShotResult {
        shots: 1000,
        counts,
    };
    let matrix = ConfusionMatrix::symmetric(0.1).unwrap();
    let mitigated = result.mitigate_readout(&[matrix]).unwrap();
    assert!((mitigated[0] - 0.8).abs() < 1e-12);
    assert!((mitigated[1] - 0.2).abs() < 1e-12);
}

#[test]
fn distribution_from_counts_ignores_malformed_outcomes() {
    let mut counts = BTreeMap::new();
    counts.insert(vec![0u8], 10);
    counts.insert(vec![1u8], 30);
    counts.insert(vec![0u8, 1u8], 500); // wrong length, ignored
    let distribution = distribution_from_counts(&counts, 1, 40);
    assert!((distribution[0] - 0.25).abs() < 1e-12);
    assert!((distribution[1] - 0.75).abs() < 1e-12);
}

#[test]
fn validation_rejects_bad_matrices() {
    assert!(matches!(
        ConfusionMatrix::new([[1.1, 0.0], [0.0, 1.0]]),
        Err(ReadoutError::InvalidEntry(_))
    ));
    assert!(matches!(
        ConfusionMatrix::new([[0.5, 0.0], [0.4, 1.0]]),
        Err(ReadoutError::ColumnSum { column: 0, .. })
    ));
    let singular = ConfusionMatrix::new([[1.0, 1.0], [0.0, 0.0]]).unwrap();
    assert!(matches!(singular.inverse(), Err(ReadoutError::Singular)));
}

#[test]
fn unfold_matches_the_inverse_for_an_asymmetric_channel() {
    // Columns sum to one but the matrix is not symmetric, so the EM update
    // cannot be replaced by a diagonal multiplicative shortcut.
    let matrix = ConfusionMatrix::new([[0.9, 0.05], [0.1, 0.95]]).unwrap();
    let prepared = [0.7, 0.3];
    let observed = matrix.apply(prepared);
    let exact = matrix.mitigate(observed).unwrap();
    let unfolded = matrix.unfold(observed, 200);
    assert!(
        (unfolded[0] - exact[0]).abs() < 1e-6,
        "unfold {unfolded:?} vs inverse {exact:?}"
    );
    assert!((unfolded[0] - prepared[0]).abs() < 1e-6);
    assert!(unfolded[0] > 0.0 && unfolded[1] > 0.0);
}

#[test]
fn mitigated_parity_expectation_recovers_the_exact_value() {
    use quantumforge::readout::mitigated_parity_expectation;
    use quantumforge::{Circuit, Gate, SmallRng, StateVector};

    let theta: f64 = 0.7;
    let exact = theta.cos();
    let shots = 200_000;
    let mut circuit = Circuit::new();
    circuit.push(Gate::Ry(theta), 0);
    circuit.measure(0, 0);
    let counts = circuit
        .run_shots(&StateVector::zero(1).unwrap(), shots, 5)
        .unwrap()
        .counts;

    let matrix = ConfusionMatrix::symmetric(0.1).unwrap();
    let mut rng = SmallRng::seeded(11);
    let mitigated =
        mitigated_parity_expectation(&counts, &[0], &[matrix], shots, &mut rng).unwrap();
    assert!(
        (mitigated - exact).abs() < 0.02,
        "mitigated {mitigated} vs exact {exact}"
    );

    // Corrupting the same counts without mitigating scales <Z> by (1 - 2 eps).
    let mut rng = SmallRng::seeded(11);
    let corrupted = quantumforge::readout::apply_readout_error(&counts, &[matrix], &mut rng);
    let corrupted_z =
        1.0 - 2.0 * corrupted.get(&vec![1u8]).copied().unwrap_or(0) as f64 / shots as f64;
    assert!(
        (corrupted_z - 0.8 * exact).abs() < 0.02,
        "corrupted {corrupted_z}"
    );
    assert!((mitigated - exact).abs() < (corrupted_z - exact).abs());
}

//! Verification of QFT and QPE against independent references.
//!
//! The DFT oracle below is computed directly from `exp(2*pi*i*x*y/N)`; it does
//! not use any circuit primitive. The QPE oracle uses the known eigenphase of
//! `Rz(theta)|1> = exp(i*theta/2)|1>`.

use quantumforge::algorithms::{qft_circuit, qpe_circuit};
use quantumforge::{Circuit, Complex64, Gate, StateVector};

fn dft_reference(n: usize, input: usize) -> Vec<Complex64> {
    let dimension = 1 << n;
    let angle = 2.0 * std::f64::consts::PI / dimension as f64;
    let root = Complex64::new(angle.cos(), angle.sin());
    let scale = Complex64::new(1.0 / (dimension as f64).sqrt(), 0.0);
    (0..dimension)
        .map(|output| {
            let mut value = Complex64::ONE;
            for _ in 0..(input * output % dimension) {
                value = value * root;
            }
            value * scale
        })
        .collect()
}

#[test]
fn qft_matches_the_dft_up_to_a_global_phase() {
    let n = 3usize;
    let qft = qft_circuit(n);
    for input in 0..(1 << n) {
        let mut circuit = Circuit::new();
        for qubit in 0..n {
            if (input >> qubit) & 1 == 1 {
                circuit.push(Gate::X, qubit);
            }
        }
        for operation in qft.operations() {
            circuit.operations_mut().push(*operation);
        }
        let state = circuit.run(StateVector::zero(n).unwrap()).unwrap();
        let amplitudes = state.amplitudes();
        let expected = dft_reference(n, input);

        let pivot = (0..expected.len())
            .max_by(|a, b| {
                expected[*a]
                    .norm_sqr()
                    .partial_cmp(&expected[*b].norm_sqr())
                    .unwrap()
            })
            .unwrap();
        let phase = amplitudes[pivot] * expected[pivot].conj() / expected[pivot].norm_sqr();
        let residual: f64 = (0..expected.len())
            .map(|i| (amplitudes[i] - phase * expected[i]).norm_sqr())
            .sum();
        assert!(
            residual < 1e-20,
            "input {input:03b}: DFT residual {residual}"
        );
    }
}

fn phase_register_masses(deg: f64, precision: usize) -> Vec<f64> {
    let mut unitary = Circuit::new();
    unitary.push(Gate::Rz(deg.to_radians()), 0);
    let circuit = qpe_circuit(unitary, precision);
    let mut prep = Circuit::new();
    prep.push(Gate::X, precision); // target eigenstate |1>
    for operation in circuit.operations() {
        prep.operations_mut().push(*operation);
    }
    let state = prep.run(StateVector::zero(precision + 1).unwrap()).unwrap();
    let probabilities: Vec<f64> = state
        .amplitudes()
        .iter()
        .map(|amplitude| amplitude.norm_sqr())
        .collect();
    (0..(1 << precision))
        .map(|outcome| {
            (0..(1 << (precision + 1)))
                .filter(|index| index & ((1 << precision) - 1) == outcome)
                .map(|index| probabilities[index])
                .sum()
        })
        .collect()
}

#[test]
fn qpe_recovers_representable_eigenphases() {
    let precision = 3usize;
    for deg in [0.0f64, 90.0, 180.0, 270.0] {
        let masses = phase_register_masses(deg, precision);
        // Rz(theta)|1> has eigenphase theta/(4*pi), so the estimator returns
        // m = theta*N/(4*pi).
        let expected = (deg.to_radians() / (4.0 * std::f64::consts::PI) * (1 << precision) as f64)
            .round() as usize
            % (1 << precision);
        assert!(
            masses[expected] > 0.99,
            "theta {deg}: expected {expected:03b}, masses {masses:?}"
        );
    }
}

#[test]
fn qpe_depends_on_the_supplied_unitary() {
    // A regression guard for the old bug: the precision register was never
    // entangled with the unitary, so every unitary gave the same distribution.
    let uniform = phase_register_masses(0.0, 3);
    let rotated = phase_register_masses(90.0, 3);
    assert!(uniform[0] > 0.99);
    assert!(rotated[1] > 0.99);
    assert!(rotated[0] < 1e-9);
}

#[test]
fn qpe_splits_half_integer_phases_between_neighbouring_bins() {
    let masses = phase_register_masses(45.0, 3);
    // theta=45 deg gives m=0.5: the probability sits in bins 000 and 001.
    assert!(
        masses[0] > 0.35 && masses[1] > 0.35,
        "half-integer phase masses {masses:?}"
    );
}

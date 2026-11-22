//! Tests for the stabilizer tableau simulator (Gottesman-Knill).
//!
//! The module had no test coverage at all, which is how six bugs survived.
//! Exposing it through the Python bindings is what surfaced them:
//!
//! * [`StabilizerTableau::measure`] searched all `2n` rows for one
//!   anticommuting with `Z_q`. When the first such row was a *destabilizer*
//!   (index `>= n`), `r + n` indexed past the end and panicked.
//! * `measure` overwrote stabilizer row `r` with `±Z_q` before copying it to
//!   the destabilizer row, so the copy read the new row.
//! * `cnot` applied `X_c -> X_c X_t` and `Z_t -> Z_c Z_t` with the operands
//!   swapped, producing the stabilizer group of a different state.
//! * `h` multiplied a Y row by `i` where `H Y H = -Y` requires `-1`.
//! * `s` skipped the factor of `i` on the `Y -> -X` path.
//! * [`StabilizerTableau::equivalent`] was a stub that only compared `n`, so
//!   it reported any two same-width tableaux as equal.
//! * The deterministic branch of `measure` summed the phases of whichever
//!   stabilizers had `Z` on the measured qubit. That is only correct when those
//!   rows carry no `X` support; in general it returns the wrong outcome. It now
//!   projects onto the destabilizer dual basis to find the combination equal to
//!   `±Z_q`.
//! * `measure` returned a fixed `false` on the random branch. `measure_seeded`
//!   and `measure_with_rng` now draw a real, reproducible outcome.

use quantumforge::stabilizer::StabilizerTableau;
use quantumforge::{CircuitOp, Gate, StateVector, parse_circuit};

/// Tableau carrying the same Clifford circuit as `source`.
fn tableau_for(source: &str) -> StabilizerTableau {
    let circuit = parse_circuit(source).expect("parses");
    let mut tableau = StabilizerTableau::new(3);
    for operation in circuit.operations() {
        match operation {
            CircuitOp::SingleQubit { gate, target } => match gate {
                Gate::H => tableau.h(*target),
                Gate::S => tableau.s(*target),
                Gate::X => tableau.x(*target),
                Gate::Z => tableau.z(*target),
                _ => {}
            },
            CircuitOp::Cnot { control, target } => tableau.cnot(*control, *target),
            _ => {}
        }
    }
    tableau
}

/// For a stabilizer state, a qubit is deterministic exactly when its state
/// vector marginal is 0 or 1, and then the tableau's outcome must match. This
/// catches a wrong stabilizer group even though `measure` returns a fixed
/// outcome for the random branch.
#[test]
fn tableau_predicts_the_state_vector_marginals() {
    let circuits = [
        "Circuit[H[0], CNOT[0,1]]",
        "Circuit[X[0], H[1], Z[0], CNOT[0,1]]",
        "Circuit[H[0], S[1], CNOT[0,1], H[2], CNOT[1,2], X[0], Z[1]]",
        "Circuit[H[0], H[1], CNOT[0,2], S[2], CNOT[1,2], X[1]]",
    ];
    for source in circuits {
        let circuit = parse_circuit(source).expect("parses");
        let qubits = 3;
        let state = circuit
            .run(StateVector::zero(qubits).expect("state"))
            .expect("runs");
        let probabilities = state.probabilities();
        assert_eq!(probabilities.len(), 8, "wrong width for {source}");
        for qubit in 0..3 {
            let marginal_one: f64 = probabilities
                .iter()
                .enumerate()
                .filter(|(index, _)| ((*index) >> qubit) & 1 == 1)
                .map(|(_, p)| *p)
                .sum();
            let (outcome, probability) = tableau_for(source).measure(qubit);
            if probability == 1.0 {
                let expected = marginal_one > 0.5;
                assert_eq!(
                    outcome, expected,
                    "{source}: qubit {qubit} deterministic, tableau {outcome}, marginal {marginal_one}"
                );
            } else {
                assert!(
                    (marginal_one - 0.5).abs() < 1e-12,
                    "{source}: qubit {qubit} random in the tableau but marginal is {marginal_one}"
                );
            }
        }
    }
}

/// Regression: `measure` used to search all `2n` rows for one anticommuting
/// with `Z_q`. On a single qubit in an X eigenstate that finds the
/// *destabilizer* row (index `n`), and `r + n` then indexes past the end.
#[test]
fn x_eigenstate_measures_without_panicking() {
    let mut tableau = StabilizerTableau::new(1);
    tableau.x(0);
    assert_eq!(tableau.num_qubits(), 1);
    // X|0> = |1>, so a Z measurement is deterministic and true.
    assert_eq!(tableau.measure(0), (true, 1.0));
}

#[test]
fn z_eigenstate_measures_false() {
    let mut tableau = StabilizerTableau::new(1);
    assert_eq!(tableau.measure(0), (false, 1.0));
}

/// After a random-outcome measurement the state collapses, so re-measuring the
/// same qubit must be deterministic and agree with the first result.
#[test]
fn random_branch_collapses() {
    let mut tableau = StabilizerTableau::new(1);
    tableau.h(0);
    let (first, probability) = tableau.measure(0);
    assert_eq!(probability, 0.5);
    assert_eq!(tableau.measure(0), (first, 1.0));
}

/// The Bell state's two qubits are correlated. This exercises the row
/// reordering: the destabilizer must receive the *old* stabilizer row before
/// that row is replaced by `Z_q`.
#[test]
fn bell_state_measurements_are_correlated() {
    let circuit = parse_circuit("Circuit[H[0], CNOT[0,1]]").expect("parses");
    let state = circuit
        .run(StateVector::zero(2).expect("state"))
        .expect("runs");
    let probabilities = state.probabilities();
    assert!((probabilities[0] - 0.5).abs() < 1e-12);
    assert!(probabilities[1] < 1e-12);
    assert!(probabilities[2] < 1e-12);
    assert!((probabilities[3] - 0.5).abs() < 1e-12);

    let mut tableau = StabilizerTableau::new(2);
    tableau.h(0);
    tableau.cnot(0, 1);
    let (first, probability) = tableau.measure(0);
    assert_eq!(probability, 0.5);
    let (second, collapsed) = tableau.measure(1);
    assert_eq!(collapsed, 1.0);
    assert_eq!(first, second, "Bell qubits must agree");
}

/// GHZ: three correlated qubits, all equal.
#[test]
fn ghz_state_measurements_are_correlated() {
    let mut tableau = StabilizerTableau::new(3);
    tableau.h(0);
    tableau.cnot(0, 1);
    tableau.cnot(1, 2);
    let (first, _) = tableau.measure(0);
    let (second, probability) = tableau.measure(1);
    assert_eq!(probability, 1.0);
    let (third, probability) = tableau.measure(2);
    assert_eq!(probability, 1.0);
    assert_eq!(first, second);
    assert_eq!(second, third);
}

/// `measure_seeded` must actually sample both outcomes and be reproducible.
#[test]
fn seeded_measurement_samples_and_reproduces() {
    let trials = 400;
    let mut ones = 0;
    for seed in 0..trials {
        let mut tableau = StabilizerTableau::new(1);
        tableau.h(0);
        let (outcome, probability) = tableau.measure_seeded(0, seed);
        assert_eq!(probability, 0.5);
        if outcome {
            ones += 1;
        }
    }
    assert!(
        (100..300).contains(&ones),
        "400 fair draws should be near half ones, got {ones}"
    );

    let mut a = StabilizerTableau::new(1);
    a.h(0);
    let mut b = StabilizerTableau::new(1);
    b.h(0);
    assert_eq!(
        a.measure_seeded(0, 42),
        b.measure_seeded(0, 42),
        "the same seed must reproduce the same outcome"
    );
}

/// A deterministic qubit must not depend on the seed.
#[test]
fn deterministic_measurement_ignores_the_seed() {
    for seed in [0, 1, 7, 123_456] {
        let mut tableau = StabilizerTableau::new(1);
        assert_eq!(tableau.measure_seeded(0, seed), (false, 1.0));
    }
}

/// With real draws the Bell state's two qubits must agree, and the first
/// measurement must come out both ways across seeds.
#[test]
fn bell_correlations_hold_under_seeded_measurement() {
    let mut ones = 0;
    for seed in 0..200u64 {
        let mut tableau = StabilizerTableau::new(2);
        tableau.h(0);
        tableau.cnot(0, 1);
        let (first, probability) = tableau.measure_seeded(0, seed);
        assert_eq!(probability, 0.5);
        let (second, collapsed) = tableau.measure_seeded(1, seed.wrapping_add(1));
        assert_eq!(collapsed, 1.0, "the collapse must be deterministic");
        assert_eq!(first, second, "Bell qubits must agree (seed {seed})");
        if first {
            ones += 1;
        }
    }
    assert!(
        (40..160).contains(&ones),
        "200 fair draws should be near half ones, got {ones}"
    );
}

#[test]
fn equivalent_tableaux_agree_and_fresh_ones_do_not() {
    let mut a = StabilizerTableau::new(2);
    a.h(0);
    a.cnot(0, 1);
    let mut b = StabilizerTableau::new(2);
    b.h(0);
    b.cnot(0, 1);
    assert!(a.equivalent(&b));

    let fresh = StabilizerTableau::new(2);
    assert!(!a.equivalent(&fresh));
}

/// `equivalent` must not be fooled by a different state of the same width.
#[test]
fn equivalent_distinguishes_states() {
    let bell = tableau_for("Circuit[H[0], CNOT[0,1]]");
    assert!(bell.equivalent(&tableau_for("Circuit[H[0], CNOT[0,1]]")));
    assert!(!bell.equivalent(&tableau_for("Circuit[H[0]]")));
    assert!(!bell.equivalent(&tableau_for("Circuit[]")));
}

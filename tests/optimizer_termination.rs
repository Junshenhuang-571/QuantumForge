//! Regression tests for optimizer termination.
//!
//! `commute_for_cancellation` used to step its index backwards after every swap,
//! so two gates that merely commuted exchanged places and then swapped straight
//! back on the next iteration. The inner sweep never advanced and the process
//! spun forever; `max_passes` only bounded the outer loop, so a three-gate
//! circuit such as `cz(1,0); t(1); h(0)` hung `quantumforge optimize ... all`
//! indefinitely. `conservative` mode skipped the pass entirely and looked fine,
//! which is why this survived.

use quantumforge::Circuit;
use quantumforge::optimizer::{OptimizeConfig, optimize_circuit, verify_unitary_equivalent};

/// Parse a QuantumForge circuit expression.
fn parse(source: &str) -> Circuit {
    quantumforge::parser::parse_circuit(source).expect("parses")
}

#[test]
fn commuting_gates_no_longer_spin_the_optimizer() {
    // This exact shape hung before the fix. If it regresses the test hangs and
    // the harness times out rather than failing an assertion.
    let cases = [
        "Circuit[CZ[1,0], T[1], H[0]]",
        "Circuit[CZ[0,1], T[1], H[0]]",
        "Circuit[CZ[1,0], Tdg[1], H[0]]",
        "Circuit[CZ[1,0], T[1], X[0]]",
        // The same pattern with padding gates either side.
        "Circuit[X[1], CZ[1,0], T[1], H[0], Y[1]]",
        // A longer alternating chain, which stressed the pass limit.
        "Circuit[CZ[1,0], T[1], H[0], S[1], Z[0], Tdg[1], Y[0]]",
    ];
    for source in cases {
        let mut circuit = parse(source);
        let reference = circuit.clone();
        optimize_circuit(&mut circuit, &OptimizeConfig::all());
        let check = verify_unitary_equivalent(&reference, &circuit, 2, 1e-9)
            .expect("equivalence check runs");
        assert!(
            check.equivalent,
            "{source} changed the unitary: max amplitude error {:?}",
            check.max_amplitude_error
        );
    }
}

#[test]
fn commuting_still_brings_cancellable_pairs_together() {
    // The pass exists to make inverse pairs adjacent, so the fix must not turn
    // it into a no-op: H(0), X(1), H(0) has to collapse to X(1).
    let mut circuit =
        quantumforge::parser::parse_circuit("Circuit[H[0], X[1], H[0]]").expect("parses");
    let result = optimize_circuit(&mut circuit, &OptimizeConfig::all());
    // One inverse PAIR removes two gates.
    assert_eq!(result.cancellations, 1, "the H pair should cancel");
    assert_eq!(circuit.operations().len(), 1);
}

#[test]
fn conservative_mode_is_unaffected() {
    let source = "Circuit[CZ[1,0], T[1], H[0]]";
    let mut circuit = parse(source);
    let reference = circuit.clone();
    optimize_circuit(&mut circuit, &OptimizeConfig::conservative());
    let check = verify_unitary_equivalent(&reference, &circuit, 2, 1e-9).expect("runs");
    assert!(check.equivalent);
}

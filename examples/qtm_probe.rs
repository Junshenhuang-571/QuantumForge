//! Demonstrate the two QTM reversibility paths in `quantumforge::qtm`:
//!
//! 1. A **total, injective** transition table (writes the read symbol back,
//!    moves right which wraps, flips state) forms a unitary directly.
//! 2. An **information-erasing** table (always writes 0) is *not* injective on
//!    the plain configuration register, so `step_quantum` rejects it —
//!    but `permutation_with_ancilla` builds a whole-space bijection via
//!    Bennett-style uncomputation (one ancilla slot per head position).
//!
//! Run with: `cargo run --release --example qtm_probe`

use quantumforge::qtm::{Direction, QuantumTuringMachine, Transition};
use std::collections::BTreeSet;

/// A reversible scanner: write the read symbol back, move right, flip state.
fn reversible_machine() -> QuantumTuringMachine {
    let transitions = vec![
        vec![
            Some(Transition {
                next_state: 1,
                write: 0,
                direction: Direction::Right,
            }),
            Some(Transition {
                next_state: 1,
                write: 1,
                direction: Direction::Right,
            }),
        ],
        vec![
            Some(Transition {
                next_state: 0,
                write: 0,
                direction: Direction::Right,
            }),
            Some(Transition {
                next_state: 0,
                write: 1,
                direction: Direction::Right,
            }),
        ],
    ];
    QuantumTuringMachine::new(2, 2, 4, 4, transitions, 0).expect("valid machine")
}

/// An erasing machine: always write 0 regardless of the read symbol.
fn erasing_machine() -> QuantumTuringMachine {
    let transitions = vec![
        vec![
            Some(Transition {
                next_state: 1,
                write: 0,
                direction: Direction::Right,
            }),
            Some(Transition {
                next_state: 1,
                write: 0,
                direction: Direction::Right,
            }),
        ],
        vec![
            Some(Transition {
                next_state: 0,
                write: 0,
                direction: Direction::Right,
            }),
            Some(Transition {
                next_state: 0,
                write: 0,
                direction: Direction::Right,
            }),
        ],
    ];
    QuantumTuringMachine::new(2, 2, 4, 4, transitions, 0).expect("valid machine")
}

fn main() {
    // 1. Reversible machine: runs as a unitary, no ancilla needed.
    let rev = reversible_machine();
    println!(
        "reversible machine: reversibility gap = {}",
        rev.reversibility_gap()
    );
    let sup = rev
        .prepare_superposition(&[0, 1, 0, 0], 2)
        .expect("superposition");
    let stepped = rev.step_quantum(&sup).expect("quantum step");
    println!(
        "  unitary step preserved norm: {} -> {}",
        sup.norm_sqr(),
        stepped.norm_sqr()
    );

    // 2. Erasing machine: rejected on the plain register, fixed by ancilla.
    let era = erasing_machine();
    println!(
        "erasing machine: reversibility gap = {}",
        era.reversibility_gap()
    );
    assert!(
        era.step_quantum(&era.prepare_superposition(&[0, 0, 0, 0], 0).unwrap())
            .is_err(),
        "erasing table must be rejected on the plain register"
    );
    println!("  step_quantum rejected the plain register (not a bijection)");

    let (perm, qubits) = era
        .permutation_with_ancilla(4)
        .expect("ancilla permutation");
    let distinct: BTreeSet<_> = perm.values().copied().collect();
    println!(
        "  ancilla uncomputation: {qubits} qubits, {} distinct targets over {} entries -> {}",
        distinct.len(),
        perm.len(),
        if distinct.len() == perm.len() {
            "bijection"
        } else {
            "NOT a bijection"
        }
    );
    assert_eq!(
        distinct.len(),
        perm.len(),
        "ancilla permutation must be a bijection"
    );
    println!("all checks passed");
}

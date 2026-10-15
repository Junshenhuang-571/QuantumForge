//! Demonstrate the two QTM reversibility paths in `quantumforge::qtm`:
//!
//! 1. A **total, injective** transition table (writes the read symbol back,
//!    moves right which wraps, flips state) forms a unitary directly via
//!    `step_quantum`.
//! 2. An **information-erasing** table (always writes 0) is *not* injective on
//!    the plain register, so `step_quantum` rejects it — but
//!    `step_quantum_ancilla` runs it as a unitary on an ancilla-augmented
//!    register that preserves the erased symbol. The ancilla map is an
//!    involution: one step advances, a second step returns to the start
//!    (reversible single-step ancilla stepping; full unbounded multi-step
//!    Bennett uncomputation is out of scope for a bounded tape).
//!
//! Run with: `cargo run --release --example qtm_probe`

use quantumforge::qtm::{Direction, QuantumTuringMachine, Transition};

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

    // 2. Erasing machine: rejected on the plain register, run on ancilla instead.
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

    let asup = era
        .prepare_superposition_ancilla(&[1, 0, 1, 0], 0, 4)
        .expect("ancilla superposition");
    let (_, qubits) = era
        .permutation_with_ancilla(4)
        .expect("ancilla permutation");
    let before = asup.norm_sqr();
    let once = era.step_quantum_ancilla(&asup, 4).expect("ancilla step");
    let twice = era.step_quantum_ancilla(&once, 4).expect("ancilla step 2");
    println!(
        "  ancilla register ({qubits} qubits): step preserved norm {} -> {}; second step restores {} (involution)",
        before,
        once.norm_sqr(),
        twice.norm_sqr()
    );
    assert!(
        (before - once.norm_sqr()).abs() < 1e-12,
        "ancilla step preserves norm"
    );
    assert!(
        (before - twice.norm_sqr()).abs() < 1e-12,
        "two steps uncompute"
    );
    println!("all checks passed");
}

//! Integration tests for the Quantum Turing Machine module (Bernstein–Vazirani
//! unitary formulation).

use quantumforge::qtm::{Configuration, Direction, QuantumTuringMachine, Transition};

/// A genuinely reversible QTM used for the quantum-step tests: 2 states that
/// toggle each step, head stays put, tape unchanged. Successor of
/// `(s, h, tape)` is `(1-s, h, tape)` — a perfect involution, hence a bijection
/// over the whole configuration space (no off-tape moves, no collisions).
fn toggle_machine() -> QuantumTuringMachine {
    let transitions = vec![
        // state 0: reading 0 or 1 -> flip to state 1, write same symbol, stay.
        vec![
            Some(Transition {
                next_state: 1,
                write: 0,
                direction: Direction::Stay,
            }),
            Some(Transition {
                next_state: 1,
                write: 1,
                direction: Direction::Stay,
            }),
        ],
        // state 1: -> flip back to state 0, write same symbol, stay.
        vec![
            Some(Transition {
                next_state: 0,
                write: 0,
                direction: Direction::Stay,
            }),
            Some(Transition {
                next_state: 0,
                write: 1,
                direction: Direction::Stay,
            }),
        ],
    ];
    QuantumTuringMachine::new(2, 2, 4, 4, transitions, 0).expect("valid machine")
}

#[test]
fn encode_decode_round_trips() {
    let m = toggle_machine();
    for state in 0..m.states {
        for head in 0..m.tape_cells {
            for a in 0..m.alphabet {
                for b in 0..m.alphabet {
                    let tape = vec![a as u8, b as u8, 0, 0];
                    let config = Configuration { tape, head, state };
                    let index = m.encode(&config).expect("encode");
                    let back = m.decode(index);
                    assert_eq!(config, back, "round-trip failed for {config:?}");
                }
            }
        }
    }
}

#[test]
fn classical_toggle_flips_state() {
    let m = toggle_machine();
    let start = Configuration {
        tape: vec![0, 1, 0, 0],
        head: 2,
        state: 0,
    };
    let end = m.run_classical(&start, 1);
    assert_eq!(end.state, 1, "one step toggles state 0 -> 1");
    assert_eq!(end.tape, start.tape, "tape unchanged");
    assert_eq!(end.head, 2, "head stays");
    let back = m.run_classical(&start, 2);
    assert_eq!(back.state, 0, "two steps return to start (involution)");
}

#[test]
fn quantum_step_matches_classical_on_basis_state() {
    let m = toggle_machine();
    let start = Configuration {
        tape: vec![0, 1, 0, 0],
        head: 2,
        state: 0,
    };
    let mut amps = vec![quantumforge::Complex64::ZERO; 1usize << m.qubits()];
    amps[m.encode(&start).unwrap()] = quantumforge::Complex64::ONE;
    let state = quantumforge::StateVector::from_amplitudes(amps).unwrap();

    let stepped = m.step_quantum(&state).expect("quantum step");
    let successor = m.step_classical(&start).expect("has successor");
    let succ_index = m.encode(&successor).unwrap();
    for (i, amp) in stepped.amplitudes().iter().enumerate() {
        let expected = if i == succ_index { 1.0 } else { 0.0 };
        assert!(
            (amp.re - expected).abs() < 1e-12,
            "index {i}: got {}",
            amp.re
        );
    }
}

#[test]
fn quantum_step_is_unitary_on_superposition() {
    let m = toggle_machine();
    let state = m
        .prepare_superposition(&[0, 1, 0, 0], 2)
        .expect("superposition");
    let before = state.norm_sqr();
    let after = m.step_quantum(&state).expect("quantum step");
    assert!(
        (before - after.norm_sqr()).abs() < 1e-12,
        "norm not preserved"
    );
    assert!((after.norm_sqr() - 1.0).abs() < 1e-12, "not normalized");
}

#[test]
fn non_reversible_table_is_rejected() {
    // Reading 0 and 1 both map to the same successor config -> not a bijection.
    let transitions = vec![
        vec![
            Some(Transition {
                next_state: 1,
                write: 1,
                direction: Direction::Right,
            }),
            Some(Transition {
                next_state: 1,
                write: 1,
                direction: Direction::Right,
            }),
        ],
        vec![None, None],
    ];
    let m = QuantumTuringMachine::new(2, 2, 4, 4, transitions, 0).expect("constructs");
    let mut amps = vec![quantumforge::Complex64::ZERO; 1usize << m.qubits()];
    amps[0] = quantumforge::Complex64::ONE;
    let state = quantumforge::StateVector::from_amplitudes(amps).unwrap();
    assert!(
        m.step_quantum(&state).is_err(),
        "non-reversible table should be rejected"
    );
}

/// A "move right, flip state" scanner: successor of `(s, h, tape)` is
/// `(1-s, (h+1) mod cells, tape)`. Reversible now that head moves wrap.
fn scan_machine() -> QuantumTuringMachine {
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

#[test]
fn head_moves_wrap_and_scan_is_reversible() {
    let m = scan_machine();
    // Right from the last cell wraps to cell 0 (previously this halted).
    let start = Configuration {
        tape: vec![0, 0, 0, 0],
        head: 3,
        state: 0,
    };
    let next = m.step_classical(&start).expect("has successor");
    assert_eq!(next.head, 0, "Right off the end wraps to cell 0");
    assert_eq!(next.state, 1, "state flips");
    // A total, injective table with wrapping heads is a bijection -> gap 0.
    assert_eq!(
        m.reversibility_gap(),
        0,
        "scan machine should be a bijection"
    );
    // And it runs as a quantum unitary.
    let sup = m
        .prepare_superposition(&[0, 0, 0, 0], 3)
        .expect("superposition");
    let before = sup.norm_sqr();
    let after = m.step_quantum(&sup).expect("quantum step");
    assert!((before - after.norm_sqr()).abs() < 1e-12, "norm preserved");
}

#[test]
fn is_buildable_rejects_oversized_state_register() {
    // 8 states need 3 register bits, but head_cells=2 gives only 2, so the
    // state register cannot encode them injectively. `new` accepts the shape
    // (8 is a power of two); `is_buildable` is what rejects it.
    let transitions = vec![vec![None, None]; 8];
    let m = QuantumTuringMachine::new(8, 2, 4, 2, transitions, 0).expect("8 states ok to build");
    // 8 states need 3 bits but head_cells=2 gives 2 -> is_buildable fails.
    assert!(
        m.is_buildable().is_err(),
        "8 states in 2 bits should be rejected"
    );
}

/// Binary counter machine, reversible by construction: 2 states {carry, done}
/// that ALWAYS transition (no fixed points), head moves right and wraps, tape
/// unchanged, state flips each step. A clean bijection (involution on state +
/// shift on head) — shows a total injective table needs no ancilla.
fn counter_machine() -> QuantumTuringMachine {
    let transitions = vec![
        // state 0 (carry): move right, flip to done.
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
        // state 1 (done): move right, flip back to carry.
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

/// A machine whose tape write ERASES information (reads 0 or 1, always writes
/// 0) — not injective on the plain register, so it needs ancilla.
fn erasing_machine() -> QuantumTuringMachine {
    let transitions = vec![
        // state 0: read anything, write 0, move right, flip state.
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
        // state 1: read anything, write 0, move right, flip back.
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

#[test]
fn total_injective_machine_needs_no_ancilla() {
    let m = counter_machine();
    assert_eq!(
        m.reversibility_gap(),
        0,
        "total injective table is a bijection"
    );
    let sup = m
        .prepare_superposition(&[1, 0, 1, 0], 0)
        .expect("superposition");
    let after = m.step_quantum(&sup).expect("quantum step");
    assert!(
        (sup.norm_sqr() - after.norm_sqr()).abs() < 1e-12,
        "norm preserved"
    );
}

#[test]
fn erasing_machine_needs_ancilla_and_builds_bijection() {
    let m = erasing_machine();
    // Plain register: the write erases the read symbol -> not injective.
    assert!(
        m.step_quantum(&m.prepare_superposition(&[0, 0, 0, 0], 0).unwrap())
            .is_err(),
        "erasing write should be rejected on the plain register"
    );
    assert!(
        m.reversibility_gap() > 0,
        "erasing machine has a reversibility gap"
    );
    // Too-small ancilla is rejected.
    assert!(
        m.permutation_with_ancilla(2).is_err(),
        "ancilla < tape_cells rejected"
    );
    // With a full ancilla register it must form a bijection.
    let (perm, qubits) = m.permutation_with_ancilla(4).expect("ancilla permutation");
    assert_eq!(
        perm.len(),
        1usize << qubits,
        "covers the whole enlarged space"
    );
    let distinct: std::collections::BTreeSet<_> = perm.values().copied().collect();
    assert_eq!(
        distinct.len(),
        perm.len(),
        "ancilla permutation is a bijection"
    );
}

#[test]
fn erasing_machine_runs_quantumly_on_ancilla_register() {
    // The real fix: the erasing table runs as a unitary on the ancilla-augmented
    // register. One step preserves norm; because the map is an involution, two
    // steps return to the start (reversible ancilla stepping).
    let m = erasing_machine();
    let sup = m
        .prepare_superposition_ancilla(&[1, 0, 1, 0], 0, 4)
        .expect("ancilla superposition");
    let before = sup.norm_sqr();
    let once = m.step_quantum_ancilla(&sup, 4).expect("ancilla step");
    assert!(
        (before - once.norm_sqr()).abs() < 1e-12,
        "ancilla step preserves norm"
    );
    // The step actually moved the state (not a no-op).
    assert!(
        once.amplitudes()[..]
            .iter()
            .zip(sup.amplitudes())
            .any(|(a, b)| (*a - *b).norm_sqr() > 1e-12),
        "ancilla step changes the state"
    );
    // Involution: applying twice returns to the original amplitudes.
    let twice = m.step_quantum_ancilla(&once, 4).expect("ancilla step 2");
    for (a, b) in twice.amplitudes().iter().zip(sup.amplitudes()) {
        assert!((*a - *b).norm_sqr() < 1e-12, "two ancilla steps uncompute");
    }
}

#[test]
fn ancilla_capacity_guard_rejects_oversized_tape() {
    // A tape whose ancilla-augmented register exceeds the qubit cap is rejected
    // instead of trying to allocate an enormous state vector.
    let m = erasing_machine();
    assert!(
        m.permutation_with_ancilla(1024).is_err(),
        "oversized ancilla tape must be rejected"
    );
}

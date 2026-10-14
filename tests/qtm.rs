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
            Some(Transition { next_state: 1, write: 0, direction: Direction::Stay }),
            Some(Transition { next_state: 1, write: 1, direction: Direction::Stay }),
        ],
        // state 1: -> flip back to state 0, write same symbol, stay.
        vec![
            Some(Transition { next_state: 0, write: 0, direction: Direction::Stay }),
            Some(Transition { next_state: 0, write: 1, direction: Direction::Stay }),
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
    let start = Configuration { tape: vec![0, 1, 0, 0], head: 2, state: 0 };
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
    let start = Configuration { tape: vec![0, 1, 0, 0], head: 2, state: 0 };
    let mut amps = vec![quantumforge::Complex64::ZERO; 1usize << m.qubits()];
    amps[m.encode(&start).unwrap()] = quantumforge::Complex64::ONE;
    let state = quantumforge::StateVector::from_amplitudes(amps).unwrap();

    let stepped = m.step_quantum(&state).expect("quantum step");
    let successor = m.step_classical(&start).expect("has successor");
    let succ_index = m.encode(&successor).unwrap();
    for (i, amp) in stepped.amplitudes().iter().enumerate() {
        let expected = if i == succ_index { 1.0 } else { 0.0 };
        assert!((amp.re - expected).abs() < 1e-12, "index {i}: got {}", amp.re);
    }
}

#[test]
fn quantum_step_is_unitary_on_superposition() {
    let m = toggle_machine();
    let state = m.prepare_superposition(&[0, 1, 0, 0], 2).expect("superposition");
    let before = state.norm_sqr();
    let after = m.step_quantum(&state).expect("quantum step");
    assert!((before - after.norm_sqr()).abs() < 1e-12, "norm not preserved");
    assert!((after.norm_sqr() - 1.0).abs() < 1e-12, "not normalized");
}

#[test]
fn non_reversible_table_is_rejected() {
    // Reading 0 and 1 both map to the same successor config -> not a bijection.
    let transitions = vec![
        vec![
            Some(Transition { next_state: 1, write: 1, direction: Direction::Right }),
            Some(Transition { next_state: 1, write: 1, direction: Direction::Right }),
        ],
        vec![None, None],
    ];
    let m = QuantumTuringMachine::new(2, 2, 4, 4, transitions, 0).expect("constructs");
    let mut amps = vec![quantumforge::Complex64::ZERO; 1usize << m.qubits()];
    amps[0] = quantumforge::Complex64::ONE;
    let state = quantumforge::StateVector::from_amplitudes(amps).unwrap();
    assert!(m.step_quantum(&state).is_err(), "non-reversible table should be rejected");
}

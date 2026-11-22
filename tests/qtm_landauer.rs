//! Landauer accounting for the classical step of a quantum Turing machine.

use quantumforge::qtm::{Direction, QuantumTuringMachine, Transition};

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

#[test]
fn reversible_step_produces_no_entropy() {
    let machine = reversible_machine();
    assert_eq!(machine.reversibility_gap(), 0);
    assert!(machine.entropy_production_per_step().abs() < 1e-12);
    assert!(machine.landauer_heat_per_step(300.0).abs() < 1e-30);
}

#[test]
fn erasing_one_bit_costs_kt_ln2() {
    let machine = erasing_machine();
    assert!(machine.reversibility_gap() > 0);
    // The table erases exactly one bit per step, so the uniform-input entropy
    // production is exactly ln 2 and the Landauer heat is k_B T ln 2.
    let production = machine.entropy_production_per_step();
    assert!(
        (production - std::f64::consts::LN_2).abs() < 1e-12,
        "entropy production {production} vs ln 2"
    );
    let temperature = 300.0;
    let expected = 1.380649e-23 * temperature * std::f64::consts::LN_2;
    assert!(
        (machine.landauer_heat_per_step(temperature) - expected).abs() < 1e-35,
        "Landauer heat {} vs {expected}",
        machine.landauer_heat_per_step(temperature)
    );
}

#[test]
fn landauer_heat_is_linear_in_temperature() {
    let machine = erasing_machine();
    let cold = machine.landauer_heat_per_step(100.0);
    let hot = machine.landauer_heat_per_step(300.0);
    assert!((hot - 3.0 * cold).abs() < 1e-35);
}

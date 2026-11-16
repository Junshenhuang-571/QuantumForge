//! QEC round-trip: verify a surface code's structure, then run a full
//! encode -> inject -> syndrome -> recover -> verify cycle on the repetition
//! code (whose Z-type stabilizers make the measurement deterministic).
//!
//! Syndrome extraction circuits act on data + ancilla qubits, so the encoded
//! state is padded with |0> ancillas before measurement and traced back down
//! before decoding.
//!
//! Run with: cargo run --example qec_roundtrip

use quantumforge::qec::{
    QecCode, encode_repetition_bitflip, repetition_code_bitflip, surface_code_rotated, verify_code,
    verify_single_error_syndromes,
};
use quantumforge::{Circuit, Complex64, Gate, StateVector};

/// Pad a state with `n_ancilla` fresh |0> qubits (appended as the least
/// significant qubits, matching the syndrome circuit's ancilla indices).
fn with_ancillas(
    state: &StateVector,
    n_ancilla: usize,
) -> Result<StateVector, quantumforge::QuantumError> {
    let mut amps = vec![Complex64::ZERO; 1usize << (state.qubits() + n_ancilla)];
    for (i, a) in state.amplitudes().iter().enumerate() {
        amps[i << n_ancilla] = *a;
    }
    StateVector::from_amplitudes(amps)
}

fn measure_syndrome(code: &QecCode, state: &StateVector) -> u64 {
    let n_ancilla = code.stabilizers.len();
    let shots = code
        .syndrome_circuit
        .run_shots(state, 1, 0)
        .expect("syndrome extraction must produce an outcome");
    let bits = shots
        .counts
        .keys()
        .next()
        .expect("syndrome measurement produced no outcome");
    assert_eq!(bits.len(), n_ancilla, "one classical bit per stabilizer");
    bits.iter()
        .enumerate()
        .filter(|&(_, b)| *b == 1)
        .fold(0u64, |acc, (i, _)| acc | (1 << i))
}

fn main() {
    // --- Part 1: surface code construction and structural verification ---
    let surface = surface_code_rotated(3);
    println!(
        "rotated surface code d=3: {} physical qubits, {} logical",
        surface.n_physical, surface.n_logical
    );
    println!(
        "stabilizer commutation + logical pair: {}",
        verify_code(&surface)
    );
    println!(
        "every single-qubit error has a distinct, detectable syndrome: {}",
        verify_single_error_syndromes(&surface)
    );

    // --- Part 2: full round-trip on the 3-qubit repetition code ---
    let code = repetition_code_bitflip(3);
    let n_ancilla = code.stabilizers.len();
    println!(
        "\nrepetition code n=3: {} physical qubits, {} logical, {} stabilizers",
        code.n_physical, code.n_logical, n_ancilla
    );

    // Encode the logical |0> state, then pad with ancillas for extraction.
    let logical = StateVector::zero(1).unwrap();
    let encoded = encode_repetition_bitflip(&logical, 3).unwrap();
    let mut state = with_ancillas(&encoded, n_ancilla).unwrap();
    println!(
        "encoded |0> -> |000>; syndrome = {:#b}",
        measure_syndrome(&code, &state)
    );

    // Inject a bit-flip on the middle data qubit.
    let mut error = Circuit::new();
    error.push(Gate::X, 1);
    state = error.run(state).unwrap();
    let syndrome = measure_syndrome(&code, &state);
    println!("after X error on qubit 1: syndrome = {syndrome:#b}");
    assert_ne!(syndrome, 0, "the error must be detected");

    // Look up and apply the recovery operation.
    let recovery = code
        .recovery
        .get(&syndrome)
        .expect("recovery table has no entry for this syndrome");
    println!(
        "recovery: {:?}",
        recovery
            .operations()
            .iter()
            .map(|op| format!("{op:?}"))
            .collect::<Vec<_>>()
    );
    state = recovery.run(state).unwrap();
    let after = measure_syndrome(&code, &state);
    println!("syndrome after recovery: {after:#b}");
    assert_eq!(after, 0, "recovery must return the state to the code space");

    // The recovery acts only on data qubits, so the ancillas are still |0>:
    // joint index 0 is |000>|00>, i.e. the logical |0> payload is intact.
    let joint_probs = state.probabilities();
    println!(
        "joint state |00000> probability: {:.6} (payload intact)",
        joint_probs[0]
    );
    assert!(
        joint_probs[0] > 0.99,
        "logical |0> must survive the round-trip"
    );
    println!("round-trip OK: error detected, corrected, syndrome cleared");
}

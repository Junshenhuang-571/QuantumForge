//! Classical stabilizer codes: bit/phase flip repetition, Shor, Steane.

use super::{Circuit, Gate, PauliString, QecCode};
use crate::StateVector;
use std::collections::HashMap;

pub fn repetition_code_bitflip(n: usize) -> QecCode {
    assert!(n >= 3 && n % 2 == 1, "n must be odd and >= 3");

    let mut stabilizers = Vec::new();
    for i in 0..n - 1 {
        let mut pauli = PauliString::new(n);
        pauli.set(i, 3); // Z
        pauli.set(i + 1, 3); // Z
        stabilizers.push(pauli);
    }

    // Logical operators
    let mut logical_z = PauliString::new(n);
    for i in 0..n {
        logical_z.set(i, 3); // Z⊗n
    }

    let mut logical_x = PauliString::new(n);
    for i in 0..n {
        logical_x.set(i, 1); // X⊗n
    }

    // Syndrome extraction: CNOTs from data to ancillas for ZZ measurements
    let syndrome_circuit = build_repetition_syndrome(n, true);
    let recovery = build_repetition_recovery(n, true);

    QecCode {
        n_physical: n,
        n_logical: 1,
        stabilizers,
        logical_x: vec![logical_x],
        logical_z: vec![logical_z],
        syndrome_circuit,
        recovery,
    }
}

/// Build a phase-flip repetition code (n qubits, corrects 1 phase-flip).
/// Uses Hadamard-rotated basis: Logical |+⟩ = |+...+⟩, Logical |-⟩ = |-...-⟩
/// Stabilizers: X₀X₁, X₁X₂, ..., X_{n-2}X_{n-1}
pub fn repetition_code_phaseflip(n: usize) -> QecCode {
    assert!(n >= 3 && n % 2 == 1, "n must be odd and >= 3");

    let mut stabilizers = Vec::new();
    for i in 0..n - 1 {
        let mut pauli = PauliString::new(n);
        pauli.set(i, 1); // X
        pauli.set(i + 1, 1); // X
        stabilizers.push(pauli);
    }

    let mut logical_x = PauliString::new(n);
    for i in 0..n {
        logical_x.set(i, 1); // X⊗n
    }

    let mut logical_z = PauliString::new(n);
    for i in 0..n {
        logical_z.set(i, 3); // Z⊗n
    }

    let syndrome_circuit = build_repetition_syndrome(n, false);
    let recovery = build_repetition_recovery(n, false);

    QecCode {
        n_physical: n,
        n_logical: 1,
        stabilizers,
        logical_x: vec![logical_x],
        logical_z: vec![logical_z],
        syndrome_circuit,
        recovery,
    }
}

/// Build Shor's 9-qubit code (concatenated bit-flip + phase-flip).
/// Corrects arbitrary single-qubit errors.
pub fn shor_code() -> QecCode {
    let n = 9;
    let mut stabilizers = Vec::new();

    // Z-type stabilizers (detect X errors): Z⊗Z on adjacent qubits within each 3-qubit block
    for block in 0..3 {
        for i in 0..2 {
            let mut pauli = PauliString::new(n);
            pauli.set(block * 3 + i, 3); // Z
            pauli.set(block * 3 + i + 1, 3); // Z
            stabilizers.push(pauli);
        }
    }

    // X-type stabilizers (detect Z errors): X⊗X⊗...⊗X on pairs of 3-qubit blocks
    // S1 = X0 X1 X2 X3 X4 X5 (blocks 0 and 1)
    // S2 = X3 X4 X5 X6 X7 X8 (blocks 1 and 2)
    let mut p1 = PauliString::new(n);
    for i in 0..6 {
        p1.set(i, 1);
    }
    stabilizers.push(p1);

    let mut p2 = PauliString::new(n);
    for i in 3..9 {
        p2.set(i, 1);
    }
    stabilizers.push(p2);

    QecCode {
        n_physical: n,
        n_logical: 1,
        stabilizers,
        logical_x: vec![PauliString { paulis: vec![1; n] }], // X⊗9
        logical_z: vec![PauliString { paulis: vec![3; n] }], // Z⊗9
        syndrome_circuit: Circuit::new(),
        recovery: HashMap::new(),
    }
}

/// Build Steane's 7-qubit code.
/// CSS code from the \[7,4,3\] classical Hamming code.
pub fn steane_code() -> QecCode {
    let n = 7;
    let mut stabilizers = Vec::new();

    // X stabilizers (from parity checks of [7,4,3] Hamming code)
    // H = [1 1 1 0 1 0 0; 1 1 0 1 0 1 0; 1 0 1 1 0 0 1]
    let h_rows = [
        [1, 1, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 1, 0],
        [1, 0, 1, 1, 0, 0, 1],
    ];
    for row in h_rows {
        let mut pauli = PauliString::new(n);
        for (i, &bit) in row.iter().enumerate() {
            if bit == 1 {
                pauli.set(i, 1);
            }
        }
        stabilizers.push(pauli);
    }

    // Z stabilizers (same parity checks)
    for row in h_rows {
        let mut pauli = PauliString::new(n);
        for (i, &bit) in row.iter().enumerate() {
            if bit == 1 {
                pauli.set(i, 3);
            }
        }
        stabilizers.push(pauli);
    }

    let mut logical_x = PauliString::new(n);
    logical_x.paulis = vec![1, 1, 1, 1, 1, 1, 1]; // X⊗7
    let mut logical_z = PauliString::new(n);
    logical_z.paulis = vec![3, 3, 3, 3, 3, 3, 3]; // Z⊗7

    QecCode {
        n_physical: n,
        n_logical: 1,
        stabilizers,
        logical_x: vec![logical_x],
        logical_z: vec![logical_z],
        syndrome_circuit: Circuit::new(),
        recovery: HashMap::new(),
    }
}

/// Check that all stabilizers in `code` pairwise commute (necessary for a
fn build_repetition_syndrome(n: usize, bitflip: bool) -> Circuit {
    let mut circuit = Circuit::new();
    let n_ancilla = n - 1;
    let _total = n + n_ancilla;

    if bitflip {
        // Z-stabilizers: CNOT from data to ancilla (for ZZ measurement)
        for i in 0..n - 1 {
            circuit.cnot(i, n + i); // data i → ancilla i
            circuit.cnot(i + 1, n + i); // data i+1 → ancilla i
        }
    } else {
        // X-stabilizers: Hadamard + CNOT + Hadamard (for XX measurement)
        for i in 0..n - 1 {
            circuit.push(Gate::H, n + i);
            circuit.cnot(i, n + i);
            circuit.cnot(i + 1, n + i);
            circuit.push(Gate::H, n + i);
        }
    }
    circuit
}

pub(crate) fn build_repetition_recovery(n: usize, bitflip: bool) -> HashMap<u64, Circuit> {
    let mut recovery: HashMap<u64, Circuit> = HashMap::new();
    let n_syndromes = 1 << (n - 1);
    let recovery_gate = if bitflip { Gate::X } else { Gate::Z };

    for syndrome in 0..n_syndromes {
        let mut circuit = Circuit::new();

        // Decode syndrome to find which qubit has the error
        // For repetition codes, syndrome bits s_i = z_i XOR z_{i+1} (bitflip)
        // or s_i = x_i XOR x_{i+1} (phaseflip)
        // The syndrome pattern tells us which qubit flipped:
        // - syndrome = 0: no error
        // - syndrome = 1...0: error on qubit 0
        // - syndrome = 11...0: error on qubit 1
        // - syndrome = 0...11...0: error on qubit k (k 1's in middle)
        // - syndrome = 0...01: error on qubit n-1

        // Find the error position by finding the boundary between 0 and 1
        // in the syndrome (treating it as bits from LSB to MSB)
        let mut error_qubit = None;

        if syndrome == 0 {
            // No error
        } else {
            // Find error qubit from syndrome
            // Syndrome bit i = z_i XOR z_{i+1}
            // Single qubit k error produces:
            //   k=0: bit 0 set
            //   k=n-1: bit n-2 set
            //   0<k<n-1: bits k-1 and k set (contiguous pair)

            // Count set bits
            let positions: Vec<usize> = (0..n - 1).filter(|&i| (syndrome >> i) & 1 == 1).collect();

            if positions.len() == 1 {
                let i = positions[0];
                if i == 0 {
                    error_qubit = Some(0);
                } else if i == n - 2 {
                    error_qubit = Some(n - 1);
                }
                // else: invalid syndrome for single error
            } else if positions.len() == 2 && positions[1] == positions[0] + 1 {
                error_qubit = Some(positions[0] + 1);
            }
        }

        if let Some(qubit) = error_qubit {
            circuit.push(recovery_gate, qubit);
        }

        recovery.insert(syndrome, circuit);
    }
    recovery
}

/// Minimum-weight perfect matching (MWPM) decoder for QEC.
/// Given a syndrome (set of violated stabilizer indices), find the
/// minimum-weight set of single-qubit errors that produces it.
/// Uses brute-force search (up to ~3 errors) for small codes.
pub fn encode_repetition_bitflip(
    state: &StateVector,
    n: usize,
) -> Result<StateVector, crate::QuantumError> {
    // Input state should be 1-qubit (logical qubit)
    // We create an n-qubit state by tensoring with |0⟩^{n-1} then applying CNOTs
    let mut full_state = StateVector::zero(n)?;
    // Copy the 1-qubit state amplitudes to the first qubit of the n-qubit state
    // |ψ⟩|0...0⟩ = (α|0⟩ + β|1⟩)|0...0⟩ = α|0...0⟩ + β|0...01⟩ (qubit 0 = 1)
    let input_amps = state.amplitudes();
    if input_amps.len() >= 2 {
        full_state.amplitudes_mut()[0] = input_amps[0];
        full_state.amplitudes_mut()[1] = input_amps[1]; // |0...01⟩ (qubit 0 = 1)
    }

    // Apply CNOTs from qubit 0 to all others
    let mut circuit = Circuit::new();
    for i in 1..n {
        circuit.cnot(0, i);
    }
    circuit.run(full_state)
}

/// Decode a repetition code with majority vote.
pub fn decode_repetition_bitflip(
    state: &StateVector,
    _n: usize,
) -> Result<StateVector, crate::QuantumError> {
    // Syndrome measurement + recovery would go here
    // For now, just return the logical qubit (first qubit)
    let circuit = Circuit::new();
    // Majority vote decoding would be implemented here
    circuit.run(state.clone())
}

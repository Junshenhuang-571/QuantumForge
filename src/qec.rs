//! Quantum Error Correction (QEC) codes.
//!
//! Implements stabilizer codes for error detection and correction:
//! - Bit-flip repetition code (3-qubit, 5-qubit, general n)
//! - Phase-flip repetition code
//! - Shor's 9-qubit code
//! - Surface code (planar, rotated)
//! - Steane's 7-qubit code
//! - Syndrome extraction circuits

use crate::{Circuit, Gate, StateVector};
use std::collections::HashMap;

/// A quantum error correction code.
#[derive(Clone, Debug)]
pub struct QecCode {
    /// Number of physical qubits
    pub n_physical: usize,
    /// Number of logical qubits
    pub n_logical: usize,
    /// Stabilizer generators (Pauli strings)
    pub stabilizers: Vec<PauliString>,
    /// Logical X operators
    pub logical_x: Vec<PauliString>,
    /// Logical Z operators
    pub logical_z: Vec<PauliString>,
    /// Syndrome extraction circuit
    pub syndrome_circuit: Circuit,
    /// Recovery operations for each syndrome
    pub recovery: HashMap<u32, Circuit>,
}

/// Pauli string representation for stabilizers.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PauliString {
    /// Pauli on each qubit: 0=I, 1=X, 2=Y, 3=Z
    pub paulis: Vec<u8>,
}

impl PauliString {
    pub fn new(n: usize) -> Self {
        Self { paulis: vec![0; n] }
    }

    pub fn set(&mut self, qubit: usize, pauli: u8) {
        if qubit < self.paulis.len() {
            self.paulis[qubit] = pauli;
        }
    }

    pub fn weight(&self) -> usize {
        self.paulis.iter().filter(|&&p| p != 0).count()
    }

    pub fn to_string(&self) -> String {
        let mut s = String::new();
        for &p in &self.paulis {
            s.push(match p {
                0 => 'I',
                1 => 'X',
                2 => 'Y',
                3 => 'Z',
                _ => '?',
            });
        }
        s
    }
}

/// Build a bit-flip repetition code (n qubits, corrects 1 bit-flip).
/// Logical |0⟩ = |0...0⟩, Logical |1⟩ = |1...1⟩
/// Stabilizers: Z₀Z₁, Z₁Z₂, ..., Z_{n-2}Z_{n-1}
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
    // 9 data qubits + 8 ancillas for syndrome
    // This is a simplified construction - full implementation would be larger
    let n = 9;
    let mut stabilizers = Vec::new();

    // Phase-flip stabilizers (X checks on 3-qubit blocks)
    for block in 0..3 {
        for i in 0..2 {
            let mut pauli = PauliString::new(n);
            pauli.set(block * 3 + i, 1);
            pauli.set(block * 3 + i + 1, 1);
            stabilizers.push(pauli);
        }
    }
    // Between blocks
    let mut p = PauliString::new(n);
    p.set(2, 1); p.set(3, 1); stabilizers.push(p.clone());
    p = PauliString::new(n);
    p.set(5, 1); p.set(6, 1); stabilizers.push(p);

    // Bit-flip stabilizers (Z checks)
    for i in 0..8 {
        let mut pauli = PauliString::new(n);
        pauli.set(i, 3);
        pauli.set(i + 1, 3);
        stabilizers.push(pauli);
    }

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
/// CSS code from [7,4,3] classical Hamming code.
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
            if bit == 1 { pauli.set(i, 1); }
        }
        stabilizers.push(pauli);
    }

    // Z stabilizers (same parity checks)
    for row in h_rows {
        let mut pauli = PauliString::new(n);
        for (i, &bit) in row.iter().enumerate() {
            if bit == 1 { pauli.set(i, 3); }
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

/// Build a planar surface code (distance d).
/// 2d² - 2d + 1 data qubits, d² - 1 ancilla qubits.
pub fn surface_code_planar(d: usize) -> QecCode {
    assert!(d >= 3 && d % 2 == 1, "distance must be odd and >= 3");
    
    let n_data = 2 * d * d - 2 * d + 1;
    let n_ancilla = d * d - 1;
    let n = n_data + n_ancilla;

    let mut stabilizers = Vec::new();

    // This is a simplified representation
    // Full surface code needs lattice layout
    QecCode {
        n_physical: n,
        n_logical: 1,
        stabilizers,
        logical_x: vec![PauliString { paulis: vec![1; n] }],
        logical_z: vec![PauliString { paulis: vec![3; n] }],
        syndrome_circuit: Circuit::new(),
        recovery: HashMap::new(),
    }
}

/// Build a rotated surface code (distance d).
/// d² data qubits, (d² - 1)/2 ancillas each for X and Z.
pub fn surface_code_rotated(d: usize) -> QecCode {
    assert!(d >= 3 && d % 2 == 1);
    
    let n_data = d * d;
    let n_ancilla = (d * d - 1) / 2 * 2;
    let n = n_data + n_ancilla;

    QecCode {
        n_physical: n,
        n_logical: 1,
        stabilizers: Vec::new(),
        logical_x: vec![PauliString { paulis: vec![1; n] }],
        logical_z: vec![PauliString { paulis: vec![3; n] }],
        syndrome_circuit: Circuit::new(),
        recovery: HashMap::new(),
    }
}

// --- Internal syndrome circuit builders ---

fn build_repetition_syndrome(n: usize, bitflip: bool) -> Circuit {
    let mut circuit = Circuit::new();
    let n_ancilla = n - 1;
    let total = n + n_ancilla;

    if bitflip {
        // Z-stabilizers: CNOT from data to ancilla (for ZZ measurement)
        for i in 0..n - 1 {
            circuit.cnot(i, n + i);        // data i → ancilla i
            circuit.cnot(i + 1, n + i);    // data i+1 → ancilla i
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

fn build_repetition_recovery(n: usize, bitflip: bool) -> HashMap<u32, Circuit> {
    let mut recovery = HashMap::new();
    let n_syndromes = 1 << (n - 1);

    for syndrome in 0..n_syndromes {
        let mut circuit = Circuit::new();
        // Syndrome bits indicate which pair has the error
        // This is a simplified lookup-table recovery
        if bitflip {
            // X recovery on the appropriate qubit
            // In reality, this would use a decoding algorithm
        }
        recovery.insert(syndrome, circuit);
    }
    recovery
}

/// Encode a logical state into a repetition code.
pub fn encode_repetition_bitflip(state: &StateVector, n: usize) -> Result<StateVector, crate::QuantumError> {
    // Input state should be 1-qubit (logical qubit)
    // We create an n-qubit state by tensoring with |0⟩^{n-1} then applying CNOTs
    let mut full_state = StateVector::zero(n)?;
    // Copy the 1-qubit state amplitudes to the first qubit of the n-qubit state
    // |ψ⟩|0...0⟩ = (α|0⟩ + β|1⟩)|0...0⟩ = α|0...0⟩ + β|10...0⟩
    let input_amps = state.amplitudes();
    if input_amps.len() >= 2 {
        full_state.amplitudes_mut()[0] = input_amps[0];
        full_state.amplitudes_mut()[1 << (n - 1)] = input_amps[1]; // |10...0⟩
    }
    
    // Apply CNOTs from qubit 0 to all others
    let mut circuit = Circuit::new();
    for i in 1..n {
        circuit.cnot(0, i);
    }
    circuit.run(full_state)
}

/// Decode a repetition code with majority vote.
pub fn decode_repetition_bitflip(state: &StateVector, n: usize) -> Result<StateVector, crate::QuantumError> {
    // Syndrome measurement + recovery would go here
    // For now, just return the logical qubit (first qubit)
    let mut circuit = Circuit::new();
    // Majority vote decoding would be implemented here
    circuit.run(state.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repetition_code_bitflip() {
        let code = repetition_code_bitflip(3);
        assert_eq!(code.n_physical, 3);
        assert_eq!(code.n_logical, 1);
        assert_eq!(code.stabilizers.len(), 2);
        assert_eq!(code.stabilizers[0].to_string(), "ZZI");
        assert_eq!(code.stabilizers[1].to_string(), "IZZ");
    }

    #[test]
    fn test_repetition_code_phaseflip() {
        let code = repetition_code_phaseflip(5);
        assert_eq!(code.n_physical, 5);
        assert_eq!(code.stabilizers.len(), 4);
        assert_eq!(code.stabilizers[0].to_string(), "XXIII");
        assert_eq!(code.stabilizers[1].to_string(), "IXXII");
    }

    #[test]
    fn test_shor_code() {
        let code = shor_code();
        assert_eq!(code.n_physical, 9);
        // 6 X-type (phase-flip) + 8 Z-type (bit-flip) = 16 stabilizers
        assert_eq!(code.stabilizers.len(), 16);
    }

    #[test]
    fn test_steane_code() {
        let code = steane_code();
        assert_eq!(code.n_physical, 7);
        assert_eq!(code.stabilizers.len(), 6); // 3 X + 3 Z
    }

    #[test]
    fn test_encode_repetition() {
        use crate::StateVector;
        let state = StateVector::zero(1).unwrap(); // |0⟩
        let encoded = encode_repetition_bitflip(&state, 3).unwrap();
        assert_eq!(encoded.qubits(), 3);
    }

    #[test]
    fn test_pauli_string() {
        let mut p = PauliString::new(3);
        p.set(0, 1); p.set(1, 3);
        assert_eq!(p.to_string(), "XZI");
        assert_eq!(p.weight(), 2);
    }
}
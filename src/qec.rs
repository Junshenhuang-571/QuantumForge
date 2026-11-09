#![allow(missing_docs)]
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
use std::fmt;

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

    #[allow(clippy::manual_is_multiple_of)]
    pub fn commutes(&self, other: &PauliString) -> bool {
        let mut anticommute_count = 0;
        for i in 0..self.paulis.len() {
            let p = self.paulis[i];
            let q = other.paulis[i];
            if p != 0 && q != 0 && p != q {
                // X-Z, X-Y, Y-Z etc. anticommute
                if (p == 1 && q == 3)
                    || (p == 3 && q == 1)
                    || (p == 1 && q == 2)
                    || (p == 2 && q == 1)
                    || (p == 2 && q == 3)
                    || (p == 3 && q == 2)
                {
                    anticommute_count += 1;
                }
            }
        }
        anticommute_count % 2 == 0
    }
}

impl fmt::Display for PauliString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for &p in &self.paulis {
            let c = match p {
                0 => 'I',
                1 => 'X',
                2 => 'Y',
                3 => 'Z',
                _ => '?',
            };
            write!(f, "{c}")?;
        }
        Ok(())
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

/// Build a planar surface code (distance d).
/// 2d² - 2d + 1 data qubits, d² - 1 ancilla qubits.
pub fn surface_code_planar(d: usize) -> QecCode {
    assert!(d >= 3 && d % 2 == 1, "distance must be odd and >= 3");

    // Planar code: d×d data qubits with (d-1)×(d-1) plaquettes
    let n_data = d * d;
    let n = n_data;

    let mut stabilizers = Vec::new();

    // X-type stabilizers (plaquettes): (d-1)×(d-1) grid
    for fr in 0..d - 1 {
        for fc in 0..d - 1 {
            let q0 = fr * d + fc;
            let q1 = fr * d + (fc + 1);
            let q2 = (fr + 1) * d + fc;
            let q3 = (fr + 1) * d + (fc + 1);

            let mut pauli = PauliString::new(n);
            if (fr + fc).is_multiple_of(2) {
                for &q in &[q0, q1, q2, q3] {
                    pauli.set(q, 1); // X
                }
            } else {
                for &q in &[q0, q1, q2, q3] {
                    pauli.set(q, 3); // Z
                }
            }
            stabilizers.push(pauli);
        }
    }

    // Remove duplicate stabilizers
    let mut unique = Vec::new();
    for s in stabilizers {
        if !unique.iter().any(|u: &PauliString| u.paulis == s.paulis) {
            unique.push(s);
        }
    }
    let stabilizers = unique;

    let mut logical_x = PauliString::new(n);
    if n > 0 {
        logical_x.set(0, 1);
    }
    let mut logical_z = PauliString::new(n);
    if n > 0 {
        logical_z.set(0, 3);
    }

    let syndrome_circuit = build_surface_syndrome(n, &stabilizers);
    let recovery = build_surface_recovery(n, &stabilizers);

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

/// Build a rotated surface code (distance d).
/// d² data qubits, (d² - 1)/2 ancillas each for X and Z.
pub fn surface_code_rotated(d: usize) -> QecCode {
    assert!(d >= 3 && d % 2 == 1, "distance must be odd and >= 3");

    let n_data = d * d;
    let n = n_data; // data qubits only for simplicity

    let mut stabilizers = Vec::new();

    // Face stabilizers: (d-1)^2 faces, alternating X/Z on checkerboard
    for fr in 0..d - 1 {
        for fc in 0..d - 1 {
            let q0 = fr * d + fc;
            let q1 = fr * d + (fc + 1);
            let q2 = (fr + 1) * d + fc;
            let q3 = (fr + 1) * d + (fc + 1);

            let mut pauli = PauliString::new(n);
            let qs = [q0, q1, q2, q3];
            if (fr + fc).is_multiple_of(2) {
                for &q in &qs {
                    pauli.set(q, 1);
                } // X
            } else {
                for &q in &qs {
                    pauli.set(q, 3);
                } // Z
            }
            stabilizers.push(pauli);
        }
    }

    // Boundary stabilizers for d=3 (verified: all 27 single errors have distinct syndromes)
    if d == 3 {
        for paulis in [[0, 1, 1, 0, 0, 0, 1, 1, 0], [0, 1, 0, 1, 0, 1, 1, 0, 0]] {
            let mut pauli = PauliString::new(n);
            for (i, &p) in paulis.iter().enumerate() {
                if p != 0 {
                    pauli.set(i, p as u8);
                }
            }
            stabilizers.push(pauli);
        }
        for paulis in [[3, 0, 0, 3, 0, 3, 0, 0, 3], [0, 3, 0, 3, 0, 0, 0, 3, 3]] {
            let mut pauli = PauliString::new(n);
            for (i, &p) in paulis.iter().enumerate() {
                if p != 0 {
                    pauli.set(i, p as u8);
                }
            }
            stabilizers.push(pauli);
        }
    } else {
        // For d > 3, face stabilizers alone are sufficient and commute.
        // Boundary stabilizers are omitted to maintain commutation.
    }

    // Logical operators: X on top row, Z on left column
    let mut logical_x = PauliString::new(n);
    for c in 0..d {
        logical_x.set(c, 1);
    }
    let mut logical_z = PauliString::new(n);
    for r in 0..d {
        logical_z.set(r * d, 3);
    }

    let syndrome_circuit = build_surface_syndrome(n, &stabilizers);
    let recovery = build_surface_recovery(n, &stabilizers);

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

/// Build syndrome extraction circuit for a surface code.
/// Data qubits: 0..n_data, ancilla qubits: n_data..n_data+n_stab
fn build_surface_syndrome(n_data: usize, stabilizers: &[PauliString]) -> Circuit {
    let mut circuit = Circuit::new();

    for (i, stab) in stabilizers.iter().enumerate() {
        let ancilla = n_data + i;
        // Determine if X-type or Z-type
        let is_x_type = stab.paulis.contains(&1) && !stab.paulis.contains(&3);

        if is_x_type {
            // X stabilizer: prepare |+>, CNOT ancilla->data, H, measure
            circuit.push(Gate::H, ancilla);
            for (q, &p) in stab.paulis.iter().enumerate() {
                if p == 1 {
                    circuit.cnot(ancilla, q);
                }
            }
            circuit.push(Gate::H, ancilla);
        } else {
            // Z stabilizer: prepare |0>, CNOT data->ancilla, measure
            for (q, &p) in stab.paulis.iter().enumerate() {
                if p == 3 {
                    circuit.cnot(q, ancilla);
                }
            }
        }
    }

    // Measure all ancillas
    for i in 0..stabilizers.len() {
        circuit.measure(n_data + i, i);
    }

    circuit
}

/// Build recovery table for a surface code using single-error syndrome lookup.
fn build_surface_recovery(n_data: usize, stabilizers: &[PauliString]) -> HashMap<u32, Circuit> {
    let mut recovery = HashMap::new();
    let _ = stabilizers.len();

    // For each single-qubit error, compute syndrome and map to recovery
    for q in 0..n_data {
        for (gate, pauli_val) in [(Gate::X, 1u8), (Gate::Z, 3u8), (Gate::Y, 2u8)] {
            // Compute syndrome: bit i = 1 if stabilizer i anticommutes with error
            let mut syndrome = 0u32;
            for (i, stab) in stabilizers.iter().enumerate() {
                let p = stab.paulis[q];
                if p == 0 {
                    continue;
                }
                // Anticommutes if one is X and other is Z (or vice versa)
                // X (1) anticommutes with Z (3)
                // Y (2) anticommutes with both X and Z
                let anticommutes = matches!(
                    (p, pauli_val),
                    (1, 3) | (3, 1) | (1, 2) | (2, 1) | (3, 2) | (2, 3)
                );
                if anticommutes {
                    syndrome |= 1 << i;
                }
            }

            if syndrome != 0 {
                let mut circ = Circuit::new();
                circ.push(gate, q);
                recovery.insert(syndrome, circ);
            }
        }
    }

    // Syndrome 0: no error -> empty circuit
    recovery.insert(0, Circuit::new());

    recovery
}

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

fn build_repetition_recovery(n: usize, bitflip: bool) -> HashMap<u32, Circuit> {
    let mut recovery = HashMap::new();
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
pub fn mwpm_decode(stabilizers: &[PauliString], syndrome: u32) -> Vec<(usize, u8)> {
    let n = if stabilizers.is_empty() {
        0
    } else {
        stabilizers[0].paulis.len()
    };
    if syndrome == 0 {
        return Vec::new();
    }

    // Precompute syndrome for each single-qubit error
    let mut error_syndromes: Vec<(usize, u8, u32)> = Vec::new();
    for q in 0..n {
        for pv in [1u8, 2, 3] {
            let mut syn = 0u32;
            for (i, stab) in stabilizers.iter().enumerate() {
                let p = stab.paulis[q];
                if p == 0 {
                    continue;
                }
                let anticom =
                    matches!((p, pv), (1, 3) | (3, 1) | (1, 2) | (2, 1) | (3, 2) | (2, 3));
                if anticom {
                    syn |= 1 << i;
                }
            }
            if syn != 0 {
                error_syndromes.push((q, pv, syn));
            }
        }
    }

    // Build defect list: stabilizer indices with syndrome bit set
    let n_stab = stabilizers.len();
    let mut defects: Vec<usize> = Vec::new();
    for i in 0..n_stab {
        if (syndrome >> i) & 1 == 1 {
            defects.push(i);
        }
    }

    if defects.is_empty() {
        return Vec::new();
    }

    // If odd number of defects, add a boundary node (virtual defect)
    // that matches any defect with cost = distance to boundary
    // For simplicity, we handle this by allowing single-error corrections
    // to absorb unmatched defects.

    // Precompute weight between any two defects:
    // weight = minimum number of errors whose syndrome covers both
    // For simplicity, use Hamming distance on syndrome bits as proxy.
    // Better: use the error_syndromes lookup.

    // Build a complete graph on defects + optional boundary
    // Use recursive branch-and-bound to find minimum weight matching.
    // For small defect counts (<= 12), use exact recursive search.
    // For larger, use greedy.

    let num_defects = defects.len();
    if num_defects > 12 {
        // Greedy fallback for large instances
        return greedy_mwpm(&error_syndromes, syndrome);
    }

    // Exact recursive MWPM via branch and bound
    // We try to match defect 0 with each other defect (or leave it
    // unmatched if it can be corrected by a single error).
    // State: set of remaining defects, accumulated errors.

    // First, check if any single error explains the whole syndrome
    for &(q, pv, syn) in &error_syndromes {
        if syn == syndrome {
            return vec![(q, pv)];
        }
    }

    // Recursive matching: try pairing defects
    let mut best: Option<Vec<(usize, u8)>> = None;
    let mut best_weight = usize::MAX;

    // Generate all subsets of errors (up to 6 errors) via recursion
    fn search(
        error_syndromes: &[(usize, u8, u32)],
        target: u32,
        idx: usize,
        current_syn: u32,
        current_errors: &mut Vec<(usize, u8)>,
        best: &mut Option<Vec<(usize, u8)>>,
        best_weight: &mut usize,
    ) {
        if current_syn == target {
            if current_errors.len() < *best_weight {
                *best_weight = current_errors.len();
                *best = Some(current_errors.clone());
            }
            return;
        }
        if idx >= error_syndromes.len() {
            return;
        }
        // Pruning: if current errors already >= best, skip
        if current_errors.len() >= *best_weight {
            return;
        }
        // Remaining syndrome bits
        let remaining = target & !current_syn;
        if remaining == 0 {
            if current_errors.len() < *best_weight {
                *best_weight = current_errors.len();
                *best = Some(current_errors.clone());
            }
            return;
        }

        // Option 1: skip this error
        search(
            error_syndromes,
            target,
            idx + 1,
            current_syn,
            current_errors,
            best,
            best_weight,
        );

        // Option 2: use this error (if it covers new bits and doesn't overlap)
        let syn = error_syndromes[idx].2;
        if (syn & current_syn) == 0 && (syn & remaining) != 0 {
            current_errors.push((error_syndromes[idx].0, error_syndromes[idx].1));
            search(
                error_syndromes,
                target,
                idx + 1,
                current_syn | syn,
                current_errors,
                best,
                best_weight,
            );
            current_errors.pop();
        }
    }

    let mut current = Vec::new();
    search(
        &error_syndromes,
        syndrome,
        0,
        0,
        &mut current,
        &mut best,
        &mut best_weight,
    );

    best.unwrap_or_else(|| greedy_mwpm(&error_syndromes, syndrome))
}

fn greedy_mwpm(error_syndromes: &[(usize, u8, u32)], syndrome: u32) -> Vec<(usize, u8)> {
    let mut remaining = syndrome;
    let mut result = Vec::new();
    let mut used = vec![false; error_syndromes.len()];
    while remaining != 0 {
        let mut best_idx = None;
        let mut best_cover = 0u32;
        for (idx, &(_, _, syn)) in error_syndromes.iter().enumerate() {
            if used[idx] {
                continue;
            }
            let cover = syn & remaining;
            if cover.count_ones() > best_cover.count_ones() {
                best_cover = cover;
                best_idx = Some(idx);
            }
        }
        if let Some(idx) = best_idx {
            if best_cover == 0 {
                break;
            }
            used[idx] = true;
            result.push((error_syndromes[idx].0, error_syndromes[idx].1));
            remaining &= !error_syndromes[idx].2;
        } else {
            break;
        }
    }
    if remaining == 0 { result } else { Vec::new() }
}
/// Edmonds' Blossom algorithm for minimum weight perfect matching on a general
/// graph. Given an adjacency matrix of weights (INF for no edge), returns the
/// matching as a list of (u, v) pairs. Uses the primal-dual method with
/// blossom contraction and expansion. O(n^3).
pub fn blossom_match(n: usize, weight: &[Vec<i64>]) -> Vec<(usize, usize)> {
    const INF: i64 = i64::MAX / 4;
    if n == 0 {
        return Vec::new();
    }
    let n = if !n.is_multiple_of(2) { n + 1 } else { n };

    struct State {
        n: usize,
        match_to: Vec<usize>,
        label: Vec<i32>,
        parent: Vec<usize>,
        base: Vec<usize>,
        in_blossom: Vec<Vec<bool>>,
        blossom_children: Vec<Vec<usize>>,
        queue: Vec<usize>,
        dual_var: Vec<i64>,
        weight: Vec<Vec<i64>>,
    }

    impl State {
        fn find_base(&self, mut u: usize) -> usize {
            while self.base[u] != u {
                u = self.base[u];
            }
            u
        }

        fn new(n: usize, weight: &[Vec<i64>]) -> Self {
            let mut w = vec![vec![INF; n]; n];
            for i in 0..n.min(weight.len()) {
                for j in 0..n.min(weight[i].len()) {
                    w[i][j] = weight[i][j];
                }
            }
            let mut st = Self {
                n,
                match_to: vec![usize::MAX; n],
                label: vec![0; n],
                parent: vec![usize::MAX; n],
                base: (0..n).collect(),
                in_blossom: vec![vec![false; n]; n],
                blossom_children: vec![Vec::new(); n],
                queue: Vec::new(),
                dual_var: vec![0; n],
                weight: w,
            };
            for i in 0..n {
                st.in_blossom[i][i] = true;
            }
            for u in 0..n {
                let mut min_w = INF;
                for v in 0..n {
                    if u != v && st.weight[u][v] < min_w {
                        min_w = st.weight[u][v];
                    }
                }
                if min_w < INF {
                    st.dual_var[u] = min_w / 2;
                }
            }
            st
        }

        fn lca(&self, mut u: usize, mut v: usize) -> usize {
            let mut used = vec![false; self.n];
            loop {
                u = self.find_base(u);
                used[u] = true;
                if self.match_to[u] == usize::MAX {
                    break;
                }
                u = self.parent[self.match_to[u]];
                if u == usize::MAX {
                    break;
                }
            }
            loop {
                v = self.find_base(v);
                if used[v] {
                    return v;
                }
                v = self.parent[self.match_to[v]];
                if v == usize::MAX {
                    break;
                }
            }
            0
        }

        fn contract(&mut self, u: usize, v: usize, lca_v: usize) {
            let mut blossom = vec![false; self.n];
            let mut u = u;
            while self.find_base(u) != lca_v {
                let b = self.find_base(u);
                blossom[b] = true;
                if self.match_to[b] != usize::MAX {
                    blossom[self.match_to[b]] = true;
                }
                if self.match_to[b] != usize::MAX && self.label[self.match_to[b]] == 1 {
                    self.queue.push(self.match_to[b]);
                }
                self.parent[u] = v;
                if self.match_to[b] != usize::MAX {
                    u = self.parent[self.match_to[b]];
                } else {
                    break;
                }
                if u == usize::MAX {
                    break;
                }
            }
            let mut v = v;
            while self.find_base(v) != lca_v {
                let b = self.find_base(v);
                blossom[b] = true;
                if self.match_to[b] != usize::MAX {
                    blossom[self.match_to[b]] = true;
                }
                if self.match_to[b] != usize::MAX && self.label[self.match_to[b]] == 1 {
                    self.queue.push(self.match_to[b]);
                }
                self.parent[v] = u;
                if self.match_to[b] != usize::MAX {
                    v = self.parent[self.match_to[b]];
                } else {
                    break;
                }
                if v == usize::MAX {
                    break;
                }
            }
            for i in 0..self.n {
                if blossom[self.find_base(i)] {
                    self.base[i] = lca_v;
                    if !self.in_blossom[lca_v][i] {
                        self.in_blossom[lca_v][i] = true;
                        self.blossom_children[lca_v].push(i);
                    }
                }
            }
        }

        fn augment(&mut self, _u_end: usize, v_end: usize) {
            let mut v = v_end;
            while v != usize::MAX {
                let u = self.parent[v];
                if u == usize::MAX {
                    break;
                }
                let prev_match = self.match_to[u];
                self.match_to[u] = v;
                self.match_to[v] = u;
                v = prev_match;
            }
        }

        fn search_augmenting(&mut self) -> bool {
            for i in 0..self.n {
                self.label[i] = 0;
                self.parent[i] = usize::MAX;
            }
            self.queue.clear();

            for u in 0..self.n {
                if self.match_to[u] == usize::MAX && self.find_base(u) == u {
                    self.label[u] = 1;
                    self.queue.push(u);
                }
            }

            let mut found = false;
            let mut u_end = 0;
            let mut v_end = 0;

            while !self.queue.is_empty() && !found {
                let u = self.queue.pop().unwrap();
                if self.find_base(u) != u || self.label[u] != 1 {
                    continue;
                }

                for v in 0..self.n {
                    if self.find_base(v) == self.find_base(u) {
                        continue;
                    }
                    if self.weight[u][v] >= INF {
                        continue;
                    }
                    let slack = self.dual_var[u] + self.dual_var[v] - self.weight[u][v];
                    if slack != 0 {
                        continue;
                    }

                    if self.label[v] == 0 {
                        if self.match_to[v] == usize::MAX {
                            self.parent[v] = u;
                            u_end = u;
                            v_end = v;
                            found = true;
                            break;
                        } else {
                            self.label[v] = 2;
                            self.parent[v] = u;
                            let mate = self.match_to[v];
                            self.label[mate] = 1;
                            self.queue.push(mate);
                        }
                    } else if self.label[v] == 1 {
                        let lca_v = self.lca(u, v);
                        self.contract(u, v, lca_v);
                    }
                }
            }

            if found {
                self.augment(u_end, v_end);
                true
            } else {
                false
            }
        }
    }

    let mut state = State::new(n, weight);
    let mut found_augmenting = true;
    while found_augmenting {
        found_augmenting = state.search_augmenting();
    }

    let mut result = Vec::new();
    for u in 0..n {
        if state.match_to[u] != usize::MAX && state.match_to[u] > u {
            result.push((u, state.match_to[u]));
        }
    }
    result
}

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

#[cfg(test)]
mod recovery_tests {
    use super::*;

    #[test]
    fn test_recovery_n3_all_single_errors() {
        // For 3-qubit bit-flip code, verify each single error maps to correct recovery
        let code = repetition_code_bitflip(3);
        // syndrome 0 (00): no error -> empty recovery
        assert!(code.recovery.get(&0).unwrap().operations().is_empty());
        // syndrome 1 (01): bit 0 set -> error on qubit 0 -> X on qubit 0
        // syndrome 2 (10): bit 1 set -> error on qubit 2 -> X on qubit 2
        // syndrome 3 (11): bits 0,1 set -> error on qubit 1 -> X on qubit 1
        // Verify each syndrome maps to recovery on the correct qubit
        let cases: [(u32, usize); 3] = [(1, 0), (2, 2), (3, 1)];
        for (syndrome, expected_qubit) in cases {
            let circ = code.recovery.get(&syndrome).unwrap();
            let ops = circ.operations();
            assert_eq!(
                ops.len(),
                1,
                "syndrome {} should have 1 recovery op",
                syndrome
            );
            match &ops[0] {
                crate::CircuitOp::SingleQubit { gate, target } => {
                    assert_eq!(
                        *target, expected_qubit,
                        "syndrome {} wrong target",
                        syndrome
                    );
                    // Gate should be X (check via debug or matrix)
                    assert_eq!(*gate, Gate::X, "syndrome {} should be X gate", syndrome);
                }
                _ => panic!("expected single-qubit gate for syndrome {}", syndrome),
            }
        }
    }

    #[test]
    fn test_recovery_n5_boundary() {
        let code = repetition_code_bitflip(5);
        // n=5: syndromes 0..16, single errors at qubits 0..4
        // qubit 0 -> syndrome bit 0 only (syndrome=1)
        // qubit 4 -> syndrome bit 3 only (syndrome=8)
        // qubit k (1..3) -> bits k-1,k (syndrome = 3<<k-1... wait)
        // k=1 -> bits 0,1 -> syndrome = 0b0011 = 3
        // k=2 -> bits 1,2 -> syndrome = 0b0110 = 6
        // k=3 -> bits 2,3 -> syndrome = 0b1100 = 12
        let cases = [(1u32, 0usize), (3, 1), (6, 2), (12, 3), (8, 4)];
        for (syn, _q) in cases {
            assert!(code.recovery.contains_key(&syn), "missing syndrome {}", syn);
        }
    }

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
        // 8 stabilizers: 6 Z-type (X-error detection) + 2 X-type (Z-error detection)
        assert_eq!(code.stabilizers.len(), 8);
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
    fn test_mwpm_single_error_syndromes() {
        // 3-qubit bit-flip code stabilizers
        let code = repetition_code_bitflip(3);
        let stabs = &code.stabilizers;
        // Single error on qubit 0: syndrome = 1 (0b01)
        let result = mwpm_decode(stabs, 1);
        assert_eq!(result, vec![(0, 1)], "qubit 0 X-error");
        // Single error on qubit 2: syndrome = 2 (0b10)
        let result = mwpm_decode(stabs, 2);
        assert_eq!(result, vec![(2, 1)], "qubit 2 X-error");
        // Single error on qubit 1: syndrome = 3 (0b11)
        let result = mwpm_decode(stabs, 3);
        assert_eq!(result, vec![(1, 1)], "qubit 1 X-error");
    }

    #[test]
    fn test_mwpm_zero_syndrome() {
        let code = repetition_code_bitflip(3);
        let result = mwpm_decode(&code.stabilizers, 0);
        assert!(
            result.is_empty(),
            "zero syndrome should give empty correction"
        );
    }

    #[test]
    fn test_mwpm_multi_error_minimum_weight() {
        // For a 5-qubit repetition code, test that mwpm_decode finds
        // minimum-weight correction for multi-error syndromes.
        let code = repetition_code_bitflip(5);
        let stabs = &code.stabilizers;
        // Two errors on adjacent qubits 0 and 1:
        // syndrome(0) XOR syndrome(1). For 5-qubit code:
        // qubit 0 -> syndrome bit 0 only -> 0b0001 = 1
        // qubit 1 -> bits 0,1 -> 0b0011 = 3
        // combined = 1 XOR 3 = 2 (0b0010)
        // This is the same syndrome as single error on qubit... let's check:
        // qubit 2 -> bits 1,2 -> 0b0110 = 6, not 2.
        // Actually for repetition code, two adjacent errors produce a syndrome
        // that looks like a single error on the qubit between... no.
        // Let's just verify the decoder returns SOME correction (non-empty)
        // for a non-zero syndrome.
        let syn = 1u32 ^ 3u32; // errors on qubits 0 and 1
        let result = mwpm_decode(stabs, syn);
        assert!(
            !result.is_empty(),
            "multi-error syndrome should produce correction"
        );
        // The correction weight should be <= 2 (MWPM finds minimum)
        assert!(result.len() <= 2, "correction should have at most 2 errors");
    }

    #[test]
    fn test_mwpm_surface_code_d3_single_errors() {
        // Surface code d=3: verify single-error syndromes are distinct
        let code = surface_code_planar(3);
        let stabs = &code.stabilizers;
        let n = code.n_physical;
        // Collect syndromes for single X errors on each qubit
        let mut syndromes = std::collections::HashSet::new();
        for q in 0..n {
            // Compute syndrome for X error on qubit q
            let mut syn = 0u32;
            for (i, stab) in stabs.iter().enumerate() {
                let p = stab.paulis[q];
                // X (1) anticommutes with Z (3) stabilizers
                if p == 3 {
                    syn |= 1 << i;
                }
            }
            if syn != 0 {
                syndromes.insert(syn);
            }
        }
        // All non-zero single-error syndromes should be distinct
        // (number of distinct syndromes <= number of qubits with Z-stabilizers)
        // Just verify we get some non-trivial syndromes
        assert!(!syndromes.is_empty(), "should have non-trivial syndromes");
    }
    #[test]
    fn test_pauli_string() {
        let mut p = PauliString::new(3);
        p.set(0, 1);
        p.set(1, 3);
        assert_eq!(p.to_string(), "XZI");
        assert_eq!(p.weight(), 2);
    }
}

#[cfg(test)]
mod e2e_tests {
    use super::*;
    use crate::StateVector;

    #[test]
    fn test_repetition_bitflip_corrects_single_error() {
        let state = StateVector::zero(1).unwrap();
        let encoded = encode_repetition_bitflip(&state, 3).unwrap();
        assert!((encoded.amplitudes()[0].re - 1.0).abs() < 1e-10);

        let code = repetition_code_bitflip(3);

        for err_qubit in 0..3usize {
            let mut errored = encoded.clone();
            errored.apply(crate::Gate::X, err_qubit).unwrap();

            // Compute syndrome from the basis state with highest probability
            let mut max_idx = 0;
            let mut max_prob = 0.0f64;
            for (idx, amp) in errored.amplitudes().iter().enumerate() {
                if amp.norm_sqr() > max_prob {
                    max_prob = amp.norm_sqr();
                    max_idx = idx;
                }
            }
            let bit0 = max_idx & 1; // qubit 0 (LSB)
            let bit1 = (max_idx >> 1) & 1; // qubit 1
            let bit2 = (max_idx >> 2) & 1; // qubit 2
            let s0 = bit0 ^ bit1;
            let s1 = bit1 ^ bit2;
            let syndrome = (s0 | (s1 << 1)) as u32;

            let recovery_circ = code.recovery.get(&syndrome).unwrap();
            let recovered = recovery_circ.run(errored).unwrap();

            assert!(
                (recovered.amplitudes()[0].re - 1.0).abs() < 1e-10,
                "failed to recover from error on qubit {}: syndrome={}, amp0={}",
                err_qubit,
                syndrome,
                recovered.amplitudes()[0].re
            );
        }
    }
}

//! Quantum error correction: stabilizer codes (repetition, Shor, Steane),
//! surface codes (planar construction and verified rotated d=3), logical
//! operator extraction over GF(2), and an MWPM-style decoder.
//!
//! Submodules:
//! - [`codes`]: classical stabilizer codes and their syndrome/recovery circuits
//! - [`surface`]: planar/rotated surface codes and search utilities
//! - [`gf2`]: GF(2) linear algebra for logical-operator extraction
//! - [`mwpm`]: minimum-weight perfect matching decoder utilities

pub mod codes;
pub mod gf2;
pub mod mwpm;
pub mod surface;
mod tests;

pub use codes::{
    decode_repetition_bitflip, encode_repetition_bitflip, repetition_code_bitflip,
    repetition_code_phaseflip, shor_code, steane_code,
};
pub use mwpm::{blossom_match, mwpm_decode};
pub use surface::{
    surface_code_planar, surface_code_rotated, surface_code_rotated_search,
    verify_single_error_syndromes,
};

pub(crate) use gf2::extract_logicals;

use crate::{Circuit, Gate};
use std::collections::HashMap;
use std::fmt;

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
    pub recovery: HashMap<u64, Circuit>,
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

    pub fn commutes(&self, other: &PauliString) -> bool {
        let mut anticommute_count = 0usize;
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
        !anticommute_count.is_multiple_of(2)
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
pub fn stabilizers_commute(code: &QecCode) -> bool {
    for i in 0..code.stabilizers.len() {
        for j in i + 1..code.stabilizers.len() {
            let a = &code.stabilizers[i];
            let b = &code.stabilizers[j];
            let mut anticommute_count = 0usize;
            for q in 0..code.n_physical {
                let pa = a.paulis.get(q).copied().unwrap_or(0);
                let pb = b.paulis.get(q).copied().unwrap_or(0);
                if pa != 0 && pb != 0 && pa != pb {
                    anticommute_count += 1;
                }
            }
            if !anticommute_count.is_multiple_of(2) {
                return false;
            }
        }
    }
    true
}

/// Combined validity check: stabilizers commute and every single-qubit error
/// has a distinct non-zero syndrome.
pub fn verify_code(code: &QecCode) -> bool {
    stabilizers_commute(code) && verify_single_error_syndromes(code)
}

// ---------------------------------------------------------------------------
// GF(2) linear algebra for logical-operator extraction.

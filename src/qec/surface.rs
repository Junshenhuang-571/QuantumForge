//! Surface codes: planar face-stabilizer construction, rotated d=3
//! (verified), an experimental boundary-stabilizer search, and the
//! single-error syndrome verifier.

use super::extract_logicals;
use super::verify_code;
use super::{Circuit, Gate, PauliString, QecCode};
use std::collections::HashMap;

/// Build the planar (unrotated) surface code of distance `d` on a `d`x`d` grid.
///
/// Uses the `(d-1)^2` alternating X/Z face stabilizers only, with no boundary
/// stabilizers, so the code encodes `2*d - 1` logical qubits rather than one;
/// the logical basis is recovered at runtime by GF(2) extraction. Panics unless
/// `d` is odd and at least 3.
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

    // Logical operators. This construction uses face stabilizers only (no
    // boundary stabilizers), so it encodes n - (d-1)^2 = 2d-1 logical qubits,
    // not one. The logical basis is computed at runtime so it is always a
    // correct symplectic pair basis (verified by tests::qec_planar_logicals).
    let mut xrows: Vec<Vec<usize>> = Vec::new();
    let mut zrows: Vec<Vec<usize>> = Vec::new();
    for stab in &stabilizers {
        let mut xs: Vec<usize> = Vec::new();
        let mut zs: Vec<usize> = Vec::new();
        for (i, p) in stab.paulis.iter().enumerate() {
            if *p == 1 {
                xs.push(i);
            } else if *p == 3 {
                zs.push(i);
            }
        }
        if !xs.is_empty() {
            xrows.push(xs);
        }
        if !zs.is_empty() {
            zrows.push(zs);
        }
    }
    let logicals = extract_logicals(n, &xrows, &zrows);
    let mut logical_x = Vec::new();
    let mut logical_z = Vec::new();
    for (xs, zs) in logicals {
        let mut lx = PauliString::new(n);
        for q in xs {
            lx.set(q, 1);
        }
        let mut lz = PauliString::new(n);
        for q in zs {
            lz.set(q, 3);
        }
        logical_x.push(lx);
        logical_z.push(lz);
    }
    let n_logical = logical_x.len();

    let syndrome_circuit = build_surface_syndrome(n, &stabilizers);
    let recovery = build_surface_recovery(n, &stabilizers);

    QecCode {
        n_physical: n,
        n_logical,
        stabilizers,
        logical_x,
        logical_z,
        syndrome_circuit,
        recovery,
    }
}

/// Check that every single-qubit Pauli error on `code` produces a distinct,
/// non-zero syndrome (a distance >= 3 necessary condition). Returns true when
/// all `3 * n_physical` errors are distinguishable.
pub fn verify_single_error_syndromes(code: &QecCode) -> bool {
    use std::collections::HashSet;
    let n = code.n_physical;
    let mut seen: HashSet<Vec<u8>> = HashSet::new();
    for q in 0..n {
        for p in 1u8..=3u8 {
            // syndrome: for each stabilizer, commutation parity
            let mut syndrome = Vec::with_capacity(code.stabilizers.len());
            for stab in &code.stabilizers {
                // parity of Pauli commutation between error (p on q) and stab
                let stab_pauli = stab.paulis.get(q).copied().unwrap_or(0);
                let commute = pauli_commute(stab_pauli, p);
                syndrome.push(if commute { 0 } else { 1 });
            }
            if syndrome.iter().all(|&b| b == 0) {
                return false; // undetectable error
            }
            if !seen.insert(syndrome) {
                return false; // duplicate syndrome
            }
        }
    }
    true
}

pub(crate) fn pauli_commute(a: u8, b: u8) -> bool {
    // Pauli labels: 0=I,1=X,2=Y,3=Z. Commute unless one is X and other Z (or vice versa) etc.
    // X(1) and Z(3) anticommute; Y(2) anticommutes with X and Z.
    if a == 0 || b == 0 {
        return true;
    }
    if a == b {
        return true;
    }
    // X and Z anticommute; X and Y anticommute; Y and Z anticommute
    false
}
/// Attempt to build a rotated surface code for distance `d` by augmenting the
/// face stabilizers with a greedy search over boundary weight-2 stabilizers.
/// Returns `Some(code)` if the search finds a set satisfying
pub fn surface_code_rotated_search(d: usize) -> Option<QecCode> {
    if d < 3 || d.is_multiple_of(2) {
        return None;
    }
    // Face stabilizers (same as surface_code_rotated).
    let n = d * d;
    let mut stabilizers: Vec<PauliString> = Vec::new();
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
                }
            } else {
                for &q in &qs {
                    pauli.set(q, 3);
                }
            }
            stabilizers.push(pauli);
        }
    }

    // Candidate boundary stabilizers: weight-2 on adjacent boundary qubits.
    let is_boundary = |q: usize| -> bool {
        let r = q / d;
        let c = q % d;
        r == 0 || r == d - 1 || c == 0 || c == d - 1
    };
    let mut candidates: Vec<PauliString> = Vec::new();
    for q1 in 0..n {
        if !is_boundary(q1) {
            continue;
        }
        for delta in [1isize, -1, d as isize, -(d as isize)] {
            let q2i = q1 as isize + delta;
            if q2i < 0 || q2i >= n as isize {
                continue;
            }
            let q2 = q2i as usize;
            if !is_boundary(q2) {
                continue;
            }
            // adjacency check (Manhattan distance 1)
            let r1 = q1 / d;
            let c1 = q1 % d;
            let r2 = q2 / d;
            let c2 = q2 % d;
            if (r1 as isize - r2 as isize).abs() + (c1 as isize - c2 as isize).abs() != 1 {
                continue;
            }
            for &pauli_type in &[1u8, 3u8] {
                let mut p = PauliString::new(n);
                p.set(q1, pauli_type);
                p.set(q2, pauli_type);
                candidates.push(p);
            }
        }
    }
    // dedupe
    candidates.sort_by(|a, b| a.paulis.cmp(&b.paulis));
    candidates.dedup_by(|a, b| a.paulis == b.paulis);

    // Greedy add.
    let mut current = stabilizers.clone();
    let mut used: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for _ in 0..candidates.len() {
        if verify_with_stabilizers(n, &current) {
            break;
        }
        // pick candidate maximizing distinct count
        let mut best_idx = None;
        let mut best_score = score_stabilizers(n, &current);
        for (i, cand) in candidates.iter().enumerate() {
            if used.contains(&i) {
                continue;
            }
            let mut trial = current.clone();
            // candidate must commute with all current stabilizers
            let mut commutes = true;
            for s in &current {
                let a = s;
                let b = cand;
                let mut ac = 0usize;
                for q in 0..n {
                    let pa = a.paulis.get(q).copied().unwrap_or(0);
                    let pb = b.paulis.get(q).copied().unwrap_or(0);
                    if pa != 0 && pb != 0 && pa != pb {
                        ac += 1;
                    }
                }
                if !ac.is_multiple_of(2) {
                    commutes = false;
                    break;
                }
            }
            if !commutes {
                continue;
            }
            trial.push(cand.clone());
            let s = score_stabilizers(n, &trial);
            if s > best_score {
                best_score = s;
                best_idx = Some(i);
            }
        }
        match best_idx {
            Some(i) => {
                current.push(candidates[i].clone());
                used.insert(i);
            }
            None => break,
        }
    }

    if !verify_with_stabilizers(n, &current) {
        return None;
    }

    // Build QecCode similar to planar (logical ops top row / left col).
    let mut logical_x = PauliString::new(n);
    for c in 0..d {
        logical_x.set(c, 1);
    }
    let mut logical_z = PauliString::new(n);
    for r in 0..d {
        logical_z.set(r * d, 3);
    }
    Some(QecCode {
        n_physical: n,
        n_logical: 1,
        stabilizers: current,
        logical_x: vec![logical_x],
        logical_z: vec![logical_z],
        syndrome_circuit: Circuit::new(),
        recovery: std::collections::HashMap::new(),
    })
}

fn score_stabilizers(n: usize, stabs: &[PauliString]) -> usize {
    use std::collections::HashSet;
    let mut seen: HashSet<Vec<u8>> = HashSet::new();
    let mut distinct = 0usize;
    let mut zero_count = 0usize;
    for q in 0..n {
        for p in 1u8..=3u8 {
            let mut syndrome = Vec::with_capacity(stabs.len());
            for stab in stabs {
                let stab_pauli = stab.paulis.get(q).copied().unwrap_or(0);
                let commute = pauli_commute(stab_pauli, p);
                syndrome.push(if commute { 0 } else { 1 });
            }
            if syndrome.iter().all(|&b| b == 0) {
                zero_count += 1;
            } else if seen.insert(syndrome) {
                distinct += 1;
            }
        }
    }
    // prefer fewer zero syndromes and more distinct
    distinct * 1000 - zero_count
}

fn verify_with_stabilizers(n: usize, stabs: &[PauliString]) -> bool {
    // reuse logic
    let code = QecCode {
        n_physical: n,
        n_logical: 1,
        stabilizers: stabs.to_vec(),
        logical_x: vec![],
        logical_z: vec![],
        syndrome_circuit: Circuit::new(),
        recovery: std::collections::HashMap::new(),
    };
    verify_code(&code)
}

/// Build the rotated surface code, which encodes one logical qubit.
///
/// Only `d = 3` is supported: the four weight-2 boundary stabilizers are
/// hard-coded and were verified so that all 27 single-qubit errors have
/// distinct syndromes, and the logical X/Z operators were found by exhaustive
/// search. Panics for any other distance.
pub fn surface_code_rotated(d: usize) -> QecCode {
    assert!(
        d == 3,
        "rotated surface code boundary stabilizers are only verified for d=3; \
         d>3 is not supported"
    );

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

    // Logical operators. These specific operators were found by exhaustive
    // search and verified computationally: they commute with every
    // stabilizer, anticommute with each other, and are outside the stabilizer
    // group (see tests::qec rotated code logical verification).
    let mut logical_x = PauliString::new(n);
    for q in [0usize, 2, 5] {
        logical_x.set(q, 1);
    }
    let mut logical_z = PauliString::new(n);
    for q in [1usize, 2, 3] {
        logical_z.set(q, 3);
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
pub(crate) fn build_surface_recovery(
    n_data: usize,
    stabilizers: &[PauliString],
) -> HashMap<u64, Circuit> {
    let mut recovery: HashMap<u64, Circuit> = HashMap::new();
    let _ = stabilizers.len();

    // For each single-qubit error, compute syndrome and map to recovery
    for q in 0..n_data {
        for (gate, pauli_val) in [(Gate::X, 1u8), (Gate::Z, 3u8), (Gate::Y, 2u8)] {
            // Compute syndrome: bit i = 1 if stabilizer i anticommutes with error
            let mut syndrome = 0u64;
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

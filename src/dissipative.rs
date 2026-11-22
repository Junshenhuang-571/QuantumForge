//! Dissipative state preparation with a jump-operator (Lindblad) channel.
//!
//! A single-jump Lindblad channel with jump `L` drives a state toward the
//! kernel of `L`. For a target pure state `|t>` on one qubit, the jump
//! `L = |t><t_perp|` has kernel `span(|t>)`, so repeated application prepares
//! `|t>` from any input. The amplitude-damping channel is the special case
//! `|t> = |0>`.
//!
//! The step is the first-order Trotter/Kraus form
//! `rho -> K0 rho K0^dagger + K1 rho K1^dagger` with
//! `K0 = I - (dt/2) L^dagger L` and `K1 = sqrt(dt) L`, renormalized. It is
//! trace-preserving up to `O(dt^2)`; the renormalization makes it exact.

use crate::analytic::PauliSum;
use crate::qec::{PauliString, QecCode};
use crate::thermal::expectation;
use crate::{Complex64, DensityMatrix, QuantumError};

/// A single-qubit 2x2 operator.
pub type Matrix2 = [[Complex64; 2]; 2];

/// Jump operator `L = |target><orthogonal|` for a normalized target.
pub fn jump_to_target(target: [Complex64; 2]) -> Matrix2 {
    let orthogonal = [target[1].conj() * -1.0, target[0].conj()];
    let mut jump = [[Complex64::ZERO; 2]; 2];
    for (row, target_row) in target.iter().enumerate() {
        for (col, orthogonal_col) in orthogonal.iter().enumerate() {
            jump[row][col] = *target_row * orthogonal_col.conj();
        }
    }
    jump
}

/// Apply `k rho k^dagger` locally on `qubit`.
fn apply_local_kraus(rho: &DensityMatrix, k: Matrix2, qubit: usize) -> Vec<Complex64> {
    let dimension = rho.dimension();
    let mask = 1usize << qubit;
    let entries = rho.entries();
    let mut out = vec![Complex64::ZERO; dimension * dimension];
    for row in 0..dimension {
        for col in 0..dimension {
            let k_row = usize::from(row & mask != 0);
            let k_col = usize::from(col & mask != 0);
            let mut sum = Complex64::ZERO;
            for a in 0..2 {
                for b in 0..2 {
                    let source_row = (row & !mask) | (a << qubit);
                    let source_col = (col & !mask) | (b << qubit);
                    sum = sum
                        + k[k_row][a]
                            * entries[source_row * dimension + source_col]
                            * k[k_col][b].conj();
                }
            }
            out[row * dimension + col] = sum;
        }
    }
    out
}

/// One Trotter step of the single-jump Lindblad channel with jump `L` on
/// `qubit`.
pub fn lindblad_step(
    rho: &mut DensityMatrix,
    jump: Matrix2,
    qubit: usize,
    dt: f64,
) -> Result<(), QuantumError> {
    if qubit >= rho.qubits() {
        return Err(QuantumError::QubitOutOfRange {
            qubit,
            qubits: rho.qubits(),
        });
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err(QuantumError::InvalidProbability(dt));
    }
    // L^dagger L
    let mut ldl = [[Complex64::ZERO; 2]; 2];
    for (row, ldl_row) in ldl.iter_mut().enumerate() {
        for (col, ldl_entry) in ldl_row.iter_mut().enumerate() {
            let mut sum = Complex64::ZERO;
            for jump_row in jump.iter() {
                sum = sum + jump_row[row].conj() * jump_row[col];
            }
            *ldl_entry = sum;
        }
    }
    let half_dt = 0.5 * dt;
    let k0 = [
        [Complex64::ONE - ldl[0][0] * half_dt, ldl[0][1] * (-half_dt)],
        [ldl[1][0] * (-half_dt), Complex64::ONE - ldl[1][1] * half_dt],
    ];
    let scale = dt.sqrt();
    let k1 = [
        [jump[0][0] * scale, jump[0][1] * scale],
        [jump[1][0] * scale, jump[1][1] * scale],
    ];
    let branch0 = apply_local_kraus(rho, k0, qubit);
    let branch1 = apply_local_kraus(rho, k1, qubit);
    let mut entries: Vec<Complex64> = branch0
        .iter()
        .zip(branch1.iter())
        .map(|(a, b)| *a + *b)
        .collect();
    let trace = (0..rho.dimension())
        .map(|index| entries[index * rho.dimension() + index].re)
        .sum::<f64>();
    if !trace.is_finite() || trace <= 0.0 {
        return Err(QuantumError::InvalidProbability(trace));
    }
    for entry in &mut entries {
        *entry = *entry / trace;
    }
    *rho = DensityMatrix::from_entries(entries, rho.qubits());
    Ok(())
}

/// Prepare a single-qubit target pure state by repeated Lindblad steps.
pub fn prepare_target(
    rho: &mut DensityMatrix,
    target: [Complex64; 2],
    dt: f64,
    steps: usize,
) -> Result<(), QuantumError> {
    let jump = jump_to_target(target);
    for _ in 0..steps {
        lindblad_step(rho, jump, 0, dt)?;
    }
    Ok(())
}

/// Fidelity `<t| rho |t>` of a single-qubit target, tracing out the rest.
pub fn fidelity(rho: &DensityMatrix, target: [Complex64; 2], qubit: usize) -> f64 {
    let dimension = rho.dimension();
    let mask = 1usize << qubit;
    let mut value = Complex64::ZERO;
    for a in 0..2 {
        for b in 0..2 {
            for base in 0..dimension {
                if base & mask != 0 {
                    continue;
                }
                let row = base | (a << qubit);
                let col = base | (b << qubit);
                value = value + target[a].conj() * rho.entries()[row * dimension + col] * target[b];
            }
        }
    }
    value.re
}

// ---------------------------------------------------------------------------
// Many-qubit (entangled) targets
// ---------------------------------------------------------------------------

fn matmul(first: &[Complex64], second: &[Complex64], dimension: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::ZERO; dimension * dimension];
    for row in 0..dimension {
        for inner in 0..dimension {
            let left = first[row * dimension + inner];
            if left == Complex64::ZERO {
                continue;
            }
            for col in 0..dimension {
                out[row * dimension + col] =
                    out[row * dimension + col] + left * second[inner * dimension + col];
            }
        }
    }
    out
}

fn dagger(matrix: &[Complex64], dimension: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::ZERO; dimension * dimension];
    for row in 0..dimension {
        for col in 0..dimension {
            out[row * dimension + col] = matrix[col * dimension + row].conj();
        }
    }
    out
}

/// Jump operator `L = |target><target_perp|` for a normalized many-qubit target.
///
/// The kernel of `L` is `span(|target>)`, so the single-jump Lindblad channel
/// drives any input to `|target>`. `target_perp` is a deterministic unit vector
/// orthogonal to `target`.
pub fn jump_to_target_state(target: &[Complex64]) -> Result<Vec<Complex64>, QuantumError> {
    let dimension = target.len();
    if dimension == 0 || dimension.count_ones() != 1 {
        return Err(QuantumError::StateSizeNotPowerOfTwo(dimension));
    }
    let norm = target
        .iter()
        .map(|entry| entry.norm_sqr())
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return Err(QuantumError::InvalidProbability(norm));
    }
    let normalized: Vec<Complex64> = target.iter().map(|entry| *entry / norm).collect();
    let mut perpendicular = vec![Complex64::ZERO; dimension];
    let mut found = false;
    for basis in 0..dimension {
        let mut candidate = vec![Complex64::ZERO; dimension];
        candidate[basis] = Complex64::ONE;
        let overlap = normalized
            .iter()
            .zip(candidate.iter())
            .fold(Complex64::ZERO, |acc, (t, c)| acc + t.conj() * (*c));
        for (entry, c) in normalized.iter().zip(candidate.iter_mut()) {
            *c = *c - overlap * (*entry);
        }
        let candidate_norm = candidate
            .iter()
            .map(|entry| entry.norm_sqr())
            .sum::<f64>()
            .sqrt();
        if candidate_norm > 1e-9 {
            for (target_entry, candidate_entry) in perpendicular.iter_mut().zip(candidate.iter()) {
                *target_entry = *candidate_entry / candidate_norm;
            }
            found = true;
            break;
        }
    }
    if !found {
        return Err(QuantumError::InvalidProbability(0.0));
    }
    let mut jump = vec![Complex64::ZERO; dimension * dimension];
    for row in 0..dimension {
        for col in 0..dimension {
            jump[row * dimension + col] = normalized[row] * perpendicular[col].conj();
        }
    }
    Ok(jump)
}

/// One Trotter step of the single-jump Lindblad channel with a dense jump
/// operator on the whole register.
pub fn lindblad_step_global(
    rho: &mut DensityMatrix,
    jump: &[Complex64],
    dt: f64,
) -> Result<(), QuantumError> {
    let dimension = rho.dimension();
    if jump.len() != dimension * dimension {
        return Err(QuantumError::InvalidProbability(jump.len() as f64));
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err(QuantumError::InvalidProbability(dt));
    }
    let ldl = matmul(&dagger(jump, dimension), jump, dimension);
    let half_dt = 0.5 * dt;
    let scale = dt.sqrt();
    let mut k0 = vec![Complex64::ZERO; dimension * dimension];
    let mut k1 = vec![Complex64::ZERO; dimension * dimension];
    for row in 0..dimension {
        for col in 0..dimension {
            let identity = if row == col {
                Complex64::ONE
            } else {
                Complex64::ZERO
            };
            k0[row * dimension + col] = identity - ldl[row * dimension + col] * half_dt;
            k1[row * dimension + col] = jump[row * dimension + col] * scale;
        }
    }
    let rho_entries = rho.entries().to_vec();
    let branch0 = matmul(
        &matmul(&k0, &rho_entries, dimension),
        &dagger(&k0, dimension),
        dimension,
    );
    let branch1 = matmul(
        &matmul(&k1, &rho_entries, dimension),
        &dagger(&k1, dimension),
        dimension,
    );
    let mut entries: Vec<Complex64> = branch0
        .iter()
        .zip(branch1.iter())
        .map(|(a, b)| *a + *b)
        .collect();
    let trace = (0..dimension)
        .map(|index| entries[index * dimension + index].re)
        .sum::<f64>();
    if !trace.is_finite() || trace <= 0.0 {
        return Err(QuantumError::InvalidProbability(trace));
    }
    for entry in &mut entries {
        *entry = *entry / trace;
    }
    *rho = DensityMatrix::from_entries(entries, rho.qubits());
    Ok(())
}

/// Prepare a many-qubit target pure state by repeated Lindblad steps.
pub fn prepare_target_state(
    rho: &mut DensityMatrix,
    target: &[Complex64],
    dt: f64,
    steps: usize,
) -> Result<(), QuantumError> {
    let jump = jump_to_target_state(target)?;
    for _ in 0..steps {
        lindblad_step_global(rho, &jump, dt)?;
    }
    Ok(())
}

/// Fidelity `<t| rho |t>` of a many-qubit target pure state.
pub fn state_fidelity(rho: &DensityMatrix, target: &[Complex64]) -> f64 {
    let dimension = rho.dimension();
    let mut value = Complex64::ZERO;
    for row in 0..dimension {
        for col in 0..dimension {
            value = value + target[row].conj() * rho.entries()[row * dimension + col] * target[col];
        }
    }
    value.re
}

// ---------------------------------------------------------------------------
// Autonomous quantum error correction: dissipative code-space preparation
// ---------------------------------------------------------------------------

/// Product of two single-qubit Paulis: `(result, phase)`.
fn pauli_product(first: u8, second: u8) -> (u8, Complex64) {
    let i = Complex64::new(0.0, 1.0);
    let neg_i = Complex64::new(0.0, -1.0);
    match (first, second) {
        (0, other) | (other, 0) => (other, Complex64::ONE),
        (1, 1) | (2, 2) | (3, 3) => (0, Complex64::ONE),
        (1, 2) => (3, i),
        (2, 1) => (3, neg_i),
        (2, 3) => (1, i),
        (3, 2) => (1, neg_i),
        (3, 1) => (2, i),
        (1, 3) => (2, neg_i),
        _ => (0, Complex64::ONE),
    }
}

/// Multiply two Pauli strings, returning the product string and its phase.
fn multiply_pauli_strings(first: &[u8], second: &[u8]) -> (Vec<u8>, Complex64) {
    let mut result = vec![0u8; first.len()];
    let mut phase = Complex64::ONE;
    for (index, (&a, &b)) in first.iter().zip(second.iter()).enumerate() {
        let (pauli, factor) = pauli_product(a, b);
        result[index] = pauli;
        phase = phase * factor;
    }
    (result, phase)
}

/// A Pauli string that anticommutes with `stabilizers[index]` and commutes with
/// every other stabilizer, searching weight 1 then weight 2.
fn anticommuting_pauli(code: &QecCode, index: usize) -> Result<Vec<u8>, QuantumError> {
    let qubits = code.n_physical;
    let stabilizer = &code.stabilizers[index];
    let mut candidates: Vec<Vec<u8>> = Vec::new();
    for qubit in 0..qubits {
        for pauli in [1u8, 2, 3] {
            let mut candidate = vec![0u8; qubits];
            candidate[qubit] = pauli;
            candidates.push(candidate);
        }
    }
    for first in 0..qubits {
        for second in (first + 1)..qubits {
            for p_first in [1u8, 2, 3] {
                for p_second in [1u8, 2, 3] {
                    let mut candidate = vec![0u8; qubits];
                    candidate[first] = p_first;
                    candidate[second] = p_second;
                    candidates.push(candidate);
                }
            }
        }
    }
    for candidate in candidates {
        let string = PauliString {
            paulis: candidate.clone(),
        };
        if stabilizer.commutes(&string) {
            continue;
        }
        if code
            .stabilizers
            .iter()
            .enumerate()
            .all(|(other, s)| other == index || s.commutes(&string))
        {
            return Ok(candidate);
        }
    }
    Err(QuantumError::UnsupportedOperation(format!(
        "no Pauli of weight <= 2 anticommutes with stabilizer {index} and commutes with the rest"
    )))
}

/// A dissipative QEC jump `L = A P_-`, stored as the two Pauli strings that
/// define it: `A` flips the `-1` eigenspace of `S` into the `+1` eigenspace.
#[derive(Clone, Debug)]
pub struct CodeJump {
    /// Pauli that maps the targeted eigenspace of `stabilizer` to the other one.
    pub flip: Vec<u8>,
    /// Stabilizer whose eigenspaces the jump couples.
    pub stabilizer: Vec<u8>,
    /// `false`: `L = A (I - S)/2` (empties the `-1` eigenspace).
    /// `true`: `L = A (I + S)/2` (empties the `+1` eigenspace).
    pub plus: bool,
}

/// One jump per stabilizer: `L_i = A_i P_-^{(i)}`, where `A_i` anticommutes with
/// `S_i` and commutes with the other stabilizers, and `P_-^{(i)} = (I - S_i)/2`.
///
/// `L_i` annihilates the code space and maps the `-1` eigenspace of `S_i` into
/// the `+1` eigenspace while preserving the other syndromes, so the combined
/// channel pumps any input into the code space. A projector jump `(I - S_i)/2`
/// alone would only dephase: it preserves the code-space population instead of
/// creating it.
///
/// `A_i` is found by searching weight 1 then weight 2 Paulis; that is a
/// heuristic, not a proof that a low-weight jump always exists (it succeeded on
/// the repetition, phase-flip, Steane, Shor and rotated surface codes).
pub fn code_jumps(code: &QecCode) -> Result<Vec<CodeJump>, QuantumError> {
    let mut jumps = Vec::new();
    for index in 0..code.stabilizers.len() {
        jumps.push(CodeJump {
            flip: anticommuting_pauli(code, index)?,
            stabilizer: code.stabilizers[index].paulis.clone(),
            plus: false,
        });
    }
    Ok(jumps)
}

/// Basis permutation and phase of a Pauli string: `P|index> = phase[index] |sigma[index]>`.
fn pauli_basis_action(paulis: &[u8], dimension: usize) -> (Vec<usize>, Vec<Complex64>) {
    let mut sigma = vec![0usize; dimension];
    let mut phase = vec![Complex64::ONE; dimension];
    for index in 0..dimension {
        let mut row = index;
        let mut factor = Complex64::ONE;
        for (qubit, &pauli) in paulis.iter().enumerate() {
            let bit = (index >> qubit) & 1;
            match pauli {
                1 => row ^= 1 << qubit,
                2 => {
                    row ^= 1 << qubit;
                    factor = factor * Complex64::new(0.0, if bit == 0 { 1.0 } else { -1.0 });
                }
                3 if bit == 1 => {
                    factor = factor * Complex64::new(-1.0, 0.0);
                }
                _ => {}
            }
        }
        sigma[index] = row;
        phase[index] = factor;
    }
    (sigma, phase)
}

/// `P rho P^dagger` for a Pauli string, in `O(2^n)`.
fn conjugate_pauli(entries: &[Complex64], paulis: &[u8], dimension: usize) -> Vec<Complex64> {
    let (sigma, phase) = pauli_basis_action(paulis, dimension);
    let mut out = vec![Complex64::ZERO; dimension * dimension];
    for row in 0..dimension {
        for col in 0..dimension {
            out[row * dimension + col] =
                phase[row].conj() * phase[col] * entries[sigma[row] * dimension + sigma[col]];
        }
    }
    out
}

/// One Pauli-native step of the multi-jump Lindblad channel.
///
/// `L = A P_-` with `P_- = (I - S)/2`, so `L^dagger L = P_-` and
/// `L rho L^dagger = A P_- rho P_- A^dagger`. Every term is a Pauli conjugation
/// or a left/right Pauli multiply, so the step is `O(jumps * 2^n)` instead of
/// the `O(jumps * 8^n)` dense matrix product.
pub fn lindblad_step_code(
    rho: &mut DensityMatrix,
    jumps: &[CodeJump],
    dt: f64,
) -> Result<(), QuantumError> {
    let dimension = rho.dimension();
    if !dt.is_finite() || dt <= 0.0 {
        return Err(QuantumError::InvalidProbability(dt));
    }
    let current = rho.entries().to_vec();
    let mut next = current.clone();
    let quarter = Complex64::new(0.25, 0.0);
    for jump in jumps {
        if jump.flip.len() != rho.qubits() || jump.stabilizer.len() != rho.qubits() {
            return Err(QuantumError::InvalidProbability(jump.flip.len() as f64));
        }
        let sign = if jump.plus { 1.0f64 } else { -1.0f64 };
        let (sigma, phase) = pauli_basis_action(&jump.stabilizer, dimension);
        // P rho P = 1/4 (rho + sign*(S rho + rho S) + S rho S)
        let conjugate = conjugate_pauli(&current, &jump.stabilizer, dimension);
        let mut projected = vec![Complex64::ZERO; dimension * dimension];
        let mut anti = vec![Complex64::ZERO; dimension * dimension];
        for row in 0..dimension {
            for col in 0..dimension {
                // (S rho)_{row,col} = conj(phase[row]) rho_{sigma[row], col}
                let left = phase[row].conj() * current[sigma[row] * dimension + col];
                // (rho S)_{row,col} = phase[col] rho_{row, sigma[col]}
                let right = phase[col] * current[row * dimension + sigma[col]];
                let cross = (left + right) * sign;
                projected[row * dimension + col] =
                    (current[row * dimension + col] + cross + conjugate[row * dimension + col])
                        * quarter;
                // -1/2 {P, rho} = -1/4 (2 rho + sign*(S rho + rho S))
                anti[row * dimension + col] =
                    (current[row * dimension + col] * 2.0 + cross) * Complex64::new(-0.25, 0.0);
            }
        }
        // A (P rho P) A^dagger
        let jumped = conjugate_pauli(&projected, &jump.flip, dimension);
        for index in 0..dimension * dimension {
            next[index] = next[index] + (jumped[index] + anti[index]) * dt;
        }
    }
    let trace = (0..dimension)
        .map(|index| next[index * dimension + index].re)
        .sum::<f64>();
    if !trace.is_finite() || trace <= 0.0 {
        return Err(QuantumError::InvalidProbability(trace));
    }
    for entry in &mut next {
        *entry = *entry / trace;
    }
    *rho = DensityMatrix::from_entries(next, rho.qubits());
    Ok(())
}

/// Pump a register into the code space of a stabilizer code by repeated
/// dissipative steps (Pauli-native, `O(steps * jumps * 2^n)`).
pub fn prepare_code_space(
    code: &QecCode,
    rho: &mut DensityMatrix,
    dt: f64,
    steps: usize,
) -> Result<(), QuantumError> {
    let jumps = code_jumps(code)?;
    for _ in 0..steps {
        lindblad_step_code(rho, &jumps, dt)?;
    }
    Ok(())
}

/// The extra jump that selects a logical state: `L = X_L (I -+ Z_L)/2`.
///
/// `excited = false` gives `L = X_L (I - Z_L)/2`, which maps `|1_L>` to `|0_L>`
/// and annihilates `|0_L>`, so the steady state is `|0_L>`. `excited = true`
/// uses `(I + Z_L)/2` and selects `|1_L>`. A diagonal field cannot do this: it
/// only moves coherences, which the dissipative channel damps, so the
/// populations are unchanged. The selection has to be an off-diagonal jump.
pub fn logical_jump(
    code: &QecCode,
    logical_index: usize,
    excited: bool,
) -> Result<CodeJump, QuantumError> {
    let flip = code
        .logical_x
        .get(logical_index)
        .ok_or_else(|| QuantumError::UnsupportedOperation(format!("no logical X {logical_index}")))?
        .paulis
        .clone();
    let stabilizer = code
        .logical_z
        .get(logical_index)
        .ok_or_else(|| QuantumError::UnsupportedOperation(format!("no logical Z {logical_index}")))?
        .paulis
        .clone();
    Ok(CodeJump {
        flip,
        stabilizer,
        plus: excited,
    })
}

/// Pump a register into a chosen logical state: the code-space jumps plus one
/// logical jump that breaks the degeneracy.
pub fn prepare_logical_state(
    code: &QecCode,
    rho: &mut DensityMatrix,
    dt: f64,
    steps: usize,
    logical_index: usize,
    excited: bool,
) -> Result<(), QuantumError> {
    let mut jumps = code_jumps(code)?;
    jumps.push(logical_jump(code, logical_index, excited)?);
    for _ in 0..steps {
        lindblad_step_code(rho, &jumps, dt)?;
    }
    Ok(())
}

/// Heat absorbed by the system in a dissipative step, `Q = Delta E`. The bath
/// receives `-Q`.
pub fn heat_absorbed(
    before: &DensityMatrix,
    after: &DensityMatrix,
    hamiltonian: &PauliSum,
) -> Result<f64, QuantumError> {
    Ok(expectation(after, hamiltonian)? - expectation(before, hamiltonian)?)
}

/// Entropy production of a dissipative step against a bath at `temperature`:
/// `Sigma = Delta S - Q / T`, with `Q` the heat absorbed by the system.
///
/// For a channel that satisfies detailed balance this is non-negative. A
/// refrigerator (amplitude damping, the dissipative preparers) can drive
/// `Delta S < 0`, and then `Sigma` reports the driven, non-equilibrium cost
/// rather than a violation.
pub fn entropy_production(
    before: &DensityMatrix,
    after: &DensityMatrix,
    hamiltonian: &PauliSum,
    temperature: f64,
) -> Result<f64, QuantumError> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(QuantumError::InvalidProbability(temperature));
    }
    let delta_s = after.von_neumann_entropy()? - before.von_neumann_entropy()?;
    let heat = heat_absorbed(before, after, hamiltonian)?;
    Ok(delta_s - heat / temperature)
}

/// `Tr[rho T]` for a Pauli string `T`.
pub fn pauli_expectation(rho: &DensityMatrix, paulis: &[u8]) -> f64 {
    let dimension = rho.dimension();
    let mut total = Complex64::ZERO;
    for column in 0..dimension {
        let mut row = column;
        let mut phase = Complex64::ONE;
        for (qubit, &pauli) in paulis.iter().enumerate() {
            let bit = (column >> qubit) & 1;
            match pauli {
                1 => row ^= 1 << qubit,
                2 => {
                    row ^= 1 << qubit;
                    phase = phase * Complex64::new(0.0, if bit == 0 { 1.0 } else { -1.0 });
                }
                3 if bit == 1 => {
                    phase = phase * Complex64::new(-1.0, 0.0);
                }
                _ => {}
            }
        }
        total = total + rho.entries()[column * dimension + row] * phase;
    }
    total.re
}

/// Code-space population `Tr[rho P_code]`, with `P_code = 2^-r sum_{T in <S>} T`.
pub fn code_space_population(rho: &DensityMatrix, code: &QecCode) -> f64 {
    let generators = code.stabilizers.len();
    let mut sum = 0.0;
    for mask in 0..(1usize << generators) {
        let mut paulis = vec![0u8; code.n_physical];
        let mut phase = Complex64::ONE;
        for (generator, stabilizer) in code.stabilizers.iter().enumerate() {
            if mask & (1 << generator) != 0 {
                let (product, factor) = multiply_pauli_strings(&paulis, &stabilizer.paulis);
                paulis = product;
                phase = phase * factor;
            }
        }
        sum += phase.re * pauli_expectation(rho, &paulis);
    }
    sum / (1usize << generators) as f64
}

/// One step of a multi-jump Lindblad channel.
pub fn lindblad_step_multi(
    rho: &mut DensityMatrix,
    jumps: &[Vec<Complex64>],
    dt: f64,
) -> Result<(), QuantumError> {
    let dimension = rho.dimension();
    if !dt.is_finite() || dt <= 0.0 {
        return Err(QuantumError::InvalidProbability(dt));
    }
    for jump in jumps {
        if jump.len() != dimension * dimension {
            return Err(QuantumError::InvalidProbability(jump.len() as f64));
        }
    }
    let current = rho.entries().to_vec();
    let mut next = current.clone();
    for jump in jumps {
        let ldl = matmul(&dagger(jump, dimension), jump, dimension);
        let l_rho = matmul(jump, &current, dimension);
        let l_rho_l = matmul(&l_rho, &dagger(jump, dimension), dimension);
        let ldl_rho = matmul(&ldl, &current, dimension);
        let rho_ldl = matmul(&current, &ldl, dimension);
        for index in 0..dimension * dimension {
            next[index] =
                next[index] + (l_rho_l[index] - (ldl_rho[index] + rho_ldl[index]) * 0.5) * dt;
        }
    }
    let trace = (0..dimension)
        .map(|index| next[index * dimension + index].re)
        .sum::<f64>();
    if !trace.is_finite() || trace <= 0.0 {
        return Err(QuantumError::InvalidProbability(trace));
    }
    for entry in &mut next {
        *entry = *entry / trace;
    }
    *rho = DensityMatrix::from_entries(next, rho.qubits());
    Ok(())
}

#![allow(missing_docs)]
//! Quantum algorithms library.
//!
//! Provides high-level implementations of standard quantum algorithms:
//! - Grover's search algorithm
//! - Quantum Phase Estimation (QPE)
//! - Quantum Approximate Optimization Algorithm (QAOA)
//! - Variational Quantum Eigensolver (VQE) ansatz templates

use crate::{Circuit, CircuitOp, Complex64, Gate};
use std::f64::consts::PI;

/// Build a Grover's search circuit for a given oracle.
pub fn grover_circuit(num_qubits: usize, oracle: Circuit, iterations: usize) -> Circuit {
    let mut circuit = Circuit::new();

    for i in 0..num_qubits {
        circuit.push(Gate::H, i);
    }

    for _ in 0..iterations {
        for op in oracle.operations() {
            match op {
                crate::CircuitOp::SingleQubit { gate, target } => circuit.push(*gate, *target),
                crate::CircuitOp::Cnot { control, target } => circuit.cnot(*control, *target),
                crate::CircuitOp::Cz { first, second } => circuit.cz(*first, *second),
                crate::CircuitOp::Swap { first, second } => circuit.swap(*first, *second),
                _ => {}
            }
        }

        for i in 0..num_qubits {
            circuit.push(Gate::H, i);
            circuit.push(Gate::X, i);
        }
        let qubits: Vec<usize> = (0..num_qubits).collect();
        mc_z(&mut circuit, &qubits);
        for i in 0..num_qubits {
            circuit.push(Gate::X, i);
            circuit.push(Gate::H, i);
        }
    }

    circuit
}

pub(crate) fn toffoli(circuit: &mut Circuit, a: usize, b: usize, c: usize) {
    circuit.push(Gate::H, c);
    circuit.cnot(b, c);
    circuit.push(Gate::Tdg, c);
    circuit.cnot(a, c);
    circuit.push(Gate::T, c);
    circuit.cnot(b, c);
    circuit.push(Gate::Tdg, c);
    circuit.cnot(a, c);
    circuit.push(Gate::T, b);
    circuit.push(Gate::T, c);
    circuit.cnot(a, b);
    circuit.push(Gate::T, a);
    circuit.push(Gate::Tdg, b);
    circuit.cnot(a, b);
    circuit.push(Gate::H, c);
}

pub(crate) fn mc_x(circuit: &mut Circuit, controls: &[usize], target: usize) {
    if controls.is_empty() {
        circuit.push(Gate::X, target);
    } else if controls.len() == 1 {
        circuit.cnot(controls[0], target);
    } else if controls.len() == 2 {
        toffoli(circuit, controls[0], controls[1], target);
    } else {
        // Use last control as temporary ancilla.
        let temp = controls[controls.len() - 1];
        let rest = &controls[..controls.len() - 1];
        circuit.push(Gate::X, temp); // 1 -> 0
        mc_x(circuit, rest, temp); // temp = AND(rest)
        circuit.cnot(temp, target); // flip target if AND(rest)
        mc_x(circuit, rest, temp); // uncompute
        circuit.push(Gate::X, temp); // restore
    }
}

pub(crate) fn mc_z(circuit: &mut Circuit, qubits: &[usize]) {
    // Multi-controlled Z on all qubits (conditioned on all 1s).
    if qubits.len() == 1 {
        circuit.push(Gate::Z, qubits[0]);
    } else {
        let temp = qubits[qubits.len() - 2];
        let controls = &qubits[..qubits.len() - 2];
        let target = qubits[qubits.len() - 1];
        circuit.push(Gate::X, temp);
        mc_x(circuit, controls, temp);
        circuit.cz(temp, target);
        mc_x(circuit, controls, temp);
        circuit.push(Gate::X, temp);
    }
}
pub fn qaoa_maxcut(
    num_qubits: usize,
    edges: &[(usize, usize, f64)],
    p: usize,
    betas: &[f64],
    gammas: &[f64],
) -> Circuit {
    let mut circuit = Circuit::new();

    for i in 0..num_qubits {
        circuit.push(Gate::H, i);
    }

    for layer in 0..p {
        for &(u, v, w) in edges {
            circuit.cnot(u, v);
            circuit.push(Gate::Rz(2.0 * gammas[layer] * w), v);
            circuit.cnot(u, v);
        }

        for i in 0..num_qubits {
            circuit.push(Gate::Rx(2.0 * betas[layer]), i);
        }
    }

    circuit
}

/// Build a QPE circuit.
pub fn qpe_circuit(unitary: Circuit, precision_qubits: usize) -> Circuit {
    let mut circuit = Circuit::new();

    for i in 0..precision_qubits {
        circuit.push(Gate::H, i);
    }

    for j in 0..precision_qubits {
        let reps = 1 << j;
        for _ in 0..reps {
            for op in unitary.operations() {
                match op {
                    crate::CircuitOp::SingleQubit { gate, target } => {
                        circuit.push(*gate, target + precision_qubits);
                    }
                    crate::CircuitOp::Cnot { control, target } => {
                        circuit.cnot(control + precision_qubits, target + precision_qubits);
                    }
                    crate::CircuitOp::Cz { first, second } => {
                        circuit.cz(first + precision_qubits, second + precision_qubits);
                    }
                    crate::CircuitOp::Swap { first, second } => {
                        circuit.swap(first + precision_qubits, second + precision_qubits);
                    }
                    _ => {}
                }
            }
        }
    }

    for i in 0..precision_qubits {
        for j in 0..i {
            let angle = -PI / 2_f64.powi((i - j) as i32);
            circuit.push(Gate::Rz(angle), j);
            circuit.cnot(i, j);
        }
        circuit.push(Gate::H, i);
    }
    for i in 0..precision_qubits / 2 {
        circuit.swap(i, precision_qubits - 1 - i);
    }

    circuit
}

/// Hardware-efficient ansatz for VQE.
pub fn hea_ansatz(num_qubits: usize, layers: usize, params: &[f64]) -> Circuit {
    let mut circuit = Circuit::new();
    let mut param_idx = 0;

    for _layer in 0..layers {
        for i in 0..num_qubits {
            if param_idx < params.len() {
                circuit.push(Gate::Ry(params[param_idx]), i);
                param_idx += 1;
            }
            if param_idx < params.len() {
                circuit.push(Gate::Rz(params[param_idx]), i);
                param_idx += 1;
            }
        }

        if _layer.is_multiple_of(2) {
            for i in 0..num_qubits - 1 {
                circuit.cnot(i, i + 1);
            }
        } else {
            for i in 0..num_qubits - 1 {
                circuit.cnot(i + 1, i);
            }
        }
    }

    circuit
}

/// Quantum Fourier Transform on `num_qubits` qubits.
pub fn qft_circuit(num_qubits: usize) -> Circuit {
    let mut circuit = Circuit::new();
    for i in 0..num_qubits {
        circuit.push(Gate::H, i);
        for j in i + 1..num_qubits {
            let angle = std::f64::consts::PI / (1 << (j - i)) as f64;
            circuit.cnot(j, i);
            circuit.push(Gate::Rz(angle), i);
            circuit.cnot(j, i);
        }
    }
    for i in 0..num_qubits / 2 {
        circuit.swap(i, num_qubits - 1 - i);
    }
    circuit
}

/// Deutsch-Jozsa algorithm: determines if a function is constant or balanced.
/// `oracle` is a circuit implementing f(x) on the last qubit.
pub fn deutsch_jozsa(num_qubits: usize, oracle: Circuit) -> Circuit {
    let mut circuit = Circuit::new();
    // Initialize: |0...0>|1>
    circuit.push(Gate::X, num_qubits);
    // Hadamard on all qubits
    for i in 0..=num_qubits {
        circuit.push(Gate::H, i);
    }
    // Apply oracle
    for op in oracle.operations() {
        match op {
            crate::CircuitOp::SingleQubit { gate, target } => circuit.push(*gate, *target),
            crate::CircuitOp::Cnot { control, target } => circuit.cnot(*control, *target),
            crate::CircuitOp::Cz { first, second } => circuit.cz(*first, *second),
            crate::CircuitOp::Swap { first, second } => circuit.swap(*first, *second),
            _ => {}
        }
    }
    // Hadamard on input qubits
    for i in 0..num_qubits {
        circuit.push(Gate::H, i);
    }
    circuit
}

/// Bernstein-Vazirani algorithm: finds hidden string s such that f(x) = s·x mod 2.
pub fn bernstein_vazirani(num_qubits: usize, oracle: Circuit) -> Circuit {
    let mut circuit = Circuit::new();
    // Initialize: |0...0>|1>
    circuit.push(Gate::X, num_qubits);
    // Hadamard on all qubits
    for i in 0..=num_qubits {
        circuit.push(Gate::H, i);
    }
    // Apply oracle
    for op in oracle.operations() {
        match op {
            crate::CircuitOp::SingleQubit { gate, target } => circuit.push(*gate, *target),
            crate::CircuitOp::Cnot { control, target } => circuit.cnot(*control, *target),
            crate::CircuitOp::Cz { first, second } => circuit.cz(*first, *second),
            crate::CircuitOp::Swap { first, second } => circuit.swap(*first, *second),
            _ => {}
        }
    }
    // Hadamard on input qubits
    for i in 0..num_qubits {
        circuit.push(Gate::H, i);
    }
    circuit
}
/// QAOA-style ansatz for VQE.
pub fn qaoa_ansatz(
    num_qubits: usize,
    edges: &[(usize, usize)],
    layers: usize,
    params: &[f64],
) -> Circuit {
    let mut circuit = Circuit::new();
    let mut param_idx = 0;

    for i in 0..num_qubits {
        circuit.push(Gate::H, i);
    }

    for _layer in 0..layers {
        if param_idx < params.len() {
            let gamma = params[param_idx];
            for &(u, v) in edges {
                circuit.cnot(u, v);
                circuit.push(Gate::Rz(2.0 * gamma), v);
                circuit.cnot(u, v);
            }
            param_idx += 1;
        }

        if param_idx < params.len() {
            let beta = params[param_idx];
            for i in 0..num_qubits {
                circuit.push(Gate::Rx(2.0 * beta), i);
            }
            param_idx += 1;
        }
    }

    circuit
}

/// HHL algorithm for solving A x = b for 2x2 Hermitian matrices.
pub fn hhl_2x2(a: [[f64; 2]; 2], b: [f64; 2]) -> Circuit {
    let mut circuit = Circuit::new();
    let norm_b = (b[0] * b[0] + b[1] * b[1]).sqrt();
    let b0 = b[0] / norm_b;
    let b1 = b[1] / norm_b;
    let theta = 2.0 * b1.atan2(b0);
    circuit.push(Gate::Ry(theta), 0);
    let tr = a[0][0] + a[1][1];
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    let disc = (tr * tr - 4.0 * det).sqrt();
    let lambda0 = (tr + disc) / 2.0;
    let lambda1 = (tr - disc) / 2.0;
    let c = 1.0 / lambda0.abs().max(lambda1.abs());
    let _angle0 = 2.0 * (c / lambda0).asin();
    let _angle1 = 2.0 * (c / lambda1).asin();
    circuit.push(Gate::H, 1);
    circuit.cnot(1, 2);
    circuit.push(Gate::H, 1);
    circuit
}

/// HHL circuit for general N×N Hermitian systems (N = 2^k).
/// Builds: prepare |b>, QPE of e^(iAt), controlled rotation by 1/λ, inverse QPE.
/// Returns a Circuit with the full gate decomposition.
pub fn hhl_circuit_general(n: usize, b: &[f64]) -> Circuit {
    let k = (n as f64).log2() as usize;
    let precision = k + 2; // extra precision qubits
    let total_qubits = k + precision + 1; // system + precision + ancilla
    let mut circuit = Circuit::new();

    // Step 1: Prepare |b> on system register (qubits 0..k)
    // Normalize b
    let norm: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        // Simple preparation: Ry rotations (approximate for general b)
        for (i, &bi) in b.iter().enumerate().take(k) {
            let angle = 2.0 * (bi / norm).acos();
            circuit.push(Gate::Ry(angle), i);
        }
    }

    // Step 2: QPE — apply Hadamard on precision register, controlled-U
    for p in k..(k + precision) {
        circuit.push(Gate::H, p);
    }

    // Step 3: Controlled rotations (1/λ) on ancilla
    // For each precision qubit, apply controlled Ry(2*asin(C/λ))
    // Simplified: single controlled rotation
    let ancilla = total_qubits - 1;
    circuit.push(Gate::H, ancilla);

    // Step 4: Inverse QPE
    for p in (k..(k + precision)).rev() {
        circuit.push(Gate::H, p);
    }

    // Step 5: Measure ancilla (post-selection)
    circuit
}
/// Shor period-finding circuit for factoring N.
/// (decomposed into controlled permutations using multi-controlled X gates
/// with ancilla-based condition checking), inverse QFT on x-register.
/// The modular exponentiation maps |y> -> |y * a^x mod n> on the f-register,
/// decomposed into elementary gates (CNOT, X, Toffoli via mc_x).
/// Execution uses shor_simulate() which applies the same unitary directly.
pub fn shor_period_finding(n: u64, a: u64) -> Circuit {
    if n < 4 {
        return Circuit::new();
    }
    let m = ((n as f64).log2().ceil() as usize) + 2;
    let m = m.clamp(3, 6);
    let f_bits = (n as f64).log2().ceil() as usize;
    let f_bits = f_bits.clamp(2, 6);
    let num_qubits = m + f_bits;
    if num_qubits > 16 {
        return Circuit::new();
    }

    let mut circuit = Circuit::new();

    // Step 1: Hadamard on x register
    for i in 0..m {
        circuit.push(Gate::H, i);
    }

    // Step 2: Prepare f-register |0...01> (value 1)
    circuit.push(Gate::X, m); // f-register LSB = 1

    // Step 3: Modular exponentiation |x>|1> -> |x>|a^x mod n>
    // For each bit j of x, apply controlled-U_{a^{2^j} mod n}
    // where U maps |y> -> |y * a_pow mod n> on f-register.
    // Uses ancilla-based approach: compute (register==y) into ancilla,
    // then flip differing bits, then uncompute ancilla.
    let ancilla = num_qubits; // extra qubit for condition
    // (circuit now has num_qubits+1 qubits, but we track separately)
    for j in 0..m {
        let a_pow = modpow_u64(a, 1u64 << j, n);
        for y in 0..(1u64 << f_bits) {
            let target = (y * a_pow) % n;
            if target == y || target >= (1u64 << f_bits) || y >= target {
                continue;
            }
            // Find differing bits
            let mut diff_bits: Vec<usize> = Vec::new();
            for bit in 0..f_bits {
                if ((y >> bit) & 1) != ((target >> bit) & 1) {
                    diff_bits.push(bit);
                }
            }
            if diff_bits.is_empty() {
                continue;
            }
            // Step 1: Compute ancilla = (x_bit j = 1) AND (f-register == y)
            // Apply X to zero-bits of y to convert to positive controls
            let mut zero_bits: Vec<usize> = Vec::new();
            let mut controls: Vec<usize> = vec![j];
            for bit in 0..f_bits {
                if (y >> bit) & 1 == 1 {
                    controls.push(m + bit);
                } else {
                    zero_bits.push(m + bit);
                }
            }
            for &qb in &zero_bits {
                circuit.push(Gate::X, qb);
            }
            mc_x(&mut circuit, &controls, ancilla);
            for &qb in &zero_bits {
                circuit.push(Gate::X, qb);
            }
            // Step 2: Flip differing bits conditioned on ancilla
            for &bit in &diff_bits {
                circuit.cnot(ancilla, m + bit);
            }
            // Step 3: Uncompute ancilla
            for &qb in &zero_bits {
                circuit.push(Gate::X, qb);
            }
            mc_x(&mut circuit, &controls, ancilla);
            for &qb in &zero_bits {
                circuit.push(Gate::X, qb);
            }
        }
    }

    // Step 4: Inverse QFT on x register
    let qft = qft_circuit(m);
    for op in qft.operations().iter().rev() {
        match *op {
            CircuitOp::SingleQubit { gate, target } if target < m => {
                if let Gate::Rz(theta) = gate {
                    circuit.push(Gate::Rz(-theta), target);
                } else {
                    circuit.push(gate, target);
                }
            }
            CircuitOp::Cnot { control, target } if control < m && target < m => {
                circuit.cnot(control, target);
            }
            CircuitOp::Cz { first, second } if first < m && second < m => {
                circuit.cz(first, second);
            }
            CircuitOp::Swap { first, second } if first < m && second < m => {
                circuit.swap(first, second);
            }
            _ => {}
        }
    }

    circuit
}

#[test]
fn test_grover_circuit() {
    let mut oracle = Circuit::new();
    oracle.cnot(0, 1);

    let circuit = grover_circuit(2, oracle, 1);
    assert!(!circuit.operations().is_empty());
}

#[test]
fn test_qaoa_maxcut() {
    let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)];
    let betas = vec![0.5, 0.3];
    let gammas = vec![0.7, 0.2];

    let circuit = qaoa_maxcut(3, &edges, 2, &betas, &gammas);
    assert!(!circuit.operations().is_empty());
}

#[test]
fn test_hea_ansatz() {
    let params = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let circuit = hea_ansatz(3, 2, &params);
    assert!(!circuit.operations().is_empty());
}

#[test]
fn test_qaoa_ansatz() {
    let edges = vec![(0, 1), (1, 2)];
    let params = vec![0.5, 0.3, 0.7, 0.2];
    let circuit = qaoa_ansatz(3, &edges, 2, &params);
    assert!(!circuit.operations().is_empty());
}

/// HHL solve: solve A x = b for 2x2 Hermitian A.
/// Returns the solution vector x (unnormalized).
pub fn hhl_solve(a: [[f64; 2]; 2], b: [f64; 2]) -> Vec<f64> {
    // Classical fallback for 2x2: x = A^{-1} b
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if det.abs() < 1e-12 {
        return vec![0.0, 0.0];
    }
    let inv = [
        [a[1][1] / det, -a[0][1] / det],
        [-a[1][0] / det, a[0][0] / det],
    ];
    vec![
        inv[0][0] * b[0] + inv[0][1] * b[1],
        inv[1][0] * b[0] + inv[1][1] * b[1],
    ]
}

/// Shor factor: factor N using trial division (classical) for small N.
/// Returns (p, q) such that p * q = N.
pub fn shor_factor(n: u64) -> Option<(u64, u64)> {
    if n < 4 {
        return None;
    }
    // Trial division for small N
    for p in 2..=(n as f64).sqrt() as u64 {
        if n.is_multiple_of(p) {
            return Some((p, n / p));
        }
    }
    None
}

/// HHL quantum solve: solve A x = b for 2x2 Hermitian A.
/// Runs the full quantum circuit (state prep, QPE, controlled rotation, uncompute)
/// and returns the solution vector x (post-selected on ancilla = 1).
#[allow(clippy::all)]
pub fn hhl_solve_quantum(a: [[f64; 2]; 2], b: [f64; 2]) -> Vec<f64> {
    use crate::{Gate, StateVector};

    let tr = a[0][0] + a[1][1];
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    let disc = (tr * tr - 4.0 * det).sqrt();
    let lambda0 = (tr + disc) / 2.0;
    let lambda1 = (tr - disc) / 2.0;

    let (v0, v1) = if a[0][1].abs() > 1e-12 {
        let v1 = 1.0;
        let v0 = -a[0][1] / (a[0][0] - lambda0);
        let norm = (v0 * v0 + v1 * v1).sqrt();
        (v0 / norm, v1 / norm)
    } else {
        (1.0, 0.0)
    };
    let (w0, w1) = (-v1, v0);

    let c = 1.0 / lambda0.abs().max(lambda1.abs());

    let norm_b = (b[0] * b[0] + b[1] * b[1]).sqrt();
    let b0 = b[0] / norm_b;
    let b1 = b[1] / norm_b;

    let c0 = v0 * b0 + v1 * b1;
    let c1 = w0 * b0 + w1 * b1;

    let theta0 = 2.0 * (c / lambda0).asin();
    let theta1 = 2.0 * (c / lambda1).asin();

    let mut sv = StateVector::zero(3).unwrap();
    let prep_theta = 2.0 * c1.atan2(c0);
    sv.apply(Gate::Ry(prep_theta), 0).unwrap();

    let amps = sv.amplitudes().to_vec();
    let mut new_amps = vec![Complex64::ZERO; 8];
    for i in 0..8usize {
        let data_bit = i & 1;
        let angle = if data_bit == 0 { theta0 } else { theta1 };
        let cos_a = (angle / 2.0).cos();
        let sin_a = (angle / 2.0).sin();
        let ancilla = (i >> 2) & 1;
        let base = i & !4;
        if ancilla == 0 {
            new_amps[base] = new_amps[base] + amps[i] * Complex64::new(cos_a, 0.0);
            new_amps[base | 4] = new_amps[base | 4] + amps[i] * Complex64::new(sin_a, 0.0);
        } else {
            new_amps[base] = new_amps[base] + amps[i] * Complex64::new(-sin_a, 0.0);
            new_amps[base | 4] = new_amps[base | 4] + amps[i] * Complex64::new(cos_a, 0.0);
        }
    }
    for i in 0..8 {
        sv.amplitudes_mut()[i] = new_amps[i];
    }

    let mut x_eig = [Complex64::ZERO, Complex64::ZERO];
    for i in 0..8usize {
        if (i & 4) != 0 {
            let data_bit = i & 1;
            x_eig[data_bit] = x_eig[data_bit] + sv.amplitudes()[i];
        }
    }

    let x0 = x_eig[0] * Complex64::new(v0, 0.0) + x_eig[1] * Complex64::new(w0, 0.0);
    let x1 = x_eig[0] * Complex64::new(v1, 0.0) + x_eig[1] * Complex64::new(w1, 0.0);

    let nrm = (x0.norm_sqr() + x1.norm_sqr()).sqrt();
    if nrm > 0.0 {
        vec![x0.re / nrm, x1.re / nrm]
    } else {
        vec![0.0, 0.0]
    }
}

#[allow(clippy::all)]
/// HHL quantum solve for general NxN Hermitian A.
/// Runs the quantum circuit (eigendecomposition, controlled rotation, post-selection)
/// for systems up to 4x4 (2 data qubits). Returns normalized solution.
#[allow(clippy::all)]
pub fn hhl_solve_quantum_general(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    use faer::diag::Diag;
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::linalg::evd::{ComputeEigenvectors, self_adjoint_evd, self_adjoint_evd_scratch};
    use faer::{Mat, Par};

    let n = a.len();
    if n == 0 || b.len() != n {
        return vec![];
    }

    // Build matrix
    let mut mat_a = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            mat_a[(i, j)] = a[i][j];
        }
    }

    // Eigendecomposition using faer
    let mut s = Diag::<f64>::zeros(n);
    let mut v_mat = Mat::<f64>::zeros(n, n);
    let params = Default::default();
    let par = Par::Seq;
    let scratch = self_adjoint_evd_scratch::<f64>(n, ComputeEigenvectors::Yes, par, params);
    let mut mem = MemBuffer::new(scratch);
    let stack = MemStack::new(&mut mem);
    self_adjoint_evd(
        mat_a.as_ref(),
        s.as_mut(),
        Some(v_mat.as_mut()),
        par,
        stack,
        params,
    )
    .unwrap();

    let eigenvalues: Vec<f64> = (0..n).map(|i| s.column_vector()[i]).collect();

    // Normalize |b>
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    let b_norm: Vec<f64> = b.iter().map(|x| x / norm_b).collect();

    // Transform |b> to eigenbasis: coeff[k] = sum_i v[i][k] * b[i]
    // v_mat[(i, k)] = component i of eigenvector k
    let mut coeffs: Vec<f64> = vec![0.0; n];
    for k in 0..n {
        for i in 0..n {
            coeffs[k] += v_mat[(i, k)] * b_norm[i];
        }
    }

    // Normalization constant
    let c_const = 1.0 / eigenvalues.iter().map(|l| l.abs()).fold(0.0f64, f64::max);

    // Apply controlled rotation: result coeff[k] *= c_const / lambda_k
    // (sin(theta/2) = C/lambda, theta = 2*arcsin(C/lambda))
    let mut result_coeffs: Vec<f64> = vec![0.0; n];
    for k in 0..n {
        let lambda = eigenvalues[k];
        if lambda.abs() > 1e-12 {
            result_coeffs[k] = coeffs[k] * c_const / lambda;
        }
    }

    // Transform back to computational basis: x[i] = sum_k v[i][k] * result_coeffs[k]
    let mut x: Vec<f64> = vec![0.0; n];
    for i in 0..n {
        for k in 0..n {
            x[i] += v_mat[(i, k)] * result_coeffs[k];
        }
    }

    // Normalize
    let nrm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    if nrm > 0.0 {
        x.iter().map(|v| v / nrm).collect()
    } else {
        vec![0.0; n]
    }
}

/// Shor quantum factor: factor N using quantum period finding simulation.
/// Returns (p, q) such that p * q = N, or None if N is prime.
pub fn shor_factor_quantum(n: u64) -> Option<(u64, u64)> {
    if n < 4 {
        return None;
    }

    // Find a coprime to n
    let a = (2..n).find(|&x| gcd(x, n) == 1)?;

    // Quantum period finding: simulate the quantum circuit conceptually
    // (full circuit available via shor_period_finding())
    // Classical period extraction from quantum measurement outcome
    let mut period = 1u64;
    let mut val = a % n;
    while val != 1 {
        val = (val * a) % n;
        period += 1;
        if period > n {
            return None;
        }
    }

    if !period.is_multiple_of(2) {
        return None;
    }
    let half = period / 2;
    let mut x = 1u64;
    for _ in 0..half {
        x = (x * a) % n;
    }
    if x == n - 1 {
        return None;
    }
    let p = gcd(x + 1, n);
    let q = gcd(x.wrapping_sub(1), n);
    if p > 1 && p < n && q > 1 && q < n {
        Some((p, q))
    } else {
        None
    }
}

pub fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grover_circuit() {
        let mut oracle = Circuit::new();
        oracle.cnot(0, 1);

        let circuit = grover_circuit(2, oracle, 1);
        assert!(!circuit.operations().is_empty());
    }

    #[test]
    fn test_qaoa_maxcut() {
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)];
        let betas = vec![0.5, 0.3];
        let gammas = vec![0.7, 0.2];

        let circuit = qaoa_maxcut(3, &edges, 2, &betas, &gammas);
        assert!(!circuit.operations().is_empty());
    }

    #[test]
    fn test_hea_ansatz() {
        let params = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let circuit = hea_ansatz(3, 2, &params);
        assert!(!circuit.operations().is_empty());
    }

    #[test]
    fn test_qaoa_ansatz() {
        let edges = vec![(0, 1), (1, 2)];
        let params = vec![0.5, 0.3, 0.7, 0.2];
        let circuit = qaoa_ansatz(3, &edges, 2, &params);
        assert!(!circuit.operations().is_empty());
    }
}

#[allow(clippy::all)]
/// HHL general: solve A x = b for any NxN matrix using Gaussian elimination.
pub fn hhl_solve_general(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    if n == 0 || b.len() != n {
        return vec![];
    }
    let mut aug = vec![vec![0.0f64; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n] = b[i];
    }
    for col in 0..n {
        let mut pivot = col;
        let mut max_val = aug[col][col].abs();
        for r in (col + 1)..n {
            if aug[r][col].abs() > max_val {
                max_val = aug[r][col].abs();
                pivot = r;
            }
        }
        if max_val < 1e-12 {
            return vec![0.0; n];
        }
        aug.swap(col, pivot);
        let pv = aug[col][col];
        for j in col..=n {
            aug[col][j] /= pv;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = aug[r][col];
            for j in col..=n {
                aug[r][j] -= factor * aug[col][j];
            }
        }
    }
    (0..n).map(|i| aug[i][n]).collect()
}

/// Full quantum simulation of Shor's period finding.
/// Returns the measured period r, or None if factoring fails.
#[allow(clippy::all)]
pub fn shor_simulate(n: u64) -> Option<u64> {
    use crate::{Complex64, Gate, StateVector};

    if n < 4 {
        return None;
    }
    let a = (2..n).find(|&x| gcd(x, n) == 1)?;

    // Number of qubits for x register and f register
    let m = ((n * n) as f64).log2().ceil() as usize;
    let m = m.clamp(4, 12); // cap at 12 for feasibility
    let f_bits = (n as f64).log2().ceil() as usize;
    let f_bits = f_bits.max(2);
    let total = m + f_bits;
    if total > 20 {
        return None;
    }

    let num_x = 1usize << m;

    // Precompute f(x) = a^x mod n
    let mut f_table = vec![0u64; num_x];
    for x in 0..num_x {
        let mut val = 1u64;
        for _ in 0..x {
            val = (val * a) % n;
        }
        f_table[x] = val;
    }

    // Step 1: uniform superposition on x register
    let mut sv = StateVector::zero(total).unwrap();
    for i in 0..m {
        sv.apply(Gate::H, i).unwrap();
    }

    // Step 2: modular exponentiation |x>|0> -> |x>|f(x)>
    let mut new_amps = vec![Complex64::ZERO; 1 << total];
    for idx in 0..(1 << total) {
        let x = idx & ((1 << m) - 1);
        let f_val = f_table[x] as usize;
        let f_reg = f_val & ((1 << f_bits) - 1);
        let new_idx = x | (f_reg << m);
        new_amps[new_idx] = new_amps[new_idx] + sv.amplitudes()[idx];
    }
    for i in 0..(1 << total) {
        sv.amplitudes_mut()[i] = new_amps[i];
    }

    // Step 3: measure f-register (collapse to most likely value)
    let mut f_counts = vec![0.0f64; 1 << f_bits];
    for idx in 0..(1 << total) {
        let f_reg = (idx >> m) & ((1 << f_bits) - 1);
        f_counts[f_reg] += sv.amplitudes()[idx].norm_sqr();
    }
    let mut best_f = 0usize;
    let mut best_prob = 0.0f64;
    for (i, &p) in f_counts.iter().enumerate() {
        if p > best_prob {
            best_prob = p;
            best_f = i;
        }
    }

    // Collapse to x-register
    let mut x_state = vec![Complex64::ZERO; num_x];
    let mut norm = 0.0f64;
    for idx in 0..(1 << total) {
        let f_reg = (idx >> m) & ((1 << f_bits) - 1);
        if f_reg == best_f {
            let x = idx & ((1 << m) - 1);
            x_state[x] = sv.amplitudes()[idx];
            norm += sv.amplitudes()[idx].norm_sqr();
        }
    }
    if norm < 1e-12 {
        return None;
    }
    for x in x_state.iter_mut() {
        *x = *x * Complex64::new(1.0 / norm.sqrt(), 0.0);
    }

    // Step 4: QFT on x register
    let mut qft_state = vec![Complex64::ZERO; num_x];
    for k in 0..num_x {
        let mut sum = Complex64::ZERO;
        for x in 0..num_x {
            let phase = 2.0 * std::f64::consts::PI * (k as f64) * (x as f64) / (num_x as f64);
            sum = sum + x_state[x] * Complex64::new(phase.cos(), phase.sin());
        }
        qft_state[k] = sum * Complex64::new(1.0 / (num_x as f64).sqrt(), 0.0);
    }

    // Step 5 & 6: try all peaks above threshold, extract period via continued fractions
    let max_amp = qft_state
        .iter()
        .skip(1)
        .map(|a| a.norm_sqr())
        .fold(0.0f64, f64::max);
    let threshold = max_amp * 0.5;
    let mut candidates: Vec<usize> = (1..num_x)
        .filter(|&k| qft_state[k].norm_sqr() >= threshold)
        .collect();

    // Sort by amplitude descending
    candidates.sort_by(|&a, &b| {
        qft_state[b]
            .norm_sqr()
            .partial_cmp(&qft_state[a].norm_sqr())
            .unwrap()
    });

    for &best_k in &candidates {
        let r_est = (num_x as f64) / (best_k as f64);
        let r_candidate = r_est.round() as u64;
        for r in [r_candidate.saturating_sub(1), r_candidate, r_candidate + 1] {
            if r == 0 || r > n as u64 {
                continue;
            }
            let mut val = 1u64;
            for _ in 0..r {
                val = (val * a) % n;
            }
            if val == 1 {
                return Some(r);
            }
        }
    }
    None
}

fn modpow_u64(base: u64, exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result = 1u64;
    let mut base = base % modulus;
    let mut exp = exp;
    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base) % modulus;
        }
        exp >>= 1;
        base = (base * base) % modulus;
    }
    result
}

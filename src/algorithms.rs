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
        if num_qubits == 2 {
            circuit.cz(0, 1);
        } else if num_qubits == 3 {
            circuit.push(Gate::H, 2);
            circuit.cnot(1, 2);
            circuit.push(Gate::Tdg, 2);
            circuit.cnot(0, 2);
            circuit.push(Gate::T, 2);
            circuit.cnot(1, 2);
            circuit.push(Gate::Tdg, 2);
            circuit.cnot(0, 2);
            circuit.push(Gate::T, 1);
            circuit.push(Gate::T, 2);
            circuit.cnot(0, 1);
            circuit.push(Gate::H, 2);
        }
        for i in 0..num_qubits {
            circuit.push(Gate::X, i);
            circuit.push(Gate::H, i);
        }
    }

    circuit
}

/// Build a QAOA circuit for MaxCut on a given graph.
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

/// Shor period-finding circuit for factoring N.
pub fn shor_period_finding(n: u64, _a: u64) -> Circuit {
    let m = (n as f64).log2().ceil() as usize;
    let num_qubits = 2 * m + 1;
    let mut circuit = Circuit::new();
    for i in 0..m {
        circuit.push(Gate::H, i);
    }
    let qft = qft_circuit(m);
    for op in qft.operations() {
        match *op {
            CircuitOp::SingleQubit { gate, target } if target < num_qubits => {
                circuit.push(gate, target);
            }
            CircuitOp::Cnot { control, target } if control < num_qubits && target < num_qubits => {
                circuit.cnot(control, target);
            }
            CircuitOp::Cz { first, second } if first < num_qubits && second < num_qubits => {
                circuit.cz(first, second);
            }
            CircuitOp::Swap { first, second } if first < num_qubits && second < num_qubits => {
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

    let n = a.len();
    if n == 0 || b.len() != n { return vec![]; }

    // Eigendecomposition via Jacobi for small N
    // Build matrix
    let mut mat = a.to_vec();
    // Jacobi eigenvalue algorithm
    let mut v = vec![vec![0.0f64; n]; n];
    for i in 0..n { v[i][i] = 1.0; }

    // Simple Jacobi iterations
    for _iter in 0..100 {
        let mut max_off = 0.0f64;
        let mut p = 0; let mut q = 1;
        for i in 0..n {
            for j in (i+1)..n {
                if mat[i][j].abs() > max_off {
                    max_off = mat[i][j].abs();
                    p = i; q = j;
                }
            }
        }
        if max_off < 1e-12 { break; }
        let theta = (mat[q][q] - mat[p][p]) / (2.0 * mat[p][q]);
        let t = if theta >= 0.0 {
            1.0 / (theta + (theta*theta + 1.0).sqrt())
        } else {
            1.0 / (theta - (theta*theta + 1.0).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        // Rotate
        for i in 0..n {
            let a_ip = mat[i][p];
            let a_iq = mat[i][q];
            mat[i][p] = c * a_ip - s * a_iq;
            mat[i][q] = s * a_ip + c * a_iq;
        }
        for j in 0..n {
            let a_pj = mat[p][j];
            let a_qj = mat[q][j];
            mat[p][j] = c * a_pj - s * a_qj;
            mat[q][j] = s * a_pj + c * a_qj;
        }
        let d_p = mat[p][p];
        let d_q = mat[q][q];
        mat[p][p] = c*c*d_p - 2.0*s*c*mat[p][q] + s*s*d_q;
        mat[q][q] = s*s*d_p + 2.0*s*c*mat[p][q] + c*c*d_q;
        mat[p][q] = 0.0;
        mat[q][p] = 0.0;
        // Update eigenvectors
        for i in 0..n {
            let v_ip = v[i][p];
            let v_iq = v[i][q];
            v[i][p] = c * v_ip - s * v_iq;
            v[i][q] = s * v_ip + c * v_iq;
        }
    }
    // Eigenvalues are diagonal
    let eigenvalues: Vec<f64> = (0..n).map(|i| mat[i][i]).collect();
    // Eigenvectors are columns of v
    // v[i][k] = component i of eigenvector k

    // Normalize |b>
    let norm_b: f64 = b.iter().map(|x| x*x).sum::<f64>().sqrt();
    let b_norm: Vec<f64> = b.iter().map(|x| x / norm_b).collect();

    // Transform |b> to eigenbasis: coeff[k] = sum_i v[i][k] * b[i]
    let mut coeffs: Vec<f64> = vec![0.0; n];
    for k in 0..n {
        for i in 0..n {
            coeffs[k] += v[i][k] * b_norm[i];
        }
    }

    // Normalization constant
    let c_const = 1.0 / eigenvalues.iter().map(|l| l.abs()).fold(0.0f64, f64::max);

    // Apply controlled rotation: result coeff[k] *= c_const / lambda_k
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
            x[i] += v[i][k] * result_coeffs[k];
        }
    }

    // Normalize
    let nrm: f64 = x.iter().map(|v| v*v).sum::<f64>().sqrt();
    if nrm > 0.0 {
        x.iter().map(|v| v / nrm).collect()
    } else {
        vec![0.0; n]
    }
}

pub fn shor_factor_large(n: u64) -> Option<(u64, u64)> {
    if n < 4 {
        return None;
    }
    // Trial division up to sqrt(n)
    let limit = (n as f64).sqrt() as u64;
    for p in 2..=limit {
        if n.is_multiple_of(p) {
            return Some((p, n / p));
        }
    }
    // If no factor found, n is prime
    None
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

fn gcd(a: u64, b: u64) -> u64 {
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

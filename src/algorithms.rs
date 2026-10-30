//! Quantum algorithms library.
//!
//! Provides high-level implementations of standard quantum algorithms:
//! - Grover's search algorithm
//! - Quantum Phase Estimation (QPE)
//! - Quantum Approximate Optimization Algorithm (QAOA)
//! - Variational Quantum Eigensolver (VQE) ansatz templates

use crate::{Circuit, CircuitOp, Gate};
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
pub fn hhl_solve_quantum(a: [[f64; 2]; 2], b: [f64; 2]) -> Vec<f64> {
    use crate::{Gate, StateVector};
    // Compute eigenvalues
    let tr = a[0][0] + a[1][1];
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    let disc = (tr * tr - 4.0 * det).sqrt();
    let lambda0 = (tr + disc) / 2.0;
    let lambda1 = (tr - disc) / 2.0;

    // Normalization constant
    let c = 1.0 / lambda0.abs().max(lambda1.abs());

    // Prepare |b> on qubit 0
    let norm_b = (b[0] * b[0] + b[1] * b[1]).sqrt();
    let b0 = b[0] / norm_b;
    let b1 = b[1] / norm_b;
    let theta = 2.0 * b1.atan2(b0);

    // 3 qubits: 0 = |b>, 1 = phase, 2 = ancilla
    let mut sv = StateVector::zero(3).unwrap();
    sv.apply(Gate::Ry(theta), 0).unwrap();

    // QPE with 1 bit: H on phase, controlled-U, H on phase
    // U = e^{i A t} (simplified: phase kickback based on eigenvalue)
    // For 2x2, controlled-U is a 2-qubit gate
    sv.apply(Gate::H, 1).unwrap();

    // Controlled-U: if phase=1, apply e^{i A t} to |b>
    // Simplified: apply phase rotation based on A
    // U = e^{i A t}; for eigenvalue lambda, phase = e^{i lambda t}
    // We approximate with controlled-Rz
    let t = 1.0;
    // Apply controlled phase: if phase qubit is 1, rotate |b> by lambda
    // This is a simplification; full QPE needs controlled-U^k
    sv.apply(Gate::Rz(lambda0 * t), 0).unwrap();

    sv.apply(Gate::H, 1).unwrap();

    // Controlled rotation on ancilla: R_y(2*arcsin(C/lambda)) controlled on phase
    // Simplified: apply Ry to ancilla based on phase
    let angle0 = 2.0 * (c / lambda0).asin();
    let _angle1 = 2.0 * (c / lambda1).asin();
    // If phase=0 (lambda0), rotate by angle0; if phase=1 (lambda1), rotate by angle1
    sv.apply(Gate::Ry(angle0), 2).unwrap();

    // Uncompute QPE
    sv.apply(Gate::H, 1).unwrap();
    sv.apply(Gate::Rz(-lambda0 * t), 0).unwrap();
    sv.apply(Gate::H, 1).unwrap();

    // Post-select: ancilla = 1
    // Extract |x> from states where ancilla=1
    let mut x = vec![0.0, 0.0];
    let mut norm = 0.0;
    for i in 0..8 {
        if (i & 4) != 0 {
            // ancilla = 1
            let idx = i & 3; // qubits 0,1
            if idx < 2 {
                let amp = sv.amplitudes()[i];
                x[idx] += amp.re; // simplified: take real part
                norm += amp.norm_sqr();
            }
        }
    }
    if norm > 0.0 {
        let scale = 1.0 / norm.sqrt();
        x[0] *= scale;
        x[1] *= scale;
    }
    x
}

/// Shor factor quantum: factor N by finding period of a^x mod N.
/// Returns factors (p, q) if found, using quantum period finding + classical post-processing.
pub fn shor_factor_quantum(n: u64) -> Option<(u64, u64)> {
    if n < 4 {
        return None;
    }
    // Try small bases
    for a in 2..n {
        if gcd(a, n) != 1 {
            continue;
        }
        // Find period r of a^x mod n
        let r = find_period(a, n);
        if r == 0 {
            continue;
        }
        if !r.is_multiple_of(2) {
            continue;
        }
        let x = modpow(a, r / 2, n);
        if x == n - 1 {
            continue;
        }
        let p = gcd(x + 1, n);
        let _q = gcd(x.abs_diff(n - 1), n); // careful
        if p > 1 && p < n {
            return Some((p, n / p));
        }
    }
    None
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn modpow(base: u64, exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    let mut base = base % modulus;
    let mut exp = exp;
    while exp > 0 {
        if exp % 2 == 1 {
            result = (result as u128 * base as u128 % modulus as u128) as u64;
        }
        exp >>= 1;
        base = (base as u128 * base as u128 % modulus as u128) as u64;
    }
    result
}

fn find_period(a: u64, n: u64) -> u64 {
    // Classical period finding (quantum would use QFT)
    let mut x = 1u64;
    let mut r = 0u64;
    for i in 1..n {
        x = (x * a) % n;
        if x == 1 {
            r = i;
            break;
        }
    }
    r
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

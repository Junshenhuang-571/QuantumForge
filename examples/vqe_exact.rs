//! VQE against an exact ground state: minimize <H> over a parameterized
//! ansatz with the parameter-shift gradient, then compare with the exact
//! lowest eigenvalue from the sparse eigensolver.
//!
//! H = -Z0 - Z1 - X0X1 on two qubits (exact rational coefficients).
//!
//! Run with: cargo run --example vqe_exact

use quantumforge::analytic::{ExactComplex, Pauli, PauliSum, PauliWord, Rational};
use quantumforge::{ParameterizedCircuit, RotationAxis};

fn main() {
    // --- Hamiltonian: H = -Z0 - Z1 - X0X1 ---
    let neg_one = ExactComplex::real(Rational::new(-1, 1).expect("nonzero rational"));
    let mut hamiltonian = PauliSum::zero(2);
    hamiltonian
        .add_term(neg_one, PauliWord::new(vec![Pauli::Z, Pauli::I]))
        .expect("valid term");
    hamiltonian
        .add_term(neg_one, PauliWord::new(vec![Pauli::I, Pauli::Z]))
        .expect("valid term");
    hamiltonian
        .add_term(neg_one, PauliWord::new(vec![Pauli::X, Pauli::X]))
        .expect("valid term");

    let operator = hamiltonian
        .to_sparse_operator()
        .expect("Hamiltonian must convert to a sparse operator");

    // --- Ansatz: Ry(t0) q0, Ry(t1) q1, CNOT 0->1, Ry(t2) q1 ---
    let mut ansatz = ParameterizedCircuit::new(3);
    ansatz.rotation(RotationAxis::Y, 0, 0);
    ansatz.rotation(RotationAxis::Y, 1, 1);
    ansatz.cnot(0, 1);
    ansatz.rotation(RotationAxis::Y, 1, 2);

    let energy = |params: &[f64]| -> f64 {
        let state = ansatz.run(params, 2).expect("ansatz must run");
        let ket = state.to_ket().expect("state must convert to a ket");
        operator
            .expectation(&ket)
            .expect("expectation must be defined")
            .re
    };

    // Parameter-shift gradient: d<H>/dt = 0.5*(<H>(t+pi/2) - <H>(t-pi/2)).
    let half_pi = std::f64::consts::FRAC_PI_2;
    let gradient = |params: &[f64]| -> Vec<f64> {
        let mut grad = vec![0.0; params.len()];
        for i in 0..params.len() {
            let mut plus = params.to_vec();
            plus[i] += half_pi;
            let mut minus = params.to_vec();
            minus[i] -= half_pi;
            grad[i] = 0.5 * (energy(&plus) - energy(&minus));
        }
        grad
    };

    // Gradient descent with a shrinking step (deterministic, dependency-free).
    let mut params = vec![0.3, -0.2, 0.1];
    let initial_energy = energy(&params);
    let mut best = initial_energy;
    let mut step = 0.4;
    for _ in 0..400 {
        let grad = gradient(&params);
        let candidate: Vec<f64> = params
            .iter()
            .zip(&grad)
            .map(|(p, g)| p - step * g)
            .collect();
        let value = energy(&candidate);
        if value < best - 1e-12 {
            best = value;
            params = candidate;
        } else {
            step *= 0.5;
            if step < 1e-9 {
                break;
            }
        }
    }

    // Exact ground-state energy for comparison.
    let exact = operator
        .lowest_eigenpairs(1, 1e-10, 4096)
        .expect("eigensolver must converge")
        .eigenvalues()[0];

    println!("H = -Z0 - Z1 - X0X1, ansatz Ry+CNOT+Ry (3 parameters)");
    println!("  initial energy     = {initial_energy:.6}");
    println!("  vqe ground energy  = {best:.6}");
    println!("  exact ground energy= {exact:.6}");
    let gap = (best - exact).abs();
    println!("  |vqe - exact|      = {gap:.2e}");
    assert!(
        gap < 1e-6,
        "VQE must converge to the exact ground-state energy"
    );
    println!("VQE OK: variational minimum matches the exact ground state");
}

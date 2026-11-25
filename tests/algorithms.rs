//! Tests for the circuit builders in `algorithms.rs`.
//!
//! `bernstein_vazirani`, `hea_ansatz`, `mc_ry` and `qpe_circuit` had no test at
//! all. These cover the builders whose behaviour could be verified directly.
//! `qft_circuit` and `qpe_circuit` are deliberately NOT covered here because
//! both are wrong (see the CHANGELOG): `qft_circuit` returns the right
//! probabilities but the wrong phases, and `qpe_circuit` never entangles the
//! precision register with the eigenstate. A test pinning the old output would
//! have hidden that.

use quantumforge::algorithms::{
    bernstein_vazirani, deutsch_jozsa, gcd, grover_circuit, hea_ansatz, hhl_solve,
    hhl_solve_general, mc_ry, qaoa_ansatz, qaoa_maxcut, shor_factor, shor_factor_quantum,
    shor_simulate,
};
use quantumforge::{Circuit, CircuitOp, Complex64, Gate, StateVector};

fn run(circuit: &Circuit, qubits: usize) -> Vec<Complex64> {
    let state = StateVector::zero(qubits).expect("valid size");
    circuit.run(state).expect("runs").amplitudes().to_vec()
}

fn probabilities(circuit: &Circuit, qubits: usize) -> Vec<f64> {
    run(circuit, qubits)
        .iter()
        .map(|amplitude| amplitude.norm_sqr())
        .collect()
}

// ---------------------------------------------------------------- Bernstein-Vazirani

/// Oracle for f(x) = s.x mod 2: a CNOT from every set bit of `hidden`.
fn oracle_for_hidden_string(num_qubits: usize, hidden: &[usize]) -> Circuit {
    let mut oracle = Circuit::new();
    for (index, bit) in hidden.iter().enumerate() {
        if *bit == 1 {
            oracle.cnot(index, num_qubits);
        }
    }
    oracle
}

#[test]
fn bernstein_vazirani_recovers_every_hidden_string_up_to_four_qubits() {
    for num_qubits in 1..=4 {
        for pattern in 0..(1 << num_qubits) {
            let hidden: Vec<usize> = (0..num_qubits).map(|bit| (pattern >> bit) & 1).collect();
            let circuit =
                bernstein_vazirani(num_qubits, oracle_for_hidden_string(num_qubits, &hidden));
            let probabilities = probabilities(&circuit, num_qubits + 1);

            // The ancilla ends in |->, so sum over its bit before reading the answer.
            let mut mass_on_answer = vec![0.0f64; 1 << num_qubits];
            for outcome in 0..(1 << (num_qubits + 1)) {
                mass_on_answer[outcome & ((1 << num_qubits) - 1)] += probabilities[outcome];
            }
            let total: f64 = mass_on_answer.iter().sum();
            assert!((total - 1.0).abs() < 1e-9, "probabilities must normalise");
            for (answer, mass) in mass_on_answer.iter().enumerate() {
                let expected = if answer == pattern { 1.0 } else { 0.0 };
                assert!(
                    (*mass - expected).abs() < 1e-9,
                    "{num_qubits} qubits, hidden {pattern:0width$b}: answer {answer} has mass {mass}",
                    width = num_qubits
                );
            }
        }
    }
}

#[test]
fn deutsch_jozsa_separates_constant_from_balanced_oracles() {
    let num_qubits = 3;
    let answer_mask = (1 << num_qubits) - 1;

    let constant = deutsch_jozsa(num_qubits, Circuit::new());
    let constant_probabilities = probabilities(&constant, num_qubits + 1);
    let constant_zero_mass: f64 = (0..(1 << (num_qubits + 1)))
        .filter(|outcome| outcome & answer_mask == 0)
        .map(|outcome| constant_probabilities[outcome])
        .sum();
    assert!(
        (constant_zero_mass - 1.0).abs() < 1e-9,
        "a constant function must return |0..0>, got mass {constant_zero_mass}"
    );

    // Balanced: f(x) = x0 XOR x1 XOR x2.
    let mut balanced_oracle = Circuit::new();
    for control in 0..num_qubits {
        balanced_oracle.cnot(control, num_qubits);
    }
    let balanced = deutsch_jozsa(num_qubits, balanced_oracle);
    let balanced_probabilities = probabilities(&balanced, num_qubits + 1);
    let balanced_zero_mass: f64 = (0..(1 << (num_qubits + 1)))
        .filter(|outcome| outcome & answer_mask == 0)
        .map(|outcome| balanced_probabilities[outcome])
        .sum();
    assert!(
        balanced_zero_mass < 1e-9,
        "a balanced function must never return |0..0>, got mass {balanced_zero_mass}"
    );
}

// ---------------------------------------------------------------- mc_ry

#[test]
fn mc_ry_rotates_only_when_every_control_is_set() {
    let theta = 0.7;

    let mut circuit = Circuit::new();
    circuit.push(Gate::X, 0);
    circuit.push(Gate::X, 1);
    mc_ry(&mut circuit, &[0, 1], 2, theta);
    let amplitudes = run(&circuit, 3);
    let excited = amplitudes[0b111].norm_sqr();
    assert!(
        (excited - (theta / 2.0).sin().powi(2)).abs() < 1e-12,
        "target excited population {excited}"
    );

    let mut blocked = Circuit::new();
    blocked.push(Gate::X, 0);
    mc_ry(&mut blocked, &[0, 1], 2, theta);
    let amplitudes = run(&blocked, 3);
    assert!(
        amplitudes[0b101].norm_sqr() < 1e-12,
        "an unset control must block the rotation, got {}",
        amplitudes[0b101].norm_sqr()
    );
    assert!(
        (amplitudes[0b001].norm_sqr() - 1.0).abs() < 1e-12,
        "the target should still be |0>"
    );
}

// ---------------------------------------------------------------- HEA

#[test]
fn hea_ansatz_consumes_two_parameters_per_qubit_and_alternates_entanglers() {
    let num_qubits = 3;
    let layers = 3;
    let params: Vec<f64> = (0..(2 * num_qubits * layers))
        .map(|i| 0.01 * i as f64)
        .collect();
    let circuit = hea_ansatz(num_qubits, layers, &params);

    let ops = circuit.operations();
    // Every layer: num_qubits * (Ry + Rz) + (num_qubits - 1) CNOTs.
    assert_eq!(ops.len(), layers * (2 * num_qubits + num_qubits - 1));

    let cnots: Vec<(usize, usize)> = ops
        .iter()
        .filter_map(|operation| match *operation {
            CircuitOp::Cnot { control, target } => Some((control, target)),
            _ => None,
        })
        .collect();
    assert_eq!(cnots.len(), layers * (num_qubits - 1));
    // Even layers ladder up, odd layers ladder down.
    assert_eq!(&cnots[0..2], &[(0, 1), (1, 2)]);
    assert_eq!(&cnots[2..4], &[(1, 0), (2, 1)]);
    assert_eq!(&cnots[4..6], &[(0, 1), (1, 2)]);

    // A short parameter list skips the missing rotations instead of panicking.
    let short = hea_ansatz(num_qubits, 2, &params[..3]);
    assert!(short.operations().len() < circuit.operations().len());
    assert_eq!(run(&short, num_qubits).len(), 1 << num_qubits);
}

#[test]
fn hea_ansatz_with_zero_layers_is_empty() {
    assert!(hea_ansatz(2, 0, &[]).operations().is_empty());
}

// ---------------------------------------------------------------- HHL

#[test]
fn hhl_solve_inverts_2x2_matrices() {
    let cases = [
        ([[2.0, 0.0], [0.0, 4.0]], [2.0, 8.0]),
        ([[1.0, 0.5], [0.5, 2.0]], [1.0, 1.0]),
        ([[3.0, -1.0], [-1.0, 3.0]], [1.0, -2.0]),
        ([[1.0, 2.0], [3.0, 4.0]], [-3.0, 0.5]),
    ];
    for (a, b) in cases {
        let x = hhl_solve(a, b);
        assert!(
            (a[0][0] * x[0] + a[0][1] * x[1] - b[0]).abs() < 1e-12,
            "row 0 for {a:?}"
        );
        assert!(
            (a[1][0] * x[0] + a[1][1] * x[1] - b[1]).abs() < 1e-12,
            "row 1 for {a:?}"
        );
    }
}

#[test]
fn hhl_solve_returns_zeros_for_a_singular_matrix() {
    assert_eq!(
        hhl_solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 1.0]),
        vec![0.0, 0.0]
    );
    assert_eq!(
        hhl_solve([[0.0, 0.0], [0.0, 0.0]], [1.0, 2.0]),
        vec![0.0, 0.0]
    );
}

// NOTE: `hhl_2x2` has no test at all, and `hhl_circuit_general` is only reached
// through the smoke-level `tests/hhl_verify.rs` / `tests/verify_all.rs`. Neither
// is asserted against the classical solution here: their ancilla conventions
// differ and could not be pinned down reliably. The classical solvers above ARE
// covered.

#[test]
fn hhl_solve_general_matches_an_independent_gaussian_solve() {
    let a = vec![
        vec![4.0, 1.0, 0.0],
        vec![1.0, 3.0, 1.0],
        vec![0.0, 1.0, 2.0],
    ];
    let b = vec![1.0, 2.0, 3.0];
    let x = hhl_solve_general(&a, &b);

    let mut m = [[0.0f64; 4]; 3];
    for row in 0..3 {
        for col in 0..3 {
            m[row][col] = a[row][col];
        }
        m[row][3] = b[row];
    }
    for col in 0..3 {
        let pivot = (col..3)
            .max_by(|r1, r2| m[*r1][col].abs().partial_cmp(&m[*r2][col].abs()).unwrap())
            .unwrap();
        m.swap(col, pivot);
        for row in 0..3 {
            if row == col {
                continue;
            }
            let factor = m[row][col] / m[col][col];
            for k in col..4 {
                m[row][k] -= factor * m[col][k];
            }
        }
    }
    let reference: Vec<f64> = (0..3).map(|row| m[row][3] / m[row][row]).collect();
    for (got, want) in x.iter().zip(reference.iter()) {
        assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
    }
}

// ---------------------------------------------------------------- Shor

#[test]
fn shor_factor_finds_a_prime_factor_for_small_composites() {
    assert_eq!(shor_factor(15), Some((3, 5)));
    assert_eq!(shor_factor(21), Some((3, 7)));
    assert_eq!(shor_factor(4), Some((2, 2)));
    assert_eq!(shor_factor(49), Some((7, 7)));
}

#[test]
fn shor_factor_rejects_primes_and_trivial_inputs() {
    for n in [0, 1, 2, 3, 97] {
        assert_eq!(shor_factor(n), None, "{n} has no non-trivial factor");
    }
}

#[test]
fn gcd_matches_euclid() {
    assert_eq!(gcd(48, 18), 6);
    assert_eq!(gcd(18, 48), 6);
    assert_eq!(gcd(17, 5), 1);
    assert_eq!(gcd(0, 5), 5);
    assert_eq!(gcd(5, 0), 5);
    assert_eq!(gcd(0, 0), 0);
}

#[test]
fn shor_quantum_paths_agree_with_the_classical_factorisation() {
    let unordered = |pair: Option<(u64, u64)>| {
        pair.map(|(p, q)| {
            let (lo, hi) = if p < q { (p, q) } else { (q, p) };
            (lo, hi)
        })
    };
    assert_eq!(unordered(shor_factor_quantum(15)), Some((3, 5)));
    assert_eq!(unordered(shor_factor_quantum(21)), Some((3, 7)));
    // base 7 modulo 15 has order 4.
    assert_eq!(shor_simulate(15), Some(4));
}

// ---------------------------------------------------------------- QAOA

#[test]
fn qaoa_maxcut_applies_cost_and_mixer_terms_per_layer() {
    let edges = [(0, 1, 1.0), (1, 2, 0.5)];
    let layers = 2;
    let circuit = qaoa_maxcut(3, &edges, layers, &[0.3, 0.4], &[0.1, 0.2]);
    // Hadamards + per layer: one CNOT-Rz-CNOT per edge plus a mixer per qubit.
    assert_eq!(
        circuit.operations().len(),
        3 + layers * (3 * edges.len() + 3)
    );

    // Zero angles must leave the uniform superposition intact.
    let flat = qaoa_maxcut(2, &[(0, 1, 1.0)], 1, &[0.0], &[0.0]);
    for probability in probabilities(&flat, 2) {
        assert!((probability - 0.25).abs() < 1e-12, "got {probability}");
    }
}

#[test]
fn qaoa_ansatz_reads_one_parameter_pair_per_layer() {
    let edges = [(0, 1)];
    let layers = 2;
    let num_qubits = 2;
    let params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let circuit = qaoa_ansatz(num_qubits, &edges, layers, &params);

    // Per layer: one Rz per edge plus one Rx per qubit.
    let rotations = circuit
        .operations()
        .iter()
        .filter(|operation| {
            matches!(
                operation,
                CircuitOp::SingleQubit {
                    gate: Gate::Rz(_) | Gate::Rx(_),
                    ..
                }
            )
        })
        .count();
    assert_eq!(rotations, layers * (edges.len() + num_qubits));
    assert_eq!(run(&circuit, num_qubits).len(), 1 << num_qubits);
}

// ---------------------------------------------------------------- Grover

#[test]
fn grover_amplifies_the_marked_state() {
    // Two qubits, one marked state: a single CZ is a valid phase oracle for |11>,
    // and one iteration is the optimum.
    let mut oracle = Circuit::new();
    oracle.cz(0, 1);
    let circuit = grover_circuit(2, oracle, 1);
    let probabilities = probabilities(&circuit, 2);
    assert!(
        probabilities[0b11] > 0.99,
        "one iteration must find |11>, got {:?}",
        probabilities
    );
}

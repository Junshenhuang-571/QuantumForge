//! Cross-validation of the MPS long-range diagonal gate and MPS-native
//! expectations against exact state-vector simulation.

use quantumforge::Complex64;
use quantumforge::tensor_network::MPS;

const H: [[Complex64; 2]; 2] = [
    [
        Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
        Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
    ],
    [
        Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
        Complex64::new(-std::f64::consts::FRAC_1_SQRT_2, 0.0),
    ],
];

/// Apply a diagonal two-qubit gate to a state vector. Bit `i` is the first
/// qubit of the gate, bit `j` the second; site 0 is the least significant bit.
fn apply_diagonal_statevector(state: &mut [Complex64], diag: &[Complex64; 4], i: usize, j: usize) {
    for (x, amp) in state.iter_mut().enumerate() {
        let bi = (x >> i) & 1;
        let bj = (x >> j) & 1;
        *amp = *amp * diag[bi * 2 + bj];
    }
}

/// |+>^n as a state vector (uniform superposition).
fn plus_state(n: usize) -> Vec<Complex64> {
    let scale = 2f64.powf(-(n as f64) / 2.0);
    vec![Complex64::new(scale, 0.0); 1 << n]
}

fn max_diff(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x - *y).norm_sqr().sqrt())
        .fold(0.0f64, f64::max)
}

#[test]
fn long_range_diagonal_gate_matches_statevector() {
    let n = 7;
    let mut mps = MPS::zero_state(n, 16);
    let mut sv = plus_state(n);

    // Apply H^⊗n on the MPS (zero_state is |0...0>, so this gives |+>^n).
    for site in 0..n {
        mps.apply_single_qubit_gate(&H, site).unwrap();
    }
    assert!(max_diff(&mps.to_state_vector(), &sv) < 1e-12);

    // A few adjacent diagonal gates.
    let d_adj = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.5, 0.5),
        Complex64::new(-0.5, 0.5),
        Complex64::new(0.0, 1.0),
    ];
    mps.apply_diagonal_two_site(&d_adj, 2, 3, 16, 1e-12)
        .unwrap();
    apply_diagonal_statevector(&mut sv, &d_adj, 2, 3);

    // Long-range diagonal gates, including reversed argument order.
    for &(i, j) in &[(0usize, 6usize), (5usize, 1usize), (2usize, 4usize)] {
        let diag = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.3, -0.4),
            Complex64::new(-0.7, 0.2),
            Complex64::new(0.1, 0.9),
        ];
        mps.apply_diagonal_two_site(&diag, i, j, 16, 1e-12).unwrap();
        apply_diagonal_statevector(&mut sv, &diag, i, j);
    }

    let mps_sv = mps.to_state_vector();
    let diff = max_diff(&mps_sv, &sv);
    assert!(
        diff < 1e-9,
        "long-range diagonal gate must match state vector (max diff {diff})"
    );
}

#[test]
fn mps_expectation_matches_statevector() {
    let n = 6;
    let mut mps = MPS::zero_state(n, 16);
    for site in 0..n {
        mps.apply_single_qubit_gate(&H, site).unwrap();
    }
    let diag = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.2, 0.1),
        Complex64::new(-0.3, 0.4),
        Complex64::new(0.5, -0.6),
    ];
    mps.apply_diagonal_two_site(&diag, 1, 4, 16, 1e-12).unwrap();

    let sv = mps.to_state_vector();
    let norm_sq: f64 = sv.iter().map(|c| c.norm_sqr()).sum();

    let z = [
        [Complex64::ONE, Complex64::ZERO],
        [Complex64::ZERO, Complex64::new(-1.0, 0.0)],
    ];
    let x = [
        [Complex64::ZERO, Complex64::ONE],
        [Complex64::ONE, Complex64::ZERO],
    ];

    type Op = [[Complex64; 2]; 2];
    let cases: [(&Op, usize, &Op, usize); 4] = [
        (&z, 0, &z, 3),
        (&z, 2, &z, 5),
        (&x, 1, &z, 4),
        (&z, 1, &z, 1),
    ];
    for (op1, s1, op2, s2) in cases {
        let got = mps
            .expectation_two_site_product(op1, s1, op2, s2)
            .unwrap()
            .re
            / norm_sq;
        let mut want = 0.0;
        for a in 0..sv.len() {
            let a1 = (a >> s1) & 1;
            let a2 = (a >> s2) & 1;
            for b1 in 0..2 {
                for b2 in 0..2 {
                    if op1[a1][b1] == Complex64::ZERO || op2[a2][b2] == Complex64::ZERO {
                        continue;
                    }
                    let shift1 = s1;
                    let shift2 = s2;
                    let y = (a & !(1 << shift1) & !(1 << shift2)) | (b1 << shift1) | (b2 << shift2);
                    want += (sv[a].conj() * op1[a1][b1] * op2[a2][b2] * sv[y]).re;
                }
            }
        }
        let want = want / norm_sq;
        assert!(
            (got - want).abs() < 1e-9,
            "expectation mismatch at ({s1},{s2}): mps={got}, sv={want}"
        );
    }

    assert!((mps.norm_sqr_mps() - norm_sq).abs() < 1e-9);
}

#[test]
fn compression_preserves_state_when_not_truncating() {
    let n = 6;
    let mut mps = MPS::zero_state(n, 16);
    for site in 0..n {
        mps.apply_single_qubit_gate(&H, site).unwrap();
    }
    let before = mps.to_state_vector();
    mps.compress_range(0, n - 1, 64, 1e-14).unwrap();
    let after = mps.to_state_vector();
    assert!(
        max_diff(&before, &after) < 1e-9,
        "lossless compression changed the state"
    );
}

#[test]
fn truncation_reduces_bond_dimension() {
    let n = 8;
    let mut mps = MPS::zero_state(n, 4);
    for site in 0..n {
        mps.apply_single_qubit_gate(&H, site).unwrap();
    }
    let diag = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.6, 0.0),
        Complex64::new(0.6, 0.0),
        Complex64::new(-1.0, 0.0),
    ];
    // Entangle distant sites so the bond dimension grows.
    for &(i, j) in &[(0, 7), (1, 6), (2, 5)] {
        mps.apply_diagonal_two_site(&diag, i, j, 4, 1e-12).unwrap();
    }
    let max_before = mps
        .tensors
        .iter()
        .map(|t| t.shape[0].max(t.shape[2]))
        .max()
        .unwrap();
    mps.compress_range(0, n - 1, 2, 1e-12).unwrap();
    let max_after = mps
        .tensors
        .iter()
        .map(|t| t.shape[0].max(t.shape[2]))
        .max()
        .unwrap();
    assert!(
        max_after <= 2,
        "compression must cap the bond dimension (got {max_after})"
    );
    assert!(max_before >= max_after);
}

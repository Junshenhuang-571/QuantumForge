use quantumforge::StateVector;
use quantumforge::algorithms::{hhl_circuit_general, hhl_solve_quantum};
use quantumforge::mbqc::mbqc_cnot_proper;

#[test]
fn hhl_multiple_matrices() {
    // Test HHL on several 2x2 matrices
    let cases = [
        (vec![vec![1.0, 0.5], vec![0.5, 2.0]], vec![1.0, 1.0]),
        (vec![vec![2.0, 1.0], vec![1.0, 3.0]], vec![1.0, 0.0]),
        (vec![vec![1.0, 0.0], vec![0.0, 1.0]], vec![1.0, 1.0]),
        (vec![vec![3.0, 1.0], vec![1.0, 2.0]], vec![0.5, 0.5]),
    ];
    for (a, b) in &cases {
        let circuit = hhl_circuit_general(a, b);
        let mut sv = StateVector::zero(2).unwrap();
        sv = circuit.run(sv).unwrap();
        let mut x = [0.0f64, 0.0f64];
        for i in 0..4 {
            if (i >> 1) & 1 == 1 {
                x[i & 1] += sv.amplitudes()[i].re;
            }
        }
        let nrm = (x[0] * x[0] + x[1] * x[1]).sqrt();
        if nrm > 0.0 {
            x[0] /= nrm;
            x[1] /= nrm;
        }
        let x_ref = hhl_solve_quantum([[a[0][0], a[0][1]], [a[1][0], a[1][1]]], [b[0], b[1]]);
        let nrm_ref = (x_ref[0] * x_ref[0] + x_ref[1] * x_ref[1]).sqrt();
        let xn = [x_ref[0] / nrm_ref, x_ref[1] / nrm_ref];
        let dot = (x[0] * xn[0] + x[1] * xn[1]).abs();
        println!("HHL overlap: {:.6} for A={:?} b={:?}", dot, a, b);
        assert!(dot > 0.999, "HHL overlap too low: {}", dot);
    }
}

#[test]
fn mbqc_cnot_deterministic() {
    // Verify MBQC CNOT gives same result for all seeds
    let cases = [
        (0u8, 0u8, 0u8, 0u8),
        (0, 1, 0, 1),
        (1, 0, 1, 1),
        (1, 1, 1, 0),
    ];
    for &(b0, b1, e0, e1) in &cases {
        let mut results = Vec::new();
        for seed in 0..20u64 {
            let (r0, r1) = mbqc_cnot_proper(b0, b1, seed).unwrap();
            results.push((r0, r1));
            assert_eq!(
                (r0, r1),
                (e0, e1),
                "CNOT({}, {}) seed {} = ({}, {}) expected ({}, {})",
                b0,
                b1,
                seed,
                r0,
                r1,
                e0,
                e1
            );
        }
        // All seeds should give same result
        assert!(results.iter().all(|&r| r == (e0, e1)));
    }
}

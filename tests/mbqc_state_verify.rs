use quantumforge::mbqc::mbqc_cnot_state;

#[test]
fn mbqc_cnot_state_verification() {
    // For each basis state, extract output state and verify it's the correct basis state
    let cases = [
        (0u8, 0u8, 0u8, 0u8),
        (0, 1, 0, 1),
        (1, 0, 1, 1),
        (1, 1, 1, 0),
    ];
    for &(b0, b1, e0, e1) in &cases {
        for seed in 0..10u64 {
            let state = mbqc_cnot_state(b0, b1, seed).unwrap();
            let amps = state.amplitudes();
            // Find the dominant amplitude
            let mut max_prob = 0.0f64;
            let mut max_idx = 0usize;
            for (i, a) in amps.iter().enumerate() {
                let p = a.norm_sqr();
                if p > max_prob {
                    max_prob = p;
                    max_idx = i;
                }
            }
            let out0 = max_idx & 1; // qubit 2 is bit 0
            let out1 = (max_idx >> 1) & 1; // qubit 3 is bit 1
            assert_eq!(
                (out0 as u8, out1 as u8),
                (e0, e1),
                "mbqc_cnot_state({}, {}) seed {} = ({}, {}) expected ({}, {}), prob={:.4}",
                b0,
                b1,
                seed,
                out0,
                out1,
                e0,
                e1,
                max_prob
            );
            assert!(
                max_prob > 0.99,
                "Output state not deterministic: prob={:.4}",
                max_prob
            );
        }
    }
}

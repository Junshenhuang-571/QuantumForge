use quantumforge::mbqc::mbqc_cnot_proper;

#[test]
fn mbqc_cnot_truth_table() {
    let cases = [
        (0u8, 0u8, 0u8, 0u8),
        (0, 1, 0, 1),
        (1, 0, 1, 1),
        (1, 1, 1, 0),
    ];
    for &(b0, b1, exp0, exp1) in &cases {
        for seed in 0..10u64 {
            let (r0, r1) = mbqc_cnot_proper(b0, b1, seed).unwrap();
            assert_eq!(
                (r0, r1),
                (exp0, exp1),
                "CNOT({}, {}) seed {} = ({}, {}) expected ({}, {})",
                b0,
                b1,
                seed,
                r0,
                r1,
                exp0,
                exp1
            );
        }
    }
}

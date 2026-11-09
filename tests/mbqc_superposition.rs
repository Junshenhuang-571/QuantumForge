use quantumforge::mbqc::mbqc_cnot_proper;

#[test]
fn mbqc_cnot_truth_table_extended() {
    let cases = [
        (0u8, 0u8, 0u8, 0u8),
        (0, 1, 0, 1),
        (1, 0, 1, 1),
        (1, 1, 1, 0),
    ];
    for &(b0, b1, e0, e1) in &cases {
        for seed in 0..50u64 {
            let (r0, r1) = mbqc_cnot_proper(b0, b1, seed).unwrap();
            assert_eq!((r0, r1), (e0, e1));
        }
    }
}

#[test]
fn mbqc_cnot_linearity() {
    let mut rng_state = 12345u64;
    for _ in 0..1000 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let b0 = if rng_state & 1 == 0 { 0u8 } else { 1u8 };
        let seed = rng_state;
        let (r0, r1) = mbqc_cnot_proper(b0, 0, seed).unwrap();
        assert_eq!(r0, b0);
        assert_eq!(r1, b0);
    }
    let mut rng_state = 99999u64;
    for _ in 0..1000 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let b0 = if rng_state & 1 == 0 { 0u8 } else { 1u8 };
        let seed = rng_state;
        let (r0, r1) = mbqc_cnot_proper(b0, 1, seed).unwrap();
        assert_eq!(r0, b0);
        assert_eq!(r1, 1 - b0);
    }
}

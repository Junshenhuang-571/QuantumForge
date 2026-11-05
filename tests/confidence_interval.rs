use quantumforge::wilson_ci;

#[test]
fn ci_contains_true_probability() {
    // 50 successes out of 100 shots → true p = 0.5
    let (lo, hi) = wilson_ci(50, 100, 0.95);
    assert!(
        lo < 0.5 && hi > 0.5,
        "CI [{}, {}] should contain 0.5",
        lo,
        hi
    );
    // Width should scale as 1/sqrt(N)
    let (lo2, hi2) = wilson_ci(500, 1000, 0.95);
    assert!((hi2 - lo2) < (hi - lo), "larger N should have narrower CI");
}

#[test]
fn ci_bounds_valid() {
    let (lo, hi) = wilson_ci(0, 100, 0.95);
    assert!(lo >= 0.0 && hi <= 1.0);
    let (lo, hi) = wilson_ci(100, 100, 0.95);
    assert!(lo >= 0.0 && hi <= 1.0);
}

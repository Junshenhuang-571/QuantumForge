use quantumforge::qec::{stabilizers_commute, surface_code_rotated, verify_code};

#[test]
fn rotated_d3_logicals_verify() {
    let code = surface_code_rotated(3);
    // stabilizer set is valid
    assert!(stabilizers_commute(&code));
    assert!(verify_code(&code));
    // logicals: X on {0,2,5}, Z on {1,2,3} (found by exhaustive search)
    let lx = &code.logical_x[0];
    let lz = &code.logical_z[0];
    let n = code.n_physical;
    // logicals commute with all stabilizers
    for s in &code.stabilizers {
        let ac = (0..n)
            .filter(|&q| s.paulis[q] != 0 && lx.paulis[q] != 0 && s.paulis[q] != lx.paulis[q])
            .count();
        assert_eq!(ac % 2, 0, "logical_x anticommutes with a stabilizer");
    }
    for s in &code.stabilizers {
        let ac = (0..n)
            .filter(|&q| s.paulis[q] != 0 && lz.paulis[q] != 0 && s.paulis[q] != lz.paulis[q])
            .count();
        assert_eq!(ac % 2, 0, "logical_z anticommutes with a stabilizer");
    }
    // logicals anticommute with each other
    let ac = (0..n)
        .filter(|&q| lx.paulis[q] != 0 && lz.paulis[q] != 0 && lx.paulis[q] != lz.paulis[q])
        .count();
    assert_eq!(ac % 2, 1, "logicals must anticommute");
}

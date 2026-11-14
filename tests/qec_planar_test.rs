use quantumforge::qec::surface_code_planar;

#[test]
fn planar_logicals_verify() {
    for d in [3usize, 5, 7] {
        let code = surface_code_planar(d);
        let n = code.n_physical;
        // face-only construction encodes n - (d-1)^2 = 2d-1 logicals
        assert_eq!(code.n_logical, 2 * d - 1, "d={} logical count", d);
        assert_eq!(code.logical_x.len(), code.n_logical);
        assert_eq!(code.logical_z.len(), code.n_logical);
        assert_eq!(code.stabilizers.len(), (d - 1) * (d - 1));
        // logicals commute with all stabilizers
        for (i, lx) in code.logical_x.iter().enumerate() {
            for s in &code.stabilizers {
                let ac = (0..n)
                    .filter(|&q| {
                        s.paulis[q] != 0 && lx.paulis[q] != 0 && s.paulis[q] != lx.paulis[q]
                    })
                    .count();
                assert!(
                    ac.is_multiple_of(2),
                    "d={} logical_x {} anticommutes with stabilizer",
                    d,
                    i
                );
            }
        }
        for (i, lz) in code.logical_z.iter().enumerate() {
            for s in &code.stabilizers {
                let ac = (0..n)
                    .filter(|&q| {
                        s.paulis[q] != 0 && lz.paulis[q] != 0 && s.paulis[q] != lz.paulis[q]
                    })
                    .count();
                assert!(
                    ac.is_multiple_of(2),
                    "d={} logical_z {} anticommutes with stabilizer",
                    d,
                    i
                );
            }
        }
        // logical pairs anticommute with each other
        for i in 0..code.n_logical {
            let lx = &code.logical_x[i];
            let lz = &code.logical_z[i];
            let ac = (0..n)
                .filter(|&q| lx.paulis[q] != 0 && lz.paulis[q] != 0 && lx.paulis[q] != lz.paulis[q])
                .count();
            assert_eq!(ac % 2, 1, "d={} logical pair {} does not anticommute", d, i);
        }
        // all logicals nonzero
        for l in &code.logical_x {
            assert!(!l.paulis.iter().all(|&p| p == 0));
        }
        for l in &code.logical_z {
            assert!(!l.paulis.iter().all(|&p| p == 0));
        }
    }
}

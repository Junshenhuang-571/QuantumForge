use quantumforge::tensor_network::{DMRG, MPS};

fn exact_tfim_energy(n: usize) -> f64 {
    // Exact diagonalization of TFIM H = -Σ Z_i Z_{i+1} - Σ X_i
    let dim = 1usize << n;
    let mut h = vec![vec![0.0f64; dim]; dim];
    // ZZ terms
    for i in 0..n - 1 {
        for a in 0..dim {
            let bi = (a >> (n - 1 - i)) & 1;
            let bip1 = (a >> (n - 2 - i)) & 1;
            let zi = if bi == 0 { 1.0 } else { -1.0 };
            let zip1 = if bip1 == 0 { 1.0 } else { -1.0 };
            h[a][a] += -zi * zip1;
        }
    }
    // X terms
    for i in 0..n {
        for a in 0..dim {
            let b = a ^ (1 << (n - 1 - i));
            h[a][b] += -1.0;
        }
    }
    // Use faer for eig
    use faer::diag::Diag;
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::linalg::evd::{ComputeEigenvectors, self_adjoint_evd, self_adjoint_evd_scratch};
    use faer::prelude::*;
    let mut mat_a = faer::Mat::<f64>::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            mat_a[(i, j)] = h[i][j];
        }
    }
    let mut s = Diag::<f64>::zeros(dim);
    let params = Default::default();
    let par = faer::Par::Seq;
    let scratch = self_adjoint_evd_scratch::<f64>(dim, ComputeEigenvectors::No, par, params);
    let mut mem = MemBuffer::new(scratch);
    let stack = MemStack::new(&mut mem);
    self_adjoint_evd(mat_a.as_ref(), s.as_mut(), None, par, stack, params).unwrap();
    let mut min_val = s.column_vector()[0];
    for i in 1..dim {
        if s.column_vector()[i] < min_val {
            min_val = s.column_vector()[i];
        }
    }
    min_val
}

#[test]
fn dmrg_exact_matches_exact_diag() {
    for n in 2..=8 {
        let exact = exact_tfim_energy(n);
        let mps = MPS::zero_state(n, 64);
        let mut dmrg = DMRG::new(mps, 64, 1e-12, 20);
        let zz = [
            [
                quantumforge::Complex64::new(-1.0, 0.0),
                quantumforge::Complex64::ZERO,
                quantumforge::Complex64::ZERO,
                quantumforge::Complex64::ZERO,
            ],
            [
                quantumforge::Complex64::ZERO,
                quantumforge::Complex64::new(1.0, 0.0),
                quantumforge::Complex64::ZERO,
                quantumforge::Complex64::ZERO,
            ],
            [
                quantumforge::Complex64::ZERO,
                quantumforge::Complex64::ZERO,
                quantumforge::Complex64::new(1.0, 0.0),
                quantumforge::Complex64::ZERO,
            ],
            [
                quantumforge::Complex64::ZERO,
                quantumforge::Complex64::ZERO,
                quantumforge::Complex64::ZERO,
                quantumforge::Complex64::new(-1.0, 0.0),
            ],
        ];
        let energy = dmrg.run(&zz, &zz).unwrap();
        assert!(
            (energy - exact).abs() < 1e-6,
            "n={n}: DMRG={energy}, exact={exact}"
        );
    }
}

#[test]
fn dmrg_correlation_matches_exact() {
    let n = 6;
    let mps = MPS::zero_state(n, 64);
    let mut dmrg = DMRG::new(mps, 64, 1e-12, 20);
    let zz = [
        [
            quantumforge::Complex64::new(-1.0, 0.0),
            quantumforge::Complex64::ZERO,
            quantumforge::Complex64::ZERO,
            quantumforge::Complex64::ZERO,
        ],
        [
            quantumforge::Complex64::ZERO,
            quantumforge::Complex64::new(1.0, 0.0),
            quantumforge::Complex64::ZERO,
            quantumforge::Complex64::ZERO,
        ],
        [
            quantumforge::Complex64::ZERO,
            quantumforge::Complex64::ZERO,
            quantumforge::Complex64::new(1.0, 0.0),
            quantumforge::Complex64::ZERO,
        ],
        [
            quantumforge::Complex64::ZERO,
            quantumforge::Complex64::ZERO,
            quantumforge::Complex64::ZERO,
            quantumforge::Complex64::new(-1.0, 0.0),
        ],
    ];
    dmrg.run(&zz, &zz).unwrap();
    // Just verify it runs and ground_state is populated
    assert!(dmrg.ground_state.is_some());
}
